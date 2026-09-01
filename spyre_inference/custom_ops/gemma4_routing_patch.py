# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre patches for Gemma4 MoE routing.

Two upstream seams do not lower on Spyre:

1. ``gemma4_routing_function_torch`` uses ``one_hot`` (scatter), ``gather`` and
   advanced indexing, none of which lower. The replacement masks the
   non-selected experts against a k-th-largest threshold instead, and returns
   the renormalized, scale-folded result as a *stick-carrying* ``[T, E, 1]``
   combine (hf-adapters' ``_moe_route_persistent_packed`` shape): rebuilding
   ``[T, E]`` from a ``[T, K]`` selection needs a scatter Spyre cannot express,
   and a bare ``[T, E]`` product has no legal trailing-stick layout for the
   expert combine to broadcast against.

2. ``CustomRoutingRouter._compute_routing`` casts weights to fp32 for CUDA. On
   Spyre fp16 and fp32 use different stick widths and the eager cast does not
   re-tile, so ``.to(float32)`` returns garbage; keep weights fp16 throughout.

hf-adapters masks with ``spyre::keep_by_index``, which is exact where the
threshold ties. We cannot: that op exists only as an Inductor lowering, and
routing has to survive eager (see :func:`_spyre_gemma4_routing_function_torch`).
Ties need equal probabilities to the last fp16 bit, and both forms then pick a
superset of the same k experts, so the masked form is equivalent in practice.
"""

from typing import Any, cast

import torch
from torch_spyre._C import get_elem_in_stick
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.models import gemma4 as _gemma4

from .fused_moe import compiled_region

logger = init_logger(__name__)

# The combine is widened to a full stick before being sliced back to [..., 1],
# so the expansion has a physical stick to live on.
STICK = get_elem_in_stick(torch.float16)

_IDENTITY: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _stick_identity(like: torch.Tensor) -> torch.Tensor:
    """Cached device-resident ``eye(STICK)`` used to place the combine on a stick."""
    key = (like.device, like.dtype)
    identity = _IDENTITY.get(key)
    if identity is None:
        identity = torch.eye(STICK, dtype=like.dtype).to(like.device)
        _IDENTITY[key] = identity
    return identity


def _route(
    gating_output: torch.Tensor,
    per_expert_scale: torch.Tensor,
    identity: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma4 routing from Spyre-lowerable ops, returning a ``[T, E, 1]`` combine.

    A single-row (T==1 decode) reduction cannot be materialized on device, so the
    whole chain runs on two identical rows and is sliced back at the very end.
    """
    probs = torch.nn.functional.softmax(gating_output, dim=-1)  # [T, E]
    tokens = probs.shape[0]
    rows = probs.expand(2, -1).contiguous() if tokens == 1 else probs

    topk_vals, topk_ids = torch.topk(rows, topk, dim=-1)  # [R, K]
    # k-th largest per row via amax: amin is unimplemented, min falls back to
    # CPU, and the offset-(k-1) slice is stick-unaligned.
    kth = -(-topk_vals).amax(dim=-1, keepdim=True)  # [R, 1]
    kept = rows * (rows >= kth).to(rows.dtype)  # [R, E]
    dense = kept / kept.sum(dim=-1, keepdim=True) * per_expert_scale.to(rows.dtype)

    # ReLU materializes the expansion; the identity matmul puts it on a stick.
    packed = torch.relu(dense.unsqueeze(-1).expand(-1, -1, STICK))
    return (packed @ identity)[:tokens, :, :1], topk_ids[:tokens]  # [T, E, 1]


def _spyre_gemma4_routing_function_torch(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route, compiled as its own graph at the one shape that lowers.

    vLLM runs the MoE inside the opaque ``vllm.moe_forward`` custom op, so this
    chain is eager unless we compile it ourselves -- worth 3.4 ms -> 0.3 ms per
    layer, i.e. ~0.1 s per decoded token at 30 layers.

    Only the two-row decode shape compiles: at a wider prefill the top-k lands a
    reduction axis inside its stick (``topkvalue stick must not contain the
    reduction or k dimension``). Prefill therefore routes eagerly -- once per
    request, against once per token, so the split is worth its ugliness.
    """
    fn = _route if gating_output.shape[0] > 1 else compiled_region(_route)
    return fn(gating_output, per_expert_scale, _stick_identity(gating_output), topk)


def _spyre_compute_routing(
    self: CustomRoutingRouter,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``CustomRoutingRouter._compute_routing`` without the fp32 weight cast.

    ``topk_ids`` also keeps its native dtype: ``spyre_topk`` materializes indices
    in the *input* dtype (fp16 floats standing in for indices), so the ``.to(int32)``
    upstream applies would be a device cast for nobody -- the dense expert compute
    reads only ``topk_weights``.
    """
    return self.custom_routing_function(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
    )


def _patch() -> None:
    # Rebinding a def-bound global / class method trips ty's invalid-assignment
    # ("implicit shadowing"); assign through cast(Any, ...) so the target is not
    # a re-narrowable local.
    current = getattr(_gemma4, "gemma4_routing_function_torch", None)
    if current is not None and not getattr(current, "_spyre_patched", False):
        _spyre_gemma4_routing_function_torch._spyre_patched = True
        cast(Any, _gemma4).gemma4_routing_function_torch = _spyre_gemma4_routing_function_torch
        logger.info("Patched gemma4_routing_function_torch for Spyre (dense [T,E,1]).")

    if not getattr(CustomRoutingRouter._compute_routing, "_spyre_patched", False):
        _spyre_compute_routing._spyre_patched = True
        cast(Any, CustomRoutingRouter)._compute_routing = _spyre_compute_routing
        logger.info("Patched CustomRoutingRouter._compute_routing for Spyre (fp16).")


_patch()
