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

"""Spyre OOT expert compute for unquantized FusedMoE.

The modular MoE kernels do not lower on Spyre (the OOT backend builds no
``moe_kernel``, so ``forward_native`` asserts). This evaluates every expert
densely with plain matmuls -- gather-free, so the whole path stays in matmul +
elementwise ops that lower -- mirroring hf-adapters' ``_moe_expert_persistent``.
Routing is done by the Gemma4 custom routing function, which hands over the
final per-expert combine.

Three properties carry the latency, measured per layer at gemma-4-26B-A4B
shapes (E=128, H=2816, I=704, T=1):

* **Weights are transposed once on the host, not per forward.** Upstream stores
  ``w13``/``w2`` in the orientation CUDA's kernels want, so the matmuls would
  need an in-graph ``transpose(1, 2)`` of a 507 MiB stack: 102 ms -> 26 ms.
* **The expert dim is placed outermost on device** (:func:`_to_device`):
  26 ms -> 12 ms.
* **The region is compiled.** vLLM routes the whole MoE through the opaque
  ``vllm.moe_forward`` custom op, so everything here runs *eagerly* even inside
  a compiled block -- one device kernel per op. On the default layout that
  barely matters for five big matmuls, but the expert-outermost layout only pays
  off compiled, and the routing chain needs it outright (3.4 ms -> 0.3 ms).

Together: 105 ms -> 12 ms per layer, which at 30 layers is essentially the whole
decode step for this model.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre
from vllm.config import CompilationMode, get_cached_compilation_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# The params we register in place of upstream's stacked w13/w2, in matmul order.
_EXPERT_PARAMS = ("gate_proj", "up_proj", "down_proj")

_COMPILED: dict[Callable, Callable] = {}

# Sampled while the vLLM config context is live (weight loading). Reading the
# config at forward time raises -- nothing sets the context around the model
# forward -- which is also why ``CompileOutermost`` samples it at construction.
_compile_enabled = True


def _to_device(weight: torch.Tensor) -> torch.Tensor:
    """Transfer a ``[E, C, F]`` expert stack with its expert dim outermost.

    ``E`` lands at device position 0 while ``F`` is split into sticks, which keeps
    the all-expert matmul streaming contiguously (and leaves ``E`` gatherable).
    Transferring here rather than letting ``model.to(device)`` do it is what makes
    the layout reachable at all: the transfer function ``_apply`` receives cannot
    carry a device layout, and the walker skips params already on device.

    Only worth it for a compiled expert region: run eagerly this layout is ~2x
    *slower* per layer than the default one (54 ms vs 26 ms at T=1), because each
    matmul then re-tiles on its own instead of the graph planning around it.
    """
    if not _compile_enabled:
        return weight.to("spyre")
    placed = dma_moe_expert_weight_to_spyre(weight)
    if placed is not None:
        return placed
    # Only when F does not span whole sticks; the helper warns with the details.
    logger.warning_once("MoE expert weights: default layout (slower dense matmul).")
    return weight.to("spyre")


def compiled_region(fn: Callable) -> Callable:
    """Compile ``fn`` once per process, so every MoE layer shares one artifact.

    Falls back to eager while another graph is tracing (nothing to gain: we would
    be inlined) and under ``enforce_eager``.
    """
    if torch.compiler.is_compiling() or not _compile_enabled:
        return fn
    compiled = _COMPILED.get(fn)
    if compiled is None:
        logger.info_once("Compiling the Spyre MoE regions as their own graphs.")
        # dynamic=False is mandatory: the Spyre backend rejects SymInt shapes.
        compiled = torch.compile(
            fn,
            backend=current_platform.simple_compile_backend,
            fullgraph=True,
            dynamic=False,
        )
        _COMPILED[fn] = compiled
    return compiled


def _dense_experts(
    x: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    gelu_tanh: bool,
) -> torch.Tensor:
    """Evaluate every expert and sum their routed outputs.

    ``route`` is the ``[T, E, 1]`` combine from the routing patch.
    """
    experts, tokens = gate.shape[0], x.shape[0]

    # Lead with E so the combine broadcasts against the [E, T, H] expert stack:
    # permuting that stack instead would copy the largest activation.
    route = route.permute(1, 0, 2).contiguous().clone()  # [E, T, 1]

    # xb must be materialized contiguous: an expanded view into the batched
    # matmul does not lower ("expected exactly 1 generated variable").
    xb = x.unsqueeze(0).expand(experts, tokens, -1).contiguous()
    gate_out = torch.matmul(xb, gate)  # [E, T, I]
    up_out = torch.matmul(xb, up)
    activated = (F.gelu(gate_out, approximate="tanh") if gelu_tanh else F.silu(gate_out)) * up_out
    expert_out = torch.matmul(activated, down)  # [E, T, H]
    return (expert_out * route.to(expert_out.dtype)).sum(dim=0)  # [T, H]


@UnquantizedFusedMoEMethod.register_oot(name="UnquantizedFusedMoEMethod")
class SpyreUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    @property
    def is_monolithic(self) -> bool:
        # Modular path: base is_monolithic would deref a None kernel.
        return False

    def process_weights_after_loading(self, layer: Any) -> None:
        """Split and transpose the expert weights on the host (see module docstring).

        Splitting also avoids an offset slice ``w13[:, I:, :]``, which is a
        stick-unaligned partial-stick start. The stacked params are
        de-registered, not emptied -- a ``[0]`` param cannot be stickified and
        would fail the transfer.
        """
        super().process_weights_after_loading(layer)

        global _compile_enabled
        _compile_enabled = get_cached_compilation_config().mode is not CompilationMode.NONE

        w13 = layer.w13_weight.data  # [E, 2I, H], on CPU (pre-transfer)
        inter = w13.shape[1] // 2

        for name, weight in (
            ("gate_proj", w13[:, :inter, :].transpose(1, 2)),  # [E, H, I]
            ("up_proj", w13[:, inter:, :].transpose(1, 2)),  # [E, H, I]
            ("down_proj", layer.w2_weight.data.transpose(1, 2)),  # [E, I, H]
        ):
            layer.register_parameter(
                name, torch.nn.Parameter(_to_device(weight.contiguous()), requires_grad=False)
            )
        del layer._parameters["w13_weight"]
        del layer._parameters["w2_weight"]

    def forward_oot(
        self,
        layer: Any,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: torch.nn.Module | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense-over-all-experts SwiGLU MoE for Spyre.

        ``topk_weights`` is the **dense** [T, E, 1] combine from the Spyre
        routing patch, not the usual [T, K] -- rebuilding [T, E] from a [T, K]
        selection needs a scatter Spyre cannot lower. ``topk_ids`` is unused.
        """
        if shared_experts is not None:
            raise NotImplementedError(
                "SpyreUnquantizedFusedMoEMethod does not support fused shared experts."
            )
        if not self.moe.is_act_and_mul:
            raise NotImplementedError(
                "SpyreUnquantizedFusedMoEMethod requires a gated (act-and-mul) expert MLP."
            )

        gate, up, down = (getattr(layer, name) for name in _EXPERT_PARAMS)

        # Fail loudly if the routing patch did not run: we cannot rebuild the
        # dense combine from a [T, K] selection here.
        if topk_weights.ndim != 3 or topk_weights.shape[1] != gate.shape[0]:
            raise NotImplementedError(
                "SpyreUnquantizedFusedMoEMethod expects a dense [T, E, 1] combine "
                f"from the Spyre routing patch, got shape {tuple(topk_weights.shape)}."
            )

        activation = getattr(layer.activation, "value", layer.activation)
        if activation not in ("gelu_tanh", "silu"):
            raise NotImplementedError(
                f"SpyreUnquantizedFusedMoEMethod: unsupported activation {activation!r}"
            )

        return compiled_region(_dense_experts)(
            x, topk_weights, gate, up, down, activation == "gelu_tanh"
        )
