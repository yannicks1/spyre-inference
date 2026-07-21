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

"""Alternative matmul (contract) Spyre RoPE kernel.

Same wiring as the canonical ``rotary_embedding`` module (shared ``_SpyreRotaryMixin``,
``gather_rotation``, rotation cache, and the ``spyre_rope_rot`` opaque op are reused
unchanged); only the on-device rotation differs. For the pad-to-stick case
(``head_size // 2 < 64``) this variant expands Q/K to the stick-aligned width, rotates,
then contracts back with a matmul (``x @ exp`` ... ``out @ con``) -- avoiding the
sub-stick strided slice + copy the canonical ``_rotate_neox_2x2`` uses. The two kernels
are numerically equivalent (the expand matmul zero-fills the pad lanes and the contract
discards them, so the canonical zero-padded cache is reused as-is; ported from
foundation-model-stack / hf-adapters ``apply_rope_matmul``).

This module registers nothing at import. The switch lives in ``custom_ops/__init__.py``
(``SPYRE_ROPE_MATMUL=1``), which calls ``use_matmul_rope()`` to swap these classes into
the OOT registry in place of the slice-based ones.
"""

from functools import lru_cache

import torch

from vllm.model_executor.custom_op import op_registry_oot

from .rotary_embedding import SpyreLlama3RotaryEmbedding, SpyreRotaryEmbedding
from .utils import convert


@lru_cache
def _get_expand_contract(
    inner: int, padded: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant expand/contract matrix pair that stick-aligns the neox halves.

    ``exp`` is the ``{0, 1}`` matrix [2*inner, 2*padded] that maps each neox half up to
    the stick-aligned ``padded`` on-device via ``x @ exp`` (zero-filling the pad lanes);
    ``con = exp.T`` [2*padded, 2*inner] contracts the rotated result back to the original
    width. Cached per ``(inner, padded, device, dtype)``.
    """
    e = torch.zeros(2 * inner, 2 * padded, dtype=dtype)
    idx = torch.arange(inner)
    e[idx, idx] = 1
    e[inner + idx, padded + idx] = 1
    return (
        convert(e, device=device, dtype=dtype),
        convert(e.t().contiguous(), device=device, dtype=dtype),
    )


def _apply_rope_matmul(
    x: torch.Tensor,
    rot: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Apply full neox RoPE via per-token 2x2 rotation matrices (contract variant).

    ``x`` is [T, H*head_size] or [T, H, head_size]; ``rot`` is [T, 2, 2, padded] with
    ``padded >= head_size // 2``. When head_size//2 is stick-aligned the split-half
    pairing is a pure view; otherwise Q/K are expanded to ``2*padded`` (``x @ exp``),
    rotated, then contracted back to head_size with ``out @ con`` -- so no sub-stick
    strided slice is materialized. Returns the rotated tensor with ``x``'s shape.
    """
    orig_shape = x.shape
    num_tokens = x.shape[0]
    inner = head_size // 2
    padded = rot.shape[-1]
    xv = x.view(num_tokens, -1, head_size)
    con: torch.Tensor | None = None
    if padded != inner:
        exp, con = _get_expand_contract(inner, padded, x.device, x.dtype)
        xv = xv @ exp  # [T, H, 2*padded]
    x_pairs = xv.reshape(num_tokens, -1, 2, padded)  # [T, H, 2, padded]
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)  # [T, H, 2, padded]
    out = out.flatten(-2)  # [T, H, 2*padded]
    if con is not None:
        out = out @ con  # contract back to [T, H, head_size]
    return out.reshape(orig_shape)


class _MatmulForwardMixin:
    """Overrides ``forward_oot`` to rotate via the contract matmul kernel.

    Everything else (``__init__``, ``gather_rotation``, ``_get_rotation_cache``,
    ``_rope_key``) is inherited from ``_SpyreRotaryMixin`` unchanged.
    """

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rot = torch.ops.vllm.spyre_rope_rot(
            positions,  # ty: ignore[invalid-argument-type]
            self._rope_key,  # ty: ignore[invalid-argument-type]
            self.head_size,
        )
        out_query = _apply_rope_matmul(query, rot, self.head_size)
        out_key = _apply_rope_matmul(key, rot, self.head_size) if key is not None else None
        return out_query, out_key


class SpyreRotaryEmbeddingMatmul(_MatmulForwardMixin, SpyreRotaryEmbedding):
    """Contract-kernel variant of SpyreRotaryEmbedding."""


class SpyreLlama3RotaryEmbeddingMatmul(_MatmulForwardMixin, SpyreLlama3RotaryEmbedding):
    """Contract-kernel variant of SpyreLlama3RotaryEmbedding."""


def use_matmul_rope() -> None:
    """Swap the matmul (contract) RoPE classes into the OOT registry in place of the
    slice-based ones that ``rotary_embedding`` registers at import. Idempotent; must run
    before the model is built (``get_rope`` resolves the OOT class from this registry)."""
    for name, cls in (
        ("RotaryEmbedding", SpyreRotaryEmbeddingMatmul),
        ("Llama3RotaryEmbedding", SpyreLlama3RotaryEmbeddingMatmul),
    ):
        cls.name = name  # mirror RotaryEmbeddingBase.register_oot
        op_registry_oot[name] = cls
