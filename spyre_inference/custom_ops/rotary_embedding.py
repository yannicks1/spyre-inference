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

"""Spyre OOT replacement for RotaryEmbedding.

Applies rotary position embeddings on the Spyre device via a complex-free 2x2
rotation-matrix formulation (ported from foundation-model-stack). The 2x2 rotation
cache is built once from the base ``cos_sin_cache`` and moved to Spyre off the
compiled path (``prime_device_cache``); ``forward_oot`` then gathers this pass's
per-token rotation slice with an **on-device** ``index_select`` inside a compiled
sub-forward (indirect access is supported on Spyre under torch.compile) and applies
the rotation. No per-token CPU gather or host->device transfer happens in the forward.

Only neox-style full rotary is supported; other configs raise
``NotImplementedError`` at construction instead of silently falling back to CPU.
"""

from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)
from vllm.utils.math_utils import round_up

from .utils import convert

logger = init_logger(__name__)

# Spyre stick size = 64 float16 elements. The 2x2 layout's inner dim is
# rotary_dim // 2; when that is not a stick multiple the split-half view has a
# sub-stick stride the inductor rejects, so it is padded up on-device.
_SPYRE_STICK = 64


def _to_gather_layout(cache_flat: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    """Move a flat ``[max_pos, W]`` cache (``W`` a stick multiple) to ``target_device``
    for on-device ``index_select``.

    On Spyre the tensor must use the explicit outer-dim-untiled / stick-innermost
    ``SpyreTensorLayout`` (``device_size=[max_pos, W//64, 64]``, ``stride_map=[W, 64, 1]``);
    the default device layout tiles the rows so a gather along dim 0 reads scrambled
    memory for any row wider than one stick. Non-Spyre devices (CPU math tests) keep the
    plain layout — eager ``index_select`` works there.
    """
    if torch.device(target_device).type != "spyre":
        return cache_flat.to(device=target_device)
    from torch.spyre import DataFormats, SpyreTensorLayout

    max_pos, width = cache_flat.shape
    stl = SpyreTensorLayout(
        device_size=[max_pos, width // _SPYRE_STICK, _SPYRE_STICK],
        stride_map=[width, _SPYRE_STICK, 1],
        device_dtype=DataFormats.SEN169_FP16,
    )
    return cache_flat.to("spyre", device_layout=stl)


@lru_cache
def _get_expand_matrix(
    inner: int, padded: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Constant ``{0, 1}`` matrix ``E`` [2*inner, 2*padded] that zero-pads each neox
    half up to the stick-aligned ``padded`` on-device via ``x @ E`` (so the sub-stick
    ``[.,2,inner]`` view is never materialized). Cached per ``(inner, padded, device, dtype)``.
    """
    e = torch.zeros(2 * inner, 2 * padded, dtype=dtype)
    idx = torch.arange(inner)
    e[idx, idx] = 1
    e[inner + idx, padded + idx] = 1
    return convert(e, device=device, dtype=dtype)


@lru_cache
def _get_compress_matrix(
    inner: int, padded: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Constant ``{0, 1}`` matrix ``C`` [2*padded, 2*inner] that extracts the meaningful
    ``inner`` elements of each padded neox half on-device via ``out @ C``.

    The inverse of ``_get_expand_matrix``: the pad path zero-pads to the stick-aligned
    ``padded`` before the rotation, then must drop the padding afterwards. A sub-stick
    ``out[..., :inner]`` slice makes the compiler emit a non-stick-aligned kernel (the
    scheduler aborts with "parameter value is multiple of stick size"), so the trim is
    done as a stick-aligned matmul instead. Cached per ``(inner, padded, device, dtype)``.
    """
    c = torch.zeros(2 * padded, 2 * inner, dtype=dtype)
    idx = torch.arange(inner)
    c[idx, idx] = 1
    c[padded + idx, inner + idx] = 1
    return convert(c, device=device, dtype=dtype)


def _rotate_neox_2x2(
    x: torch.Tensor,
    rot: torch.Tensor,
    head_size: int,
    expand_matrix: torch.Tensor | None,
    compress_matrix: torch.Tensor | None,
) -> torch.Tensor:
    """Apply full neox RoPE via per-token 2x2 rotation matrices.

    ``x`` is [T, H*head_size] or [T, H, head_size]; ``rot`` is [T, 2, 2, padded]
    with ``padded >= head_size // 2``. When the inner dim head_size//2 is stick-aligned
    ``expand_matrix``/``compress_matrix`` are ``None`` and the split-half pairing is a pure
    view; otherwise ``expand_matrix`` is the Spyre-resident ``{0, 1}`` matrix that zero-pads
    each half up to ``padded`` via ``x @ E`` (aligning the pairing-axis stride), and
    ``compress_matrix`` drops the padding again via ``out @ C`` (a stick-aligned matmul
    instead of a sub-stick slice). Both are precomputed on-device off the compiled path
    (see ``_SpyreRotaryMixin.prime_device_cache``). Returns the rotated tensor with ``x``'s shape.
    """
    num_tokens = x.shape[0]
    inner = head_size // 2
    padded = rot.shape[-1]
    if expand_matrix is not None:
        x_pairs = (x.view(num_tokens, -1, head_size) @ expand_matrix).view(
            num_tokens, -1, 2, padded
        )
    else:
        x_pairs = x.view(num_tokens, -1, 2, inner)
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)
    if compress_matrix is not None:
        out = out.reshape(num_tokens, -1, 2 * padded) @ compress_matrix
    return out.reshape(x.shape)


class _SpyreRotaryMixin:
    """Spyre RoPE wiring shared by the base, llama3, and yarn OOT classes.

    Runs the 2x2 rotation on Spyre for supported configs; unsupported configs raise
    ``NotImplementedError`` at construction. The rotation cache is derived from the base
    ``cos_sin_cache`` (inheriting all rope-scaling variants) and moved to Spyre once via
    ``prime_device_cache``; the per-token slice is then gathered on-device with
    ``index_select`` inside the compiled ``_forward_native``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only neox full rotary has a Spyre kernel; gptj/interleaved and partial
        # rotary are rejected here rather than run on CPU.
        if not (self.is_neox_style and self.rotary_dim == self.head_size):
            raise NotImplementedError(
                "SpyreRoPE supports only neox-style full rotary (rotary_dim == "
                f"head_size); got is_neox_style={self.is_neox_style}, "
                f"rotary_dim={self.rotary_dim}, head_size={self.head_size}."
            )
        inner = self.rotary_dim // 2
        self._padded_inner = round_up(inner, _SPYRE_STICK)
        self._needs_expand = self._padded_inner != inner
        # CPU 2x2 rotation cache, lazily built from cos_sin_cache.
        self._rotation_cache: torch.Tensor | None = None
        # Spyre-resident copies, primed off the compiled path in prime_device_cache.
        self._rotation_cache_dev: torch.Tensor | None = None
        self._expand_matrix: torch.Tensor | None = None
        self._compress_matrix: torch.Tensor | None = None
        # index_select is only available under torch.compile; compile the sub-forward
        # so it works even in enforce_eager and standalone (mirrors SpyreSiluAndMul).
        # Under fullgraph model compile the _forward runs inside that graph anyway.
        if not torch.compiler.is_dynamo_compiling():
            self._forward = torch.compile(self._forward_native, dynamic=False)
        else:
            self._forward = self._forward_native

    def _apply(self, fn, recurse=True):
        # cos_sin_cache has no Spyre kernel; keep cos_sin_cache on CPU.
        return self

    def _get_rotation_cache(self) -> torch.Tensor:
        """Lazily build the CPU 2x2 rotation cache [max_pos, 2, 2, padded_inner] from
        cos_sin_cache ([[cos, -sin], [sin, cos]]), zero-padding the inner dim to the
        next stick multiple."""
        if self._rotation_cache is None:
            inner = self.rotary_dim // 2
            cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
            cache = torch.stack([cos, -sin, sin, cos], dim=1).view(
                self.cos_sin_cache.shape[0], 2, 2, inner
            )
            if self._padded_inner != inner:
                cache = torch.nn.functional.pad(cache, (0, self._padded_inner - inner))
            self._rotation_cache = cache
        return self._rotation_cache

    def prime_device_cache(self, target_device: torch.device) -> None:
        """Move the flat 2x2 rotation cache (and the expand matrix, if needed) to
        ``target_device`` once, off the compiled path. ``_forward_native`` then gathers
        this pass's per-token slice on-device with ``index_select`` and reshapes it back
        to ``[T, 2, 2, padded]``.

        The cache is stored flat ``[max_pos, 4*padded]`` and moved with the explicit
        outer-dim-untiled / stick-innermost Spyre layout (see ``_to_gather_layout``): the
        default device tiling makes a dim-0 gather read scrambled memory for rows wider
        than one stick.

        Re-primes if the cache currently lives on a different device: ``get_rope`` caches
        instances in ``_ROPE_DICT``, so a single instance may be primed for CPU (math
        tests) and later reused on Spyre (device tests). Gathering with positions on a
        different device than the cache raises a device-mismatch inside the compiled graph.
        """
        target_type = torch.device(target_device).type
        if (
            self._rotation_cache_dev is None
            or self._rotation_cache_dev.device.type != target_type
        ):
            cache = self._get_rotation_cache()  # [max_pos, 2, 2, padded]
            flat = cache.reshape(cache.shape[0], -1).contiguous()
            self._rotation_cache_dev = _to_gather_layout(flat, target_device)
        if self._needs_expand and (
            self._expand_matrix is None or self._expand_matrix.device.type != target_type
        ):
            self._expand_matrix = _get_expand_matrix(
                self.rotary_dim // 2, self._padded_inner, target_device, self.dtype
            )
            self._compress_matrix = _get_compress_matrix(
                self.rotary_dim // 2, self._padded_inner, target_device, self.dtype
            )

    def _forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compiled sub-forward: gather the per-token 2x2 rotation on-device and apply it.

        ``positions`` is 1-D Spyre int; the ``index_select`` gather runs on-device (indirect
        access is supported under torch.compile). ``query``/``key`` arrive on Spyre from the
        QKV projection; the flat cache and expand matrix are primed on Spyre off the compiled
        path. The gathered flat ``[T, 4*padded]`` slice is reshaped back to ``[T, 2, 2, padded]``.
        """
        rot = torch.index_select(self._rotation_cache_dev, 0, positions).view(
            positions.shape[0], 2, 2, self._padded_inner
        )
        out_query = _rotate_neox_2x2(
            query, rot, self.head_size, self._expand_matrix, self._compress_matrix
        )
        out_key = (
            _rotate_neox_2x2(
                key, rot, self.head_size, self._expand_matrix, self._compress_matrix
            )
            if key is not None
            else None
        )
        return out_query, out_key

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # torch-spyre crashes compiling an on-device index_select whose index has
        # length 1 (the decode phase): the gather output at leading dim 1 aborts the
        # scheduler. Pad the single token to 2 by duplicating the whole row, run the
        # gather+rotation, and slice back to 1. The pad must duplicate the full tensor
        # (``cat([x, x])``) rather than append a size-1 slice (``x[-1:]``): a size-1
        # slice on-device is itself an unsupported "stick expression 1". The trailing
        # ``[:1]`` slice of a length-2 result is fine (verified on torch-spyre 7073e9c).
        if positions.shape[0] == 1:
            positions = torch.cat([positions, positions], dim=0)
            query = torch.cat([query, query], dim=0)
            key = torch.cat([key, key], dim=0) if key is not None else None
            out_query, out_key = self._forward(positions, query, key)
            return out_query[:1], (out_key[:1] if out_key is not None else None)
        return self._forward(positions, query, key)


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(_SpyreRotaryMixin, RotaryEmbedding):
    """OOT RotaryEmbedding that applies the rotation on Spyre."""

    pass


@RotaryEmbeddingBase.register_oot(name="Llama3RotaryEmbedding")
class SpyreLlama3RotaryEmbedding(_SpyreRotaryMixin, Llama3RotaryEmbedding):
    """OOT Llama3RotaryEmbedding that applies the rotation on Spyre."""

    pass


@RotaryEmbeddingBase.register_oot(name="YaRNScalingRotaryEmbedding")
class SpyreYaRNScalingRotaryEmbedding(_SpyreRotaryMixin, YaRNScalingRotaryEmbedding):
    """OOT YaRNScalingRotaryEmbedding that applies the rotation on Spyre."""

    pass
