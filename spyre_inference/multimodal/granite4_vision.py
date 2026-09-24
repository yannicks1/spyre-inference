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

"""Granite 4 Vision workarounds for Spyre.

The tower is vLLM's own ``SiglipVisionModel`` plus the BLIP-2 QFormer projectors
``granite4_vision.py`` builds on top of it, so the layers themselves already come
from the registries ``custom_ops/`` adapts. What is left are the shapes: a SigLIP
head_dim of 72 and a QFormer window of 16 queries over 64 keys are both coprime
with the 64-element stick, and neither can be laid out on device. Everything here
is a guarded, idempotent patch; ``apply()`` is the only entry point.
"""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import STICK, align_up

logger = init_logger(__name__)


def _host(t: torch.Tensor) -> torch.Tensor:
    """Detach a weight onto the host before reshaping or padding it.

    The padded weights below are built from strided slice-assignments, which lower
    silently wrong on a Spyre-resident tensor; vLLM has already moved the model when
    ``apply()`` runs, so each helper pulls its source here and moves the result back.
    """
    return convert(t.detach(), device="cpu")


def _set_weight_t(layer: nn.Module, weight_t: torch.Tensor, device: torch.device) -> None:
    """Install a rebuilt transposed weight (``[in, out]``) on a Spyre linear."""
    layer.weight = Parameter(convert(weight_t.contiguous(), device=device), requires_grad=False)


def _pad_qkv_heads(qkv: nn.Module, heads: int, orig: int, padded: int, device) -> None:
    """Widen a fused ``[q | k | v]`` projection's heads from ``orig`` to ``padded``.

    End-padding is enough here (SigLIP's tower has no rope): the zeroed query/key
    channels contribute nothing to the dot product, and the zeroed value channels
    land on the output columns ``_pad_out_proj_heads`` zeroes out in turn.
    """
    w = _host(cast(torch.Tensor, qkv.weight))  # [in, 3 * heads * orig]
    new = w.new_zeros(w.shape[0], 3, heads, padded)
    new[..., :orig] = w.view(w.shape[0], 3, heads, orig)
    _set_weight_t(qkv, new.reshape(w.shape[0], -1), device)

    if qkv.bias is not None:
        b = _host(cast(torch.Tensor, qkv.bias))
        new_b = b.new_zeros(3, heads, padded)
        new_b[..., :orig] = b.view(3, heads, orig)
        qkv.bias = Parameter(convert(new_b.reshape(-1), device=device), requires_grad=False)


def _pad_out_proj_heads(proj: nn.Module, heads: int, orig: int, padded: int, device) -> None:
    """Widen an output projection's per-head input block to match padded heads."""
    w = _host(cast(torch.Tensor, proj.weight))  # [heads * orig, out]
    new = w.new_zeros(heads, padded, w.shape[1])
    new[:, :orig] = w.view(heads, orig, w.shape[1])
    _set_weight_t(proj, new.reshape(heads * padded, w.shape[1]), device)


def _pad_vision_head_dim(tower: nn.Module, device: torch.device) -> int | None:
    """Pad the SigLIP tower's attention heads out to a stick-aligned head_dim.

    granite-vision's tower is 1152 hidden over 16 heads, so head_dim is 72: the
    ``[batch, heads, seq, 72]`` view SDPA needs has no device layout ("Unsupported
    coordinate expression 9*c1/8"), and padding the activations does not help because
    the unpadded view has to be materialized first. Widening the projections instead
    keeps every activation stick-aligned, and the attention scale still comes from the
    unpadded head_dim, which ``SiglipAttention`` passes to the attention layer explicitly.
    """
    layers = getattr(getattr(tower, "vision_model", tower), "encoder", None)
    layers = getattr(layers, "layers", None)
    if layers is None:
        return None

    first = layers[0].self_attn
    orig = first.head_dim
    padded = align_up(orig, STICK)
    if orig == padded:
        return None

    for layer in layers:
        attn = layer.self_attn
        heads = attn.num_heads_per_partition
        # The fused qkv is read as `[q | k | v]`, each head-major over `heads`.
        assert attn.attn.num_kv_heads == heads, "GQA tower: fused qkv is not 3 x heads"
        _pad_qkv_heads(attn.qkv_proj, heads, orig, padded, device)
        _pad_out_proj_heads(attn.out_proj, heads, orig, padded, device)
        # The scale stays at the unpadded head_dim; only the width changes.
        attn.attn.head_size = padded
        attn.head_dim = padded

    logger.info(
        "Spyre: padded the Granite 4 Vision SigLIP head_dim %d -> %d across %d layers.",
        orig,
        padded,
        len(layers),
    )
    return padded


@lru_cache(maxsize=4)
def _padded_key_mask(
    keys: int, keys_padded: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Additive ``[1, 1, 1, keys_padded]`` mask hiding the padded keys."""
    mask = torch.zeros(1, 1, 1, keys_padded, dtype=dtype)
    mask[..., keys:] = torch.finfo(dtype).min
    return convert(mask, device=device)


def _window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Attention over ``[B, H, L, D]`` with both sequence axes padded to the stick.

    The QFormer windows are 16 queries over 64 keys, and a 16-long axis is coprime with
    the stick: the ``[..., 64, 16]`` transposed key operand alone has no device layout.
    Padded queries produce rows that are cropped off, and padded keys are masked, so
    this is the upstream ``softmax(q @ kT * scale) @ v`` on aligned shapes.
    """
    queries, keys = q.shape[-2], k.shape[-2]
    q_pad, k_pad = align_up(queries, STICK), align_up(keys, STICK)
    q = F.pad(q, (0, 0, 0, q_pad - queries)) if q_pad != queries else q.contiguous()
    if k_pad != keys:
        k = F.pad(k, (0, 0, 0, k_pad - keys))
        v = F.pad(v, (0, 0, 0, k_pad - keys))
    else:
        k, v = k.contiguous(), v.contiguous()

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if k_pad != keys:
        scores = scores + _padded_key_mask(keys, k_pad, scores.dtype, scores.device)
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out[:, :, :queries] if q_pad != queries else out


def patch_qformer_attention() -> None:
    """Run the QFormer projectors' attention on stick-aligned window shapes.

    Upstream builds the scores with a bare ``matmul`` on the 16-query windows, which
    cannot be laid out on device ("no mechanism to resolve stick incompatibility").
    """
    from vllm.model_executor.models import blip2

    cls = blip2.Blip2QFormerMultiHeadAttention
    if getattr(cls.forward, "_spyre_patched", False):
        return
    orig_forward = cls.forward

    def _forward(self, hidden_states, encoder_hidden_states=None):
        if hidden_states.device.type != "spyre":
            return orig_forward(self, hidden_states, encoder_hidden_states)
        kv_source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        context = _window_attention(
            self.transpose_for_scores(self.query(hidden_states)),
            self.transpose_for_scores(self.key(kv_source)),
            self.transpose_for_scores(self.value(kv_source)),
            self.scaling,
        )
        context = context.permute(0, 2, 1, 3).contiguous()
        return context.view(*context.size()[:-2], self.all_head_size)

    _forward._spyre_patched = True
    cls.forward = _forward
    logger.info("Spyre: QFormer projector attention runs on stick-padded window shapes.")


@lru_cache(maxsize=8)
def _pool_matrix(
    side: int, new_side: int, offset: tuple[int, int] | None, dtype: torch.dtype, device
) -> torch.Tensor:
    """``[new_side^2, side^2]`` left-multiplier pooling a patch grid down to `new_side`.

    ``offset=None`` averages each block (area interpolation at an integer ratio);
    an offset picks that one position out of it (spatial sampling). Built on the host:
    it is a constant, and the device cannot assemble it.
    """
    block = side // new_side
    m = torch.zeros(new_side * new_side, side * side, dtype=dtype)
    weight = 1.0 if offset else 1.0 / (block * block)
    taps = [offset] if offset else [(dh, dw) for dh in range(block) for dw in range(block)]
    for i in range(new_side):
        for j in range(new_side):
            for dh, dw in taps:
                m[i * new_side + j, (block * i + dh) * side + (block * j + dw)] = weight
    return convert(m, device=device)


def _pool_tokens(
    x: torch.Tensor, side: int, new_side: int, offset: tuple[int, int] | None
) -> torch.Tensor:
    """Pool `[B, side^2, C]` patch tokens down to `[B, new_side^2, C]` by one matmul."""
    return torch.matmul(_pool_matrix(side, new_side, offset, x.dtype, x.device), x)


def patch_downsamplers() -> None:
    """Pool the projectors' patch grids with a constant matmul instead of indexing.

    ``InterpolateDownsampler`` calls ``F.interpolate(mode="area")``, i.e.
    ``aten::_adaptive_avg_pool2d``, which has no Spyre kernel at all, and
    ``SpatialOffsetDownsampler`` picks one position per 2x2 block with a strided
    multi-axis index. Rewriting either as strided reads lowers *silently wrong* (the
    2x2 mean came back with ~100% error), so both become a left-multiply by a constant
    pooling matrix: one GEMM whose only reduction is over the stick-aligned token axis.
    """
    from vllm.model_executor.models import granite4_vision as upstream

    for cls, offset_of in (
        (upstream.InterpolateDownsampler, lambda self: None),
        (upstream.SpatialOffsetDownsampler, lambda self: (self.offset_h, self.offset_w)),
    ):
        if getattr(cls.__call__, "_spyre_patched", False):
            continue
        orig_call = cls.__call__

        def _call(self, image_features: torch.Tensor, _orig=orig_call, _offset=offset_of):
            side, new_side = self.orig_image_side, self.new_image_side
            if image_features.device.type != "spyre" or side % new_side:
                return _orig(self, image_features)
            return _pool_tokens(image_features, side, new_side, _offset(self))

        _call._spyre_patched = True
        cls.__call__ = _call

    logger.info("Spyre: Granite 4 Vision patch-grid downsamplers run as a constant matmul.")


@lru_cache(maxsize=32)
def _packed_row_index(
    patches: int, side: int, grid: tuple[int, int], image_size: tuple[int, int]
) -> torch.Tensor:
    """Row indices packing anyres tile features into one token sequence.

    Index ``patches * side**2`` addresses the ``image_newline`` row appended to the
    source, so a single gather emits the newline columns too. The geometry is upstream's:
    the same view/permute/flatten/``unpad_image`` runs here on an index grid rather than
    on the features, so the crop and the row order cannot drift from it.
    """
    from transformers.models.llava_next.modeling_llava_next import unpad_image

    tokens = side * side
    rows = torch.arange(patches * tokens, dtype=torch.int64).view(patches, tokens)
    newline = patches * tokens
    if patches == 1:
        return torch.cat([rows[0], torch.tensor([newline])])

    num_patch_height, num_patch_width = grid
    tiled = (
        rows[1:]
        .view(num_patch_height, num_patch_width, side, side)
        .permute(0, 2, 1, 3)
        .flatten(0, 1)
        .flatten(1, 2)
    )
    tiled = unpad_image(tiled.unsqueeze(0), image_size)[0]
    tiled = torch.cat([tiled, torch.full((tiled.shape[0], 1), newline)], dim=1)
    return torch.cat([rows[0], tiled.flatten()])


def patch_feature_packing() -> None:
    """Pack the anyres tile features with one gather instead of a 5-D permute.

    Upstream regroups ``[tiles, side, side, C]`` into one grid with
    ``permute(4, 0, 2, 1, 3).contiguous()``, a geometry-dependent multi-counter stick
    scatter the device cannot lay out ("within-stick coordinate 12*d2 + d4 carries 2 free
    symbols"). Since the whole function -- regroup, centre-crop, newline column, flatten --
    only ever permutes and selects whole feature rows, it becomes one ``index_select`` over
    the tile features with ``image_newline`` appended as one more row.
    """
    from vllm.model_executor.models import granite4_vision as upstream

    cls = upstream.Granite4VisionForConditionalGeneration
    method = cls._pack_and_unpad_image_features
    if getattr(method, "_spyre_patched", False):
        return

    def _pack(self, image_features, image_sizes):
        if not image_features or image_features[0].device.type != "spyre":
            return method(self, image_features, image_sizes)
        config = self.config
        side = int(
            (config.vision_config.image_size // config.vision_config.patch_size)
            * self._downsample_rate
        )
        packed = []
        for image_idx, feature in enumerate(image_features):
            patches, tokens, hidden = feature.shape
            assert tokens == side * side
            size = tuple(int(v) for v in image_sizes[image_idx])
            grid = upstream.get_anyres_image_grid_shape(
                image_sizes[image_idx],
                config.image_grid_pinpoints,
                config.vision_config.image_size,
            )
            rows = _packed_row_index(patches, side, tuple(int(g) for g in grid), size)
            newline_row = patches * tokens
            source = feature.reshape(patches * tokens, hidden)
            if self.image_newline is None:
                rows = rows[rows != newline_row]
            gathered = source.index_select(
                0, convert(rows.clamp(max=newline_row - 1), device=source.device)
            )
            if self.image_newline is not None:
                # Selected in rather than appended as one more source row: a `cat` whose
                # operand is a reshaped view of a 3-D device tensor reads that operand
                # wrong (silently, and `contiguous()` does not materialize it). Reading
                # `image_newline` directly also keeps a parameter `.to()` out of the
                # forward path.
                newline_at = convert((rows == newline_row).unsqueeze(-1), device=source.device)
                gathered = torch.where(newline_at, self.image_newline, gathered)
            packed.append(gathered)
        return packed

    _pack._spyre_patched = True
    cls._pack_and_unpad_image_features = _pack
    logger.info("Spyre: Granite 4 Vision anyres feature packing runs as one row gather.")


def patch_deepstack_transport() -> None:
    """Hand the deepstack features to the decoder directly instead of via CPU buffers.

    Upstream keeps one persistent ``max_num_batched_tokens x hidden`` buffer per level so
    a replayed CUDA graph reads features written just before it, fills them with a
    boolean-mask ``index_put`` and leaves them on the host until the first image arrives.
    On Spyre none of that works: there is no graph replay, ``aten::_index_put_impl_`` has
    no kernel (see ``custom_ops/multimodal_embeddings``), and a text-only prefill reaches
    the layer loop with host buffers, which is a device mismatch. So the placement becomes
    an ``index_select`` of the packed features plus a mask multiply, and the result is
    handed straight to the layer loop.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.models import granite4_vision as upstream
    from vllm.sequence import IntermediateTensors

    cls = upstream.Granite4VisionForConditionalGeneration
    if getattr(cls.forward, "_spyre_patched", False):
        return
    orig_embed = cls.embed_input_ids

    def _embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
        handle_oov_mm_token: bool = True,
    ):
        if input_ids.device.type != "spyre":
            return orig_embed(
                self,
                input_ids,
                multimodal_embeddings,
                is_multimodal=is_multimodal,
                handle_oov_mm_token=handle_oov_mm_token,
            )
        inner = self.language_model.model
        text_embeds = inner.embed_input_ids(input_ids)
        multiplier = inner.config.embedding_multiplier
        has_vision = (
            multimodal_embeddings is not None
            and is_multimodal is not None
            and len(multimodal_embeddings) > 0
        )
        self._spyre_deepstack = None
        if not has_vision:
            return text_embeds * multiplier

        device, dtype = text_embeds.device, text_embeds.dtype
        # One byte per token, and the row index it yields cannot be built on device:
        # cumsum over bool needs the int64 promotion torch-spyre cannot lower.
        mask = convert(is_multimodal, device="cpu")
        if not bool(mask.any()):
            return text_embeds * multiplier
        rows = convert((mask.cumsum(0) - 1).clamp(min=0), dtype=torch.int64, device=device)
        keep = convert((~mask).to(dtype).unsqueeze(-1), device=device)
        take = convert(mask.to(dtype).unsqueeze(-1), device=device)

        packed = torch.cat([t.to(dtype) for t in multimodal_embeddings], dim=0)
        # Rows at text positions gather an arbitrary image row, which `take` zeroes.
        gathered = packed.index_select(0, rows)
        hidden = text_embeds.shape[-1]
        self._spyre_deepstack = [level * take for level in gathered.split(hidden, dim=-1)]
        return text_embeds * keep * multiplier

    def _forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None
        features = getattr(self, "_spyre_deepstack", None)
        deepstack = None
        if features is not None and inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack = IntermediateTensors(
                {
                    f"ds_{llm_layer}": features[level]
                    for level, llm_layer in enumerate(self._ds_layer_indices)
                }
            )
        # Released here rather than in the next call, so a text-only request cannot pick
        # up the previous one's features.
        self._spyre_deepstack = None
        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack,
        )

    _forward._spyre_patched = True
    cls.embed_input_ids = _embed_input_ids
    cls.forward = _forward
    logger.info(
        "Spyre: Granite 4 Vision deepstack features are gathered on device and passed "
        "straight to the decoder (no persistent host buffers, no mask index_put)."
    )


def apply(model: nn.Module, device: torch.device) -> None:
    """Install every Granite 4 Vision workaround, in dependency order."""
    patch_downsamplers()
    patch_qformer_attention()
    patch_feature_packing()
    patch_deepstack_transport()
    # Never read once the transport above is installed, and sized at the whole token
    # budget per level.
    model._ds_buffers = []  # ty: ignore[invalid-assignment]
    tower = getattr(model, "vision_tower", None)
    if tower is not None:
        _pad_vision_head_dim(tower, device)
