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

"""Spyre adaptations for vLLM's Gemma-4 model.

Covers the dense variants (12B / 31B) and the E-variants (E2B / E4B), which add
per-layer embeddings (PLE) and KV-sharing across the trailing layers. The
KV-sharing side needs no model patch — ``Gemma4Attention`` already skips the K/V
work and the cache write on a sharing layer, and the KV-cache group bookkeeping is
handled in ``TorchSpyreModelRunner``. What needs patching here is PLE:

- ``Gemma4Model`` registers the embedding/PLE scalars as buffers but hands them to
  ``Gemma4SelfDecoderLayers`` as plain tensor *attributes*, which
  ``nn.Module.__setattr__`` does not track. ``model.to("spyre")`` then rebinds the
  parent's buffers and leaves these aliases pointing at the original CPU tensors,
  so the compiled ``embed_input_ids`` / PLE path feeds 0-d CPU tensors into
  Inductor, which has no notion of a live CPU graph input.
- ``get_per_layer_inputs`` guards its PLE-table gather with a boolean vocab-range
  mask that the Spyre backend cannot lower, and that is a provable no-op on every
  released Gemma 4 checkpoint.
- upstream hands each block a *view* into the PLE tensor, whose storage offset a
  compiled kernel ignores (torch-spyre#3770). The block is given the whole offset-0
  tensor and selects its own row in-graph instead — see ``_spyre_model_forward`` and
  ``_spyre_layer_forward``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

# Scalars `Gemma4Model` registers as buffers and then aliases onto
# `Gemma4SelfDecoderLayers` as plain attributes. `normalizer` is always present;
# the PLE ones are None on the dense variants.
_ALIASED_SCALARS = (
    "normalizer",
    "embed_scale_per_layer",
    "per_layer_input_scale",
    "per_layer_projection_scale",
)

# Bound in `install_spyre_patches`; the Spyre forwards below delegate to them.
_orig_model_forward = None
_orig_layer_forward = None


def force_text_backbone(engine_args: EngineArgs) -> None:
    """Default gemma-4 to its text-only backbone (it ships multimodal).

    Sets ``hf_overrides["architectures"]`` so ``create_model_config`` resolves
    ``Gemma4ForCausalLM`` instead of the multimodal default. Skipped when the user
    already pinned an architecture (dict or callable ``hf_overrides``).
    """
    ov = engine_args.hf_overrides
    user_arch = callable(ov) or (isinstance(ov, dict) and "architectures" in ov)
    if "gemma-4" in (engine_args.model or "").lower() and not user_arch and isinstance(ov, dict):
        overrides = cast("dict[str, Any]", ov)
        overrides["architectures"] = ["Gemma4ForCausalLM"]
        logger.info("gemma-4: loading text-only backbone Gemma4ForCausalLM.")


def _register_aliased_scalars(self) -> None:
    """Re-register the parent's aliased scalars as buffers of this module.

    Restores the documented intent of ``Gemma4Model``'s own ``register_buffer``
    calls — move with the model, interact with torch.compile — for the aliases
    ``Gemma4SelfDecoderLayers`` holds, without changing any embedding math. A
    device-side 0-d scalar lowers fine; a stale CPU one does not.
    """
    for name in _ALIASED_SCALARS:
        value = getattr(self, name, None)
        if not isinstance(value, torch.Tensor):
            # None on the dense variants (no PLE submodules).
            continue
        delattr(self, name)
        self.register_buffer(name, value, persistent=False)


def _spyre_get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
    """``Gemma4SelfDecoderLayers.get_per_layer_inputs`` without the vocab-range mask.

    Upstream clamps out-of-range ids to 0 before gathering from the PLE table, via
    ``logical_and(ids >= 0, ids < vocab_size_per_layer_input)`` + ``where``. The
    Spyre backend cannot lower a boolean result over an int32 operand, and the mask
    is dead weight anyway whenever ``vocab_size_per_layer_input >= vocab_size``:
    every id vLLM can produce is then in range by construction (E2B / E4B ship
    ``vocab_size_per_layer_input == vocab_size == 262144``, and even their
    multimodal placeholder ids fall inside it). Dropping it also spares the decode
    path two elementwise passes over the token ids.

    A checkpoint with a genuinely smaller PLE vocab keeps upstream's masked path.
    """
    if self.embed_tokens_per_layer is None:
        return None
    per_layer_embeds = self.embed_tokens_per_layer(input_ids)
    per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer
    return per_layer_embeds.reshape(
        *input_ids.shape,
        self.config.num_hidden_layers,
        self.hidden_size_per_layer_input,
    )


def _spyre_model_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds=None,
    per_layer_inputs=None,
    **kwargs,
):
    """``Gemma4Model.forward`` for the plain single-rank text path on Spyre.

    Specialized for one reason: upstream slices ``per_layer_inputs[:, layer_idx, :]``
    per layer, and under per-block compile that view becomes a block *argument*. A
    compiled kernel reads its arguments from offset 0, ignoring ``storage_offset``
    (torch-spyre#3770 — the limitation the attention backend works around for
    page-index tables and per-block K/V), so every layer would read layer 0's PLE
    values and the model would emit device garbage with no error and no fallback
    warning. Materializing each slice outside the graph instead costs a device dispatch
    per layer per step, and those dominate what is left of decode: profiled at 35 copies
    per forward, ~884 us each, ~31 ms of a ~74 ms E2B step. They are dispatch-bound, not
    bandwidth-bound — 884 us to copy 256 elements. Handing the block the whole offset-0
    tensor fixes both: ``Gemma4DecoderLayer.forward`` selects its own row in-graph.

    Everything upstream's forward does beyond this loop is delegated, not
    reimplemented: fast prefill, pipeline parallelism, multimodal ``inputs_embeds``
    and precomputed ``per_layer_inputs``, and Eagle's auxiliary hidden states all
    take the original path. Passing ``residual=None`` every iteration is exact, not a
    simplification — ``Gemma4DecoderLayer.forward`` overwrites ``residual`` with
    ``hidden_states`` on entry and always returns ``None`` for it, which is also why
    upstream's post-loop ``norm`` never sees a residual on this path.
    """
    if (
        self.fast_prefill_enabled
        or inputs_embeds is not None
        or intermediate_tensors is not None
        or per_layer_inputs is not None
        or getattr(self, "aux_hidden_state_layers", ())
        or self.start_layer != 0
        or self.end_layer != len(self.layers)
    ):
        return _orig_model_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            per_layer_inputs,
            **kwargs,
        )

    hidden_states = self.embed_input_ids(input_ids)
    ple = self.project_per_layer_inputs(hidden_states, self.get_per_layer_inputs(input_ids))
    if ple is not None:
        # Flatten [T, num_layers, ple_dim] -> [T, num_layers * ple_dim]. Free (the tensor
        # is contiguous, so this is an offset-0 view) and necessary: torch-spyre cannot
        # build a SpyreTensorLayout for the 3-D shape at a graph boundary
        # ("Incompatible host_size and dim_order" from _inductor/propagate_layouts.py,
        # the same 3-D layout trouble behind its RetileWarning on this tensor).
        ple = ple.reshape(ple.shape[0], -1)
    for layer in self.layers:
        hidden_states, _ = layer(positions, hidden_states, None, per_layer_input=ple, **kwargs)
    return self.norm(hidden_states)


def _spyre_layer_forward(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
    """``Gemma4DecoderLayer.forward`` that takes this layer's PLE row in-graph.

    ``_spyre_model_forward`` passes every layer's row packed into one
    ``[T, num_layers * ple_dim]`` tensor, so the slice happens here, inside the block's
    compiled region, where a view is safe (see torch-spyre#3770 in
    ``_spyre_model_forward``). An argument already ``ple_dim`` wide came from another
    caller that sliced it itself, so it is used as-is.

    ``narrow`` on the packed 2-D tensor rather than ``select`` on a 3-D one: torch-spyre
    could not lay out the 3-D shape at the graph boundary at all
    ("Incompatible host_size and dim_order"). Both offsets are static — each block
    compiles separately, so ``layer_idx`` is a graph constant.
    """
    ple_dim = self.hidden_size_per_layer_input
    if per_layer_input is not None and per_layer_input.shape[-1] != ple_dim:
        per_layer_input = per_layer_input.narrow(1, self.layer_idx * ple_dim, ple_dim)
    return _orig_layer_forward(
        self, positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs
    )


def install_spyre_patches() -> None:
    """Patch ``Gemma4SelfDecoderLayers`` for the Spyre compile path (idempotent)."""
    from vllm.model_executor.models.gemma4 import (
        Gemma4DecoderLayer,
        Gemma4Model,
        Gemma4SelfDecoderLayers,
    )

    if getattr(Gemma4SelfDecoderLayers, "_spyre_patched", False):
        return

    global _orig_model_forward, _orig_layer_forward
    _orig_model_forward = Gemma4Model.forward
    _orig_layer_forward = Gemma4DecoderLayer.forward
    Gemma4Model.forward = _spyre_model_forward  # ty: ignore[invalid-assignment]
    Gemma4DecoderLayer.forward = _spyre_layer_forward  # ty: ignore[invalid-assignment]

    orig_init = Gemma4SelfDecoderLayers.__init__
    orig_get_per_layer_inputs = Gemma4SelfDecoderLayers.get_per_layer_inputs

    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        _register_aliased_scalars(self)

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.vocab_size_per_layer_input < self.config.vocab_size:
            return orig_get_per_layer_inputs(self, input_ids)
        return _spyre_get_per_layer_inputs(self, input_ids)

    Gemma4SelfDecoderLayers.__init__ = __init__  # ty: ignore[invalid-assignment]
    Gemma4SelfDecoderLayers.get_per_layer_inputs = get_per_layer_inputs  # ty: ignore[invalid-assignment]
    Gemma4SelfDecoderLayers._spyre_patched = True
    logger.info(
        "Spyre: Gemma-4 embedding/PLE scalars registered as buffers so they follow the "
        "model to device, the PLE gather's vocab-range mask is skipped when it is a no-op, "
        "and each block selects its own per-layer PLE row inside its compiled graph."
    )
