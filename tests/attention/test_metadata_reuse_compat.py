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

"""Remove with metadata_reuse_compat once upstream owns ordinary-attention reuse."""

from dataclasses import replace
from unittest.mock import patch, sentinel

import pytest
import torch
from test_update_block_table import _builder, _common, _eq
from vllm.config import get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata

from spyre_inference.v1.attention.metadata_reuse_compat import StepMetadata
from spyre_inference.v1.worker.spyre.model_states.default import SpyreDefaultModelState


@pytest.fixture
def common(default_vllm_config):
    return _common(
        [1, 1, 64], [1001, 1201, 664], torch.arange(48, dtype=torch.int32).reshape(3, 16)
    )


def test_cache_rebinds_physical_inputs_but_ignores_private_cached_fields(common):
    a, b = _builder(512), _builder(512)
    step = StepMetadata()
    first = a.build(0, common, **step.get_extra_attn_kwargs(a, common.num_reqs))
    other = replace(
        common,
        block_table_tensor=common.block_table_tensor + 100,
        slot_mapping=common.slot_mapping + 7,
    )
    common.compute_num_computed_tokens()
    with patch.object(b, "update_block_table", wraps=b.update_block_table) as update:
        second = b.build(0, other, **step.get_extra_attn_kwargs(b, other.num_reqs))
    update.assert_called_once()
    assert second.attention_mask_stacks is first.attention_mask_stacks
    _eq(second, b.build(0, other), "rebound")


@pytest.mark.parametrize(
    "field",
    [
        "seq_lens",
        "query_start_loc",
        "query_start_loc_cpu",
        "is_prefilling",
        "max_seq_len",
        "causal",
    ],
)
def test_cache_requires_same_batch_inputs(common, field):
    a, b = _builder(512), _builder(512)
    cache = StepMetadata().builds
    first = a.build(0, common, _spyre_step_cache=cache)
    value = getattr(common, field)
    changed = (
        value.clone()
        if isinstance(value, torch.Tensor)
        else (not value if field == "causal" else value + 1)
    )
    other = replace(common, **{field: changed})
    with patch.object(b, "update_block_table", wraps=b.update_block_table) as update:
        second = b.build(0, other, _spyre_step_cache=cache)
    update.assert_not_called()
    assert second.attention_mask_stacks is not first.attention_mask_stacks
    _eq(second, b.build(0, other), "fresh")


@pytest.mark.parametrize(
    "difference",
    [
        "type",
        "spec",
        "window",
        "heads",
        "causal_override",
        "reorder",
        "prefix",
        "fast_build",
        "per_seq_only",
    ],
)
def test_cache_requires_compatible_build(common, difference):
    a = _builder(512)
    b = _builder(None if difference == "spec" else 256 if difference == "window" else 512)
    prefix, kwargs = 0, {}
    if difference == "type":

        class OtherBuilder(type(b)):
            pass

        b = OtherBuilder(b.kv_cache_spec, [], get_current_vllm_config(), torch.device("cpu"))
    elif difference == "heads":
        b.num_heads *= 2
    elif difference == "causal_override":
        b.causal_override = False
    elif difference == "reorder":
        b.reorder_batch_threshold = 2
    elif difference == "prefix":
        prefix = 1
    elif difference in ("fast_build", "per_seq_only"):
        kwargs[difference] = True
    cache = StepMetadata().builds
    first = a.build(0, common, _spyre_step_cache=cache)
    with patch.object(b, "update_block_table", wraps=b.update_block_table) as update:
        second = b.build(prefix, common, _spyre_step_cache=cache, **kwargs)
    update.assert_not_called()
    assert second.attention_mask_stacks is not first.attention_mask_stacks
    _eq(second, b.build(prefix, common, **kwargs), "fresh")
    if difference == "per_seq_only":
        assert second.padded_num_seqs is None
        assert (
            b.build(0, common, _spyre_step_cache=cache).attention_mask_stacks
            is first.attention_mask_stacks
        )


def test_provider_only_supplies_spyre_builders(default_vllm_config):
    step = StepMetadata()
    builder = _builder(None)
    assert step.get_extra_attn_kwargs(attn_metadata_builder=builder, num_reqs=1) == {
        "_spyre_step_cache": step.builds
    }
    assert step.get_extra_attn_kwargs(object(), 1) == {}
    builder.causal_override = False
    assert step.get_extra_attn_kwargs(builder, 1) == {}


def test_failed_build_does_not_populate_cache(common):
    builder = _builder(512)
    cache = StepMetadata().builds
    with (
        patch.object(builder, "publish_slot_mapping", side_effect=RuntimeError("build failure")),
        pytest.raises(RuntimeError, match="build failure"),
    ):
        builder.build(0, common, _spyre_step_cache=cache)
    assert cache == {}
    recovered = builder.build(0, common, _spyre_step_cache=cache)
    assert (
        builder.build(0, common, _spyre_step_cache=cache).attention_mask_stacks
        is recovered.attention_mask_stacks
    )


@pytest.mark.parametrize(
    "mode,capture,existing",
    [
        (CUDAGraphMode.NONE, False, False),
        (CUDAGraphMode.PIECEWISE, False, False),
        (CUDAGraphMode.NONE, True, False),
        (CUDAGraphMode.FULL, False, False),
        (CUDAGraphMode.NONE, False, True),
    ],
)
def test_prepare_attn_preserves_provider_and_capture_paths(mode, capture, existing):
    state = object.__new__(SpyreDefaultModelState)
    provider = ModelSpecificAttnMetadata() if existing else None
    args = (sentinel.batch, mode, sentinel.tables, sentinel.slots, sentinel.groups, sentinel.config)
    with patch.object(DefaultModelState, "prepare_attn", return_value=sentinel.result) as upstream:
        result = state.prepare_attn(
            *args, for_capture=capture, ubatch_idx=1, model_specific_attn_metadata=provider
        )
    assert result is sentinel.result
    assert upstream.call_args.args == args
    forwarded = upstream.call_args.kwargs
    assert forwarded["for_capture"] is capture
    assert forwarded["ubatch_idx"] == 1
    used = forwarded["model_specific_attn_metadata"]
    if existing or capture or mode == CUDAGraphMode.FULL:
        assert used is provider
    else:
        assert isinstance(used, StepMetadata)
        assert used.builds == {}


def test_reentrant_prepare_and_exception_keep_caches_call_local(common):
    state = object.__new__(SpyreDefaultModelState)
    a, b = _builder(512), _builder(512)
    providers, builds = [], []
    args = (None, CUDAGraphMode.NONE, None, None, None, None)

    def upstream(*args, model_specific_attn_metadata, **kwargs):
        step = model_specific_attn_metadata
        providers.append(step)
        metadata = a.build(0, common, **step.get_extra_attn_kwargs(a, common.num_reqs))
        builds.append(metadata)
        if len(builds) == 1:
            with pytest.raises(RuntimeError, match="nested failure"):
                state.prepare_attn(*args)
            rebound = b.build(0, common, **step.get_extra_attn_kwargs(b, common.num_reqs))
            assert rebound.attention_mask_stacks is metadata.attention_mask_stacks
        elif len(builds) == 2:
            raise RuntimeError("nested failure")
        return metadata

    with patch.object(DefaultModelState, "prepare_attn", side_effect=upstream):
        state.prepare_attn(*args)
        state.prepare_attn(*args)
    assert len({id(provider.builds) for provider in providers}) == 3
    assert len({id(metadata.attention_mask_stacks) for metadata in builds}) == 3
    outside = b.build(0, common)
    assert all(outside.attention_mask_stacks is not item.attention_mask_stacks for item in builds)
