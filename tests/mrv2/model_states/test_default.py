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

"""The decoder state's prepare_attn hands SpyreAttentionMetadataBuilder.build
every field it reads, with the values V1 gave it."""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder
from spyre_inference.v1.worker.spyre.model_states.default import SpyreDefaultModelState

from .state_fakes import make_config, make_input_batch


def _fields_read(*functions) -> set[str]:
    """Every ``common_attn_metadata.<field>`` the functions' source reads."""
    fields: set[str] = set()
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(function))))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "common_attn_metadata"
            ):
                fields.add(node.attr)
    return fields


# build() also hands the metadata to split_decodes_and_prefills.
_BUILDER_FIELDS = _fields_read(
    SpyreAttentionMetadataBuilder.build,
    split_decodes_and_prefills,
)


class _Recorder:
    supports_update_block_table = False  # read by build_attn_metadata

    def __init__(self):
        self.seen: list[CommonAttentionMetadata] = []

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.seen.append(common_attn_metadata)
        return SimpleNamespace()


def _prepare_attn(batch):
    state = SpyreDefaultModelState(
        make_config(),  # ty: ignore[invalid-argument-type]
        nn.Module(),
        None,
        torch.device("cpu"),
        torch.device("cpu"),
    )
    recorder = _Recorder()
    group = AttentionGroup(
        backend=None,  # ty: ignore[invalid-argument-type]
        layer_names=["layer.0"],
        kv_cache_spec=None,  # ty: ignore[invalid-argument-type]
        kv_cache_group_id=0,
        metadata_builders=[recorder],  # ty: ignore[invalid-argument-type]
    )
    block_tables = (torch.arange(batch.num_reqs * 4, dtype=torch.int32).view(batch.num_reqs, 4),)
    slot_mappings = torch.full((1, batch.num_tokens_after_padding), -1, dtype=torch.int64)
    attn_metadata = state.prepare_attn(
        batch,
        CUDAGraphMode.NONE,
        block_tables,
        slot_mappings,
        [[group]],  # ty: ignore[invalid-argument-type]
        # build_attn_metadata only counts the groups; KVCacheTensor differs across vLLM pins.
        SimpleNamespace(kv_cache_groups=[None]),  # ty: ignore[invalid-argument-type]
    )
    assert list(attn_metadata) == ["layer.0"]
    (common,) = recorder.seen
    return common, block_tables


def test_the_field_list_is_not_vacuous():
    assert {"seq_lens", "query_start_loc", "slot_mapping", "is_prefilling"} <= _BUILDER_FIELDS


@pytest.mark.parametrize("field", sorted(_BUILDER_FIELDS))
def test_every_field_the_builder_reads_is_set(field):
    common, _ = _prepare_attn(make_input_batch([1, 1, 5], num_computed=[10, 20, 0]))
    assert getattr(common, field) is not None


def test_values_match_the_step():
    """Decode rows first, then a prefill, the body bucketed from 7 to 8 tokens."""
    batch = make_input_batch([1, 1, 5], num_computed=[10, 20, 0], num_tokens_after_padding=8)
    common, block_tables = _prepare_attn(batch)

    assert common.num_reqs == 3
    # Real tokens only; the bucket pad past them is not a request.
    assert common.num_actual_tokens == 7
    # Padded, so the traced KV write keeps one shape per bucket.
    assert common.slot_mapping.shape == (8,)
    assert common.query_start_loc.tolist() == [0, 1, 2, 7]
    assert common.query_start_loc_cpu.tolist() == [0, 1, 2, 7]
    assert common.seq_lens.tolist() == [11, 21, 5]
    assert (common.max_query_len, common.max_seq_len) == (5, 21)
    assert common.block_table_tensor is block_tables[0]
    assert common.causal is True


def test_prepare_attn_reuses_same_spec_groups_only_within_step(default_vllm_config, monkeypatch):
    cfg = get_current_vllm_config()
    cfg.model_config.get_num_attention_heads = Mock(return_value=4)
    cfg.model_config.get_num_kv_heads = Mock(return_value=1)
    cfg.cache_config.block_size = 128
    cfg.model_config.max_model_len = 2048
    spec = SlidingWindowSpec(
        block_size=128, num_kv_heads=1, head_size=256, dtype=torch.float16, sliding_window=512
    )
    state = SpyreDefaultModelState(
        make_config(max_model_len=2048),  # ty: ignore[invalid-argument-type]
        nn.Module(),
        None,
        torch.device("cpu"),
        torch.device("cpu"),
    )
    builders = [
        SpyreAttentionMetadataBuilder(spec, [f"layer.{i}"], cfg, torch.device("cpu"))
        for i in range(3)
    ]
    updates, publications = [], []
    groups = []
    for i, builder in enumerate(builders):
        updates.append(Mock(wraps=builder.update_block_table))
        publications.append(Mock(wraps=builder.publish_slot_mapping))
        monkeypatch.setattr(builder, "update_block_table", updates[-1])
        monkeypatch.setattr(builder, "publish_slot_mapping", publications[-1])
        groups.append(
            [
                AttentionGroup(
                    backend=None,  # ty: ignore[invalid-argument-type]
                    layer_names=[f"layer.{i}"],
                    kv_cache_spec=spec,
                    kv_cache_group_id=i,
                    metadata_builders=[builder],
                )
            ]
        )
    batch = make_input_batch([1, 1, 64], num_computed=[1000, 1200, 600])
    tables = tuple(torch.arange(48, dtype=torch.int32).reshape(3, 16) + 100 * i for i in range(3))
    slots = torch.arange(3 * 66, dtype=torch.int64).reshape(3, 66)
    kv_config = SimpleNamespace(kv_cache_groups=[None] * 3)

    first_masks = None
    for invocation in range(2):
        metadata = state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            tables,
            slots,
            groups,
            kv_config,  # ty: ignore[invalid-argument-type]
        )
        assert [update.call_count for update in updates] == [0, invocation + 1, invocation + 1]
        masks = metadata["layer.0"].attention_mask_stacks
        assert masks is not first_masks
        for i in range(3):
            item = metadata[f"layer.{i}"]
            assert item.attention_mask_stacks is masks
            assert item.block_table is tables[i]
            assert torch.equal(item.slot_mapping, slots[i])
            assert publications[i].call_count == invocation + 1
            assert torch.equal(publications[i].call_args.args[0], slots[i])
            if i:
                assert item is not metadata["layer.0"]
                assert not torch.equal(
                    item.page_index_tables_cpu[0], metadata["layer.0"].page_index_tables_cpu[0]
                )
        first_masks = masks
