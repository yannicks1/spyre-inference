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

"""Rebinding physical pages and slots equals a fresh build without mutating its source."""

import dataclasses
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder

BS = 128


def _builder(window):
    cfg = get_current_vllm_config()
    cfg.model_config.get_num_attention_heads = Mock(return_value=4)
    cfg.model_config.get_num_kv_heads = Mock(return_value=1)
    cfg.cache_config.block_size = BS
    cfg.model_config.max_model_len = 2048
    cfg.scheduler_config.max_num_seqs = 32
    if window is None:
        spec = FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=256, dtype=torch.float16)
    else:
        spec = SlidingWindowSpec(
            block_size=BS, num_kv_heads=1, head_size=256, dtype=torch.float16, sliding_window=window
        )
    return SpyreAttentionMetadataBuilder(spec, ["layers.0.self_attn"], cfg, torch.device("cpu"))


def _common(query_lens, seq_lens, bt):
    qsl = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
    n = int(qsl[-1])
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(seq_lens),
        num_actual_tokens=n,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=bt,
        slot_mapping=torch.arange(n, dtype=torch.int64),
        causal=True,
        is_prefilling=torch.tensor([q > 1 for q in query_lens]),
    )


def _eq(a, b, path):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b), path
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{path}[{i}]")
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            _eq(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    else:
        assert a == b, (path, a, b)


def _rebind_and_compare(b0, b1, common):
    source = b0.build(0, common)
    snapshot = deepcopy(source)
    other = dataclasses.replace(
        common,
        block_table_tensor=common.block_table_tensor + 5000,
        slot_mapping=common.slot_mapping + 7,
    )
    with patch.object(b1, "publish_slot_mapping", wraps=b1.publish_slot_mapping) as publish:
        rebound = b1.update_block_table(source, other.block_table_tensor, other.slot_mapping)
    publish.assert_called_once()
    assert publish.call_args.args[0] is other.slot_mapping
    _eq(rebound, b1.build(0, other), "rebound")
    _eq(source, snapshot, "source")
    assert rebound is not source
    assert rebound.attention_mask_stacks is source.attention_mask_stacks
    assert rebound.block_table is other.block_table_tensor
    assert rebound.slot_mapping is other.slot_mapping
    assert source.block_table is common.block_table_tensor
    assert source.slot_mapping is common.slot_mapping
    return source, rebound


@pytest.mark.parametrize("window", [None, 512])
@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [
        ([1] * 8, [1030 + 7 * i for i in range(8)]),
        ([1] * 4, [130, 700, 1500, 2000]),
        ([1, 1, 300], [900, 1200, 812]),
        ([512], [1536]),
    ],
)
def test_update_block_table_matches_build(default_vllm_config, window, query_lens, seq_lens):
    g = torch.Generator().manual_seed(0)
    bt = torch.randint(1, 5000, (len(seq_lens), 16), dtype=torch.int32, generator=g)
    source, rebound = _rebind_and_compare(
        _builder(window), _builder(window), _common(query_lens, seq_lens, bt)
    )
    assert not all(
        torch.equal(x, y)
        for x, y in zip(source.page_index_tables_cpu, rebound.page_index_tables_cpu)
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([1030] * 8, id="dense-8"),
        pytest.param([1030] * 16, id="dense-16"),
        pytest.param([1030] * 32, id="dense-32"),
        pytest.param([1030, 128] * 16, id="jagged-32"),
    ],
)
def test_dynamic_decode_update(default_vllm_config, monkeypatch, seq_lens):
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    monkeypatch.setenv("SPYRE_JAGGED_DECODE", "1")
    n = len(seq_lens)
    bt = torch.arange(n * 16, dtype=torch.int32).reshape(n, 16) + 1
    source, rebound = _rebind_and_compare(
        _builder(None), _builder(None), _common([1] * n, seq_lens, bt)
    )
    assert source.chunk_range is not None
    assert source.padded_batch_blocks < 16  # Dynamic extent, not the rounded KV bucket.
    if len(set(seq_lens)) > 1:
        assert rebound.decode_groups is not None and len(rebound.decode_groups) == 2
        assert any(
            g.chunk_range is not None and g.padded_batch_blocks < 16 for g in rebound.decode_groups
        )
        for group in rebound.decode_groups:
            for column, member in enumerate(group.rows_cpu.tolist()):
                real = (seq_lens[member] + BS - 1) // BS
                assert torch.equal(
                    group.chunk_page_ids_cpu[:real, column], rebound.block_table[member, :real]
                )
                assert not group.chunk_page_ids_cpu[real:, column].any()
            assert not group.chunk_page_ids_cpu[:, group.num_seqs :].any()


@pytest.mark.parametrize("query_len", [1, 3])
def test_short_table_page_padding(default_vllm_config, query_len):
    common = _common([query_len], [query_len], torch.tensor([[17]], dtype=torch.int32))
    source, rebound = _rebind_and_compare(_builder(None), _builder(None), common)
    assert source.page_range is not None
    pages = rebound.page_index_tables_cpu[0][:, 0]
    assert len(pages) > 1
    assert pages.tolist() == [5017] * len(pages)


@pytest.mark.parametrize("grouped", [False, True])
def test_rebinding_clears_populated_caches_without_mutating_source(default_vllm_config, grouped):
    lengths = [1030, 128] * 16 if grouped else [1030] * 8
    bt = torch.arange(len(lengths) * 16, dtype=torch.int32).reshape(len(lengths), 16) + 1
    common = _common([1] * len(lengths), lengths, bt)
    first, second = _builder(None), _builder(None)
    source = first.build(0, common)
    marker = torch.tensor([11], dtype=torch.int32)
    for name in ("kernel_index_tables", "query_row_tables", "attention_mask_stacks_device"):
        setattr(source, name, [marker])
    for name in ("rep_row_ids_dev", "chunk_page_ids_dev", "mask_by_chunk_dev"):
        setattr(source, name, marker)
    if grouped:
        assert source.decode_groups is not None
        for group in source.decode_groups:
            group.rows_dev = group.rows_cpu.clone()
            group.metadata = dataclasses.replace(source, decode_groups=None)
    snapshot = deepcopy(source)
    other = dataclasses.replace(
        common, block_table_tensor=bt + 100, slot_mapping=common.slot_mapping + 7
    )
    rebound = second.update_block_table(source, other.block_table_tensor, other.slot_mapping)
    _eq(rebound, second.build(0, other), "rebound")
    _eq(source, snapshot, "source")
    assert source.chunk_page_ids_dev is marker
    assert rebound.chunk_page_ids_dev is None
    if grouped:
        assert rebound.decode_groups is not None
        assert all(g.rows_dev is None and g.metadata is None for g in rebound.decode_groups)


@pytest.mark.parametrize(
    "encoder_only,causal_override", [(False, None), (True, None), (False, True), (False, False)]
)
def test_update_block_table_support(default_vllm_config, encoder_only, causal_override):
    builder = _builder(None)
    if encoder_only:
        spec = EncoderOnlyAttentionSpec(
            block_size=BS, num_kv_heads=1, head_size=256, dtype=torch.float16
        )
        builder = SpyreAttentionMetadataBuilder(
            spec, ["layers.0.self_attn"], get_current_vllm_config(), torch.device("cpu")
        )
    builder.causal_override = causal_override
    assert builder.supports_update_block_table is (not encoder_only and causal_override is None)
