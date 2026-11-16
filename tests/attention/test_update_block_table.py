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

"""update_block_table(build(table A), table B) equals build(table B), field by field."""

import dataclasses
from unittest.mock import Mock

import pytest
import torch
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder

BS = 128


def _builder(window):
    cfg = get_current_vllm_config()
    cfg.model_config.get_num_attention_heads = Mock(return_value=4)
    cfg.model_config.get_num_kv_heads = Mock(return_value=1)
    cfg.cache_config.block_size = BS
    cfg.model_config.max_model_len = 2048
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
            if f.name in ("metadata", "rows_dev"):
                continue
            _eq(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    else:
        assert a == b, (path, a, b)


SKIP = {"built_for", "mirror_cache", "slot_mapping"}


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
    b0 = _builder(window)
    b1 = _builder(window)
    g = torch.Generator().manual_seed(0)
    bt_a = torch.randint(1, 5000, (len(seq_lens), 16), dtype=torch.int32, generator=g)
    bt_b = torch.randint(1, 5000, (len(seq_lens), 16), dtype=torch.int32, generator=g)
    ca = _common(query_lens, seq_lens, bt_a)
    md_a = b0.build(0, ca)
    upd = b1.update_block_table(md_a, bt_b, ca.slot_mapping)
    ref = b1.build(0, _common(query_lens, seq_lens, bt_b))
    for f in dataclasses.fields(ref):
        if f.name in SKIP:
            continue
        _eq(getattr(upd, f.name), getattr(ref, f.name), f.name)
    assert not all(torch.equal(x, y) for x, y in zip(md_a.page_index_tables_cpu, upd.page_index_tables_cpu))


def test_window_mismatch_rebuilds(default_vllm_config):
    full, swa = _builder(None), _builder(512)
    bt = torch.randint(1, 5000, (2, 16), dtype=torch.int32)
    c = _common([1, 1], [1100, 1300], bt)
    md = full.build(0, c)
    got = swa.update_block_table(md, bt, c.slot_mapping)
    assert got.active_block_indices is not None
