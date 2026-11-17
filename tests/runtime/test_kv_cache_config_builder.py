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

"""SpyreKVCacheConfigBuilder grouping and placement, and how allocate_spyre_kv_caches binds it."""

from types import SimpleNamespace

import torch
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import (
    _pool_bytes_per_block_from_builder,
    get_kv_cache_groups_from_builder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from spyre_inference.v1.attention.backends import spyre_attn as spyre_attn_mod
from spyre_inference.v1.core import kv_cache_config_builder as builder_mod
from spyre_inference.v1.core.kv_cache_config_builder import SpyreKVCacheConfigBuilder, page_fold
from spyre_inference.v1.worker.kv_cache_setup import allocate_spyre_kv_caches


def _config(max_model_len: int = 4096) -> VllmConfig:
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        max_model_len=max_model_len, supports_kv_cache_grouping=False
    )
    vllm_config.cache_config.block_size = 128
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    return vllm_config


def _swa(num_kv_heads: int, head_size: int, window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=128,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
        sliding_window=window,
    )


def _full(num_kv_heads: int, head_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=128, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.float16
    )


def _five_to_one(num_layers: int, swa, full) -> dict:
    return {f"layers.{i}.attn": full if i % 6 == 5 else swa for i in range(num_layers)}


def _plan(specs: dict, num_blocks: int = 10):
    vllm_config = _config()
    builder = SpyreKVCacheConfigBuilder()
    groups = get_kv_cache_groups_from_builder(builder, vllm_config, dict(specs))
    pool = _pool_bytes_per_block_from_builder(builder, vllm_config, groups)
    return builder.get_kv_cache_config_from_groups(vllm_config, groups, num_blocks), pool


def test_one_page_shape_pools_like_upstream():
    """Gemma-3: 22 sliding and 4 global layers of one shape give 7 groups over 4 tensors."""
    swa, full = _swa(1, 256, 512), _full(1, 256)
    config, pool = _plan(_five_to_one(26, swa, full))

    assert len(config.kv_cache_groups) == 7
    assert len(config.kv_cache_tensors) == 4
    assert pool == 4 * swa.page_size_bytes
    group_of = {
        name: g for g, group in enumerate(config.kv_cache_groups) for name in group.layer_names
    }
    for tensor in config.kv_cache_tensors:
        assert tensor.layer_stride == 0
        assert tensor.size == 10 * swa.page_size_bytes
        # Layers sharing an allocation never share a block table.
        assert len({group_of[name] for name in tensor.layers}) == len(tensor.layers)
    assert sorted(n for t in config.kv_cache_tensors for n in t.layers) == sorted(group_of)


def test_larger_global_blocks_fold_into_the_sliding_page_shape():
    """Gemma-4: upstream's 256-token global block is one 8 x 128 x 256 sliding page."""
    swa, full = _swa(8, 256, 1024), _full(2, 512)
    config, pool = _plan(_five_to_one(30, swa, full))

    assert len(config.kv_cache_groups) == 6
    block_sizes = {
        type(g.kv_cache_spec).__name__: g.kv_cache_spec.block_size for g in config.kv_cache_groups
    }
    assert block_sizes == {"SlidingWindowSpec": 128, "FullAttentionSpec": 256}
    # One tensor per slot, as many bytes per block ID as upstream's pool.
    assert len(config.kv_cache_tensors) == 5
    assert pool == 5 * swa.page_size_bytes
    for tensor in config.kv_cache_tensors:
        assert tensor.block_stride == swa.page_size_bytes
        assert len(tensor.layers) == 6
        # The page's own (sliding) layers come first: allocation reads the page from it.
        assert int(tensor.layers[0].split(".")[1]) % 6 != 5
        assert int(tensor.layers[-1].split(".")[1]) % 6 == 5


def test_unfoldable_shapes_keep_the_block_size_and_split_by_shape(monkeypatch):
    """Larger blocks that fold into no page: groups at block 128, a tensor per shape."""
    monkeypatch.setattr(builder_mod, "page_fold", lambda spec, page: None)
    swa, full = _swa(8, 256, 1024), _full(2, 512)
    config, pool = _plan(_five_to_one(30, swa, full))

    assert {g.kv_cache_spec.block_size for g in config.kv_cache_groups} == {128}
    assert len(config.kv_cache_groups) == 6
    assert len(config.kv_cache_tensors) == 10
    assert pool == 5 * (swa.page_size_bytes + full.page_size_bytes)
    for tensor in config.kv_cache_tensors:
        specs = {"swa" if (int(n.split(".")[1]) % 6 != 5) else "full" for n in tensor.layers}
        assert len(specs) == 1
        assert len(tensor.layers) == (5 if specs == {"swa"} else 1)


def test_page_fold_counts_pages_and_page_heads():
    page = _swa(8, 256, 1024)
    assert page_fold(page, page) == (1, 1)
    assert page_fold(_full(2, 512).copy_with_new_block_size(256), page) == (2, 2)
    assert page_fold(_full(4, 256).copy_with_new_block_size(256), page) == (2, 1)
    # Fewer elements than the page, or a narrower head than its rows.
    assert page_fold(_full(1, 512).copy_with_new_block_size(256), page) is None
    assert page_fold(_full(16, 128), page) is None


def test_folded_layers_view_the_slot_pages_as_their_own_pages(monkeypatch):
    swa, full = _swa(8, 256, 1024), _full(2, 512)
    config, _ = _plan(_five_to_one(30, swa, full), num_blocks=6)
    calls = []

    def _fake_allocate_pages(cls, blocks, spec, device):
        calls.append(spec)
        shape = (blocks, spec.num_kv_heads, spec.block_size, spec.head_size)
        return spyre_attn_mod.SpyrePagedKVCache(torch.zeros(shape), torch.zeros(shape))

    monkeypatch.setattr(
        spyre_attn_mod.SpyreAttentionImpl, "allocate_pages", classmethod(_fake_allocate_pages)
    )
    impl = object.__new__(spyre_attn_mod.SpyreAttentionImpl)
    impl.head_split = 1
    static_ctx = {"layers.5.attn": SimpleNamespace(impl=impl)}

    caches = allocate_spyre_kv_caches(config, static_ctx, {}, torch.device("cpu"))

    assert len(calls) == 5
    assert all(spec.block_size == 128 and spec.num_kv_heads == 8 for spec in calls)
    assert tuple(caches["layers.0.attn"].k_pages.shape) == (6, 8, 128, 256)
    assert tuple(caches["layers.5.attn"].k_pages.shape) == (12, 4, 128, 256)
    assert caches["layers.5.attn"].k_pages.untyped_storage().data_ptr() == (
        caches["layers.0.attn"].k_pages.untyped_storage().data_ptr()
    )
    assert impl.head_split == 2


def test_full_attention_gives_every_layer_its_own_allocation():
    full = _full(8, 128)
    config, pool = _plan({f"layers.{i}.attn": full for i in range(32)})

    assert len(config.kv_cache_groups) == 1
    assert [t.layers for t in config.kv_cache_tensors] == [[f"layers.{i}.attn"] for i in range(32)]
    assert pool == 32 * full.page_size_bytes


def test_layers_of_one_tensor_are_bound_to_the_same_pages(monkeypatch):
    swa, full = _swa(1, 256, 512), _full(1, 256)
    config, _ = _plan(_five_to_one(26, swa, full))
    calls = []

    def _fake_allocate_pages(cls, blocks, spec, device):
        calls.append(spec)
        return object()

    monkeypatch.setattr(
        spyre_attn_mod.SpyreAttentionImpl, "allocate_pages", classmethod(_fake_allocate_pages)
    )

    caches = allocate_spyre_kv_caches(config, {}, {}, torch.device("cpu"))

    assert len(calls) == 4
    for tensor in config.kv_cache_tensors:
        assert len({id(caches[name]) for name in tensor.layers}) == 1
    assert len({id(c) for c in caches.values()}) == 4
