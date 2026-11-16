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

"""Attention-group window split and KV page allocation, shared by both model runners."""

from copy import copy
from dataclasses import replace
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyrePagedKVCache,
)
from spyre_inference.v1.core.kv_cache_config_builder import layer_spec

logger = init_logger(__name__)


def with_kv_sharing_layer_specs(
    kv_cache_config: KVCacheConfig, shared_layers: dict[str, str]
) -> KVCacheConfig:
    """Copy mapping each KV-sharing layer to its target's spec, for attention only."""
    if not shared_layers:
        return kv_cache_config
    groups = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            specs = dict(spec.kv_cache_specs)
            for layer, target in shared_layers.items():
                if target in specs:
                    specs[layer] = specs[target]
            group = replace(group, kv_cache_spec=replace(spec, kv_cache_specs=specs))
        groups.append(group)
    kv_cache_config = copy(kv_cache_config)
    kv_cache_config.kv_cache_groups = groups
    return kv_cache_config


def split_attn_groups_by_window(
    attn_groups: list[list[AttentionGroup]], layers: dict[str, Any]
) -> None:
    """Give each distinct per-layer sliding window its own attention group, in place."""
    # `FullAttentionSpec.merge` collapses a hybrid decoder onto the one window it finds,
    # clamping its full-attention layers too (gemma-3-1b-it: 512 on all 26). Spyre masks
    # per group, not per layer as upstream does, so the group is what has to split.
    for kv_cache_group_id, groups in enumerate(attn_groups):
        split_groups: list[AttentionGroup] = []
        for group in groups:
            spec = group.kv_cache_spec
            windows: dict[int | None, list[str]] = {}
            for layer_name in group.layer_names:
                window = getattr(layers.get(layer_name), "sliding_window", None)
                windows.setdefault(window, []).append(layer_name)
            # SlidingWindowSpec is single-window by construction; UniformTypeKVCacheSpecs
            # is already unwrapped per layer upstream.
            if not isinstance(spec, FullAttentionSpec) or len(windows) <= 1:
                split_groups.append(group)
                continue
            logger.info(
                "Split %d attention layers into %d groups by sliding window %s.",
                len(group.layer_names),
                len(windows),
                {w: len(n) for w, n in windows.items()},
            )
            for window, layer_names in windows.items():
                split_groups.append(
                    AttentionGroup(
                        group.backend,
                        layer_names,
                        replace(spec, sliding_window=window),
                        kv_cache_group_id,
                    )
                )
        attn_groups[kv_cache_group_id] = split_groups


def allocate_spyre_kv_caches(
    kv_cache_config: KVCacheConfig,
    static_ctx: dict[str, Any],
    shared_layers: dict[str, str],
    device: torch.device,
) -> dict[str, SpyrePagedKVCache]:
    """SpyrePagedKVCache pages per layer; the layers of a zero-stride KVCacheTensor share one."""
    spec_by_layer = {
        layer_name: layer_spec(group, layer_name)
        for group in kv_cache_config.kv_cache_groups
        for layer_name in group.layer_names
    }

    def allocate(layer_names: list[str]) -> SpyrePagedKVCache:
        # The layout belongs to the backend; a layer without an impl (fixture stubs) gets
        # the default decoder layout.
        impl_classes = {
            type(impl) if isinstance(impl, SpyreAttentionImpl) else SpyreAttentionImpl
            for impl in (getattr(static_ctx.get(name), "impl", None) for name in layer_names)
        }
        if len(impl_classes) != 1:
            raise ValueError(
                f"Layers {layer_names} share one KV cache tensor but have attention impls "
                f"{sorted(c.__name__ for c in impl_classes)}, whose page layouts may differ."
            )
        return impl_classes.pop().allocate_pages(
            kv_cache_config.num_blocks, spec_by_layer[layer_names[0]], device
        )

    kv_caches: dict[str, SpyrePagedKVCache] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if kv_cache_tensor.layer_stride == 0:
            pages = allocate(kv_cache_tensor.layers)
            kv_caches.update(dict.fromkeys(kv_cache_tensor.layers, pages))
        else:
            kv_caches.update({name: allocate([name]) for name in kv_cache_tensor.layers})

    for layer_name, target in shared_layers.items():
        kv_caches[layer_name] = kv_caches[target]
    return kv_caches
