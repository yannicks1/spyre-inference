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

"""KV cache placement for Spyre's hybrid KV cache manager.

Upstream overlays layer ``l`` of every group on one buffer viewed at each layer's shape; a
Spyre tensor cannot be re-viewed at another stick width (torch-spyre#3770), so those layers
share one tensor per page shape instead (a ``KVCacheTensor`` with ``layer_stride=0``).

A layer whose blocks are larger than the slot's smallest folds into that slot's pages when a
block holds as many elements as a page: with ``k * block_size``-token blocks of ``H`` heads
of ``r * d``, its block ``b`` is pages ``k * b .. k * b + k - 1`` of the slot tensor viewed as
``[k * num_blocks, H * r, block_size, d]``, page head ``e * H + h`` holding dims
``[e * d, (e + 1) * d)`` of head ``h`` (gemma-4: a 256-token global block of 2 heads x 512
is one 128-token sliding page of 8 heads x 256). Only outer dims are re-viewed.
"""

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _get_kv_cache_groups_uniform_page_size,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)


def layer_spec(group: KVCacheGroupSpec, layer_name: str) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name]
    return spec


def _page_shape(spec: KVCacheSpec, layer_name: str) -> tuple:
    """What a Spyre page allocation depends on; layers agreeing on it can share one."""
    if isinstance(spec, AttentionSpec):
        return (
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
            getattr(spec, "head_size_v", spec.head_size),
            spec.dtype,
        )
    # Spyre allocates attention pages only; anything else keeps an allocation per layer.
    return (layer_name,)


def page_bytes(spec: KVCacheSpec) -> int:
    """A Spyre page is unpadded, where upstream pads a Mamba state to the shared page."""
    return getattr(spec, "real_page_size_bytes", spec.page_size_bytes)


def page_fold(spec: KVCacheSpec, page: KVCacheSpec) -> tuple[int, int] | None:
    """``(token_split, head_split)`` storing ``spec``'s blocks in ``page``-shaped pages, or None.

    ``spec`` folds when its block is ``token_split`` pages and each head ``head_split`` page
    heads, so the pages hold exactly its elements; ``(1, 1)`` is ``page`` itself.
    """
    if not (isinstance(spec, AttentionSpec) and isinstance(page, AttentionSpec)):
        return None
    head_size_v = getattr(spec, "head_size_v", spec.head_size)
    page_head_size_v = getattr(page, "head_size_v", page.head_size)
    if (
        spec.dtype != page.dtype
        or head_size_v != spec.head_size
        or page_head_size_v != page.head_size
        or spec.block_size % page.block_size
        or spec.head_size % page.head_size
    ):
        return None
    token_split = spec.block_size // page.block_size
    head_split = spec.head_size // page.head_size
    if spec.num_kv_heads * token_split * head_split != page.num_kv_heads:
        return None
    return token_split, head_split


def spyre_kv_placement(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> list[tuple[KVCacheSpec, list[str]]]:
    """``(spec, layers)`` per Spyre allocation: layer ``l`` of every group, split by page shape.

    ``spec`` is the page's own; a layer with larger blocks joins the first page of its slot
    it folds into (``page_fold``).
    """
    by_slot: dict[int, list[tuple[str, KVCacheSpec]]] = {}
    for group in kv_cache_groups:
        if group.host_resident:
            continue
        for slot, layer_name in enumerate(group.layer_names):
            by_slot.setdefault(slot, []).append((layer_name, layer_spec(group, layer_name)))

    placement: list[tuple[KVCacheSpec, list[str]]] = []
    for layers in by_slot.values():
        block_sizes = [s.block_size for _, s in layers if isinstance(s, AttentionSpec)]
        smallest = min(block_sizes, default=None)
        pages: dict[tuple, tuple[KVCacheSpec, list[str]]] = {}
        folded: list[tuple[str, KVCacheSpec]] = []
        for layer_name, spec in layers:
            if isinstance(spec, AttentionSpec) and spec.block_size != smallest:
                folded.append((layer_name, spec))
                continue
            pages.setdefault(_page_shape(spec, layer_name), (spec, []))[1].append(layer_name)
        for layer_name, spec in folded:
            target = next(
                (entry for entry in pages.values() if page_fold(spec, entry[0]) is not None),
                None,
            )
            if target is None:
                target = pages.setdefault(_page_shape(spec, layer_name), (spec, []))
            target[1].append(layer_name)
        placement += pages.values()
    return placement


def folds_into_slot_pages(kv_cache_groups: list[KVCacheGroupSpec], block_size: int) -> bool:
    """Whether every layer is stored in ``block_size``-token pages under ``spyre_kv_placement``."""
    for page, layers in spyre_kv_placement(kv_cache_groups):
        if not isinstance(page, AttentionSpec) or page.block_size != block_size:
            return False
        specs = {
            name: layer_spec(group, name)
            for group in kv_cache_groups
            for name in group.layer_names
            if name in layers
        }
        if any(page_fold(spec, page) is None for spec in specs.values()):
            return False
    return True


class SpyreKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Upstream's grouping, materialised as one Spyre tensor per page shape and slot."""

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        groups = super().get_kv_cache_groups(vllm_config, kv_cache_spec)
        block_size = vllm_config.cache_config.block_size
        if not all(isinstance(spec, AttentionSpec) for spec in kv_cache_spec.values()) or all(
            layer_spec(group, name).block_size == block_size
            for group in groups
            for name in group.layer_names
        ):
            return groups
        # Upstream grows the smaller pages' block size to unify page sizes. Spyre attention
        # splits a larger block into block_size pages, at upstream's footprint if it folds.
        if folds_into_slot_pages(groups, block_size):
            logger.info_once(
                "Storing the KV cache's larger blocks as %d-token pages of their slots' "
                "page shapes.",
                block_size,
            )
            return groups
        logger.info_once(
            "Grouping the KV cache at block_size=%d with per-shape pages: the Spyre "
            "attention backend runs one block size.",
            block_size,
        )
        return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
    ) -> KVCacheConfig:
        if not kv_cache_groups:
            return super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        kv_cache_tensors = [
            KVCacheTensor(
                size=page_bytes(spec) * num_blocks,
                layers=layers,
                layer_stride=0,
                block_stride=page_bytes(spec),
            )
            for spec, layers in spyre_kv_placement(kv_cache_groups)
        ]
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    def _get_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int | None:
        return sum(page_bytes(spec) for spec, _ in spyre_kv_placement(kv_cache_groups))
