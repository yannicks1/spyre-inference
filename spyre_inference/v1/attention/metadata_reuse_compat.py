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

"""Temporary same-step reuse while MRV2 limits update_block_table to Mamba.

Once vLLM includes the generic caller (d8df1624), remove this module, the build
decorator, and SpyreDefaultModelState.prepare_attn. The backend capability and
rebinding implementation remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from functools import wraps
from typing import TYPE_CHECKING

from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata

if TYPE_CHECKING:
    from spyre_inference.v1.attention.backends.spyre_attn import (
        SpyreAttentionMetadata,
        SpyreAttentionMetadataBuilder,
    )

BuildCache = dict[tuple, tuple[CommonAttentionMetadata, "SpyreAttentionMetadata"]]


class StepMetadata(ModelSpecificAttnMetadata):
    def __init__(self) -> None:
        self.builds: BuildCache = {}

    def get_extra_attn_kwargs(
        self, attn_metadata_builder: AttentionMetadataBuilder, num_reqs: int
    ) -> dict:
        from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder

        if (
            isinstance(attn_metadata_builder, SpyreAttentionMetadataBuilder)
            and attn_metadata_builder.supports_update_block_table
        ):
            return {"_spyre_step_cache": self.builds}
        return {}


def _same_batch(a: CommonAttentionMetadata, b: CommonAttentionMetadata) -> bool:
    # Group tables/slots differ; all other public inputs must match. Tensors are
    # compared by identity, never by a device synchronization or a value hash.
    for f in fields(a):
        if f.name in ("block_table_tensor", "slot_mapping") or f.name.startswith("_"):
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        if x is y:
            continue
        if not isinstance(x, (bool, int, float, str)) or type(x) is not type(y) or x != y:
            return False
    return True


def reuse_attention_metadata(
    build: Callable[..., SpyreAttentionMetadata],
) -> Callable[..., SpyreAttentionMetadata]:
    @wraps(build)
    def wrapped(
        self: SpyreAttentionMetadataBuilder,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        per_seq_only: bool = False,
        *,
        _spyre_step_cache: BuildCache | None = None,
    ) -> SpyreAttentionMetadata:
        cache = _spyre_step_cache
        if cache is None or per_seq_only or not self.supports_update_block_table:
            return build(self, common_prefix_len, common_attn_metadata, fast_build, per_seq_only)
        key = (
            type(self),
            self.kv_cache_spec,
            self.num_heads,
            self.causal_override,
            self.reorder_batch_threshold,
            common_prefix_len,
            fast_build,
        )
        hit = cache.get(key)
        if hit is not None and _same_batch(hit[0], common_attn_metadata):
            return self.update_block_table(
                hit[1], common_attn_metadata.block_table_tensor, common_attn_metadata.slot_mapping
            )
        metadata = build(self, common_prefix_len, common_attn_metadata, fast_build, per_seq_only)
        cache[key] = (common_attn_metadata, metadata)
        return metadata

    return wrapped
