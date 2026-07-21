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

"""Tests for the alternative matmul (contract) Spyre RoPE kernel and its SPYRE_ROPE_MATMUL
switch. The canonical slice kernel is covered by test_rotary_embedding.py."""

import pytest
import torch

# Same regimes as test_rotary_embedding.py: inner dim head_size//2 stick-aligned
# (128->64, 256->128) or padded up to a stick (64->32, the case the two kernels diverge).
HEAD_SIZES = [64, 128, 256]


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_matmul_rotation_matches_reference_cpu(default_vllm_config, head_size):
    """CPU-only: the contract kernel _apply_rope_matmul matches forward_native. Reuses the
    canonical (zero-padded) rotation cache, exercising that the expand/contract path is
    correct with it. head_size=64 hits the pad-to-stick expand/contract path."""
    import spyre_inference.custom_ops.rotary_embedding_matmul as matmul
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(11)
    max_position, num_tokens, num_heads = 2048, 32, 4
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    rot = rope.gather_rotation(positions, torch.device("cpu"))
    assert rot is not None and rot.device.type == "cpu"
    actual_query = matmul._apply_rope_matmul(query, rot, head_size)
    actual_key = matmul._apply_rope_matmul(key, rot, head_size)

    expected_query, expected_key = RotaryEmbedding.forward_native(rope, positions, query, key)
    torch.testing.assert_close(actual_query.float(), expected_query.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_key.float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
def test_use_matmul_rope_swaps_oot_registry():
    """use_matmul_rope() replaces the OOT RotaryEmbedding entries with the contract
    variants so get_rope resolves to them. Restores the registry afterward for isolation."""
    import spyre_inference.custom_ops.rotary_embedding_matmul as matmul
    from vllm.model_executor.custom_op import op_registry_oot

    names = ("RotaryEmbedding", "Llama3RotaryEmbedding")
    saved = {n: op_registry_oot.get(n) for n in names}
    try:
        matmul.use_matmul_rope()
        assert op_registry_oot["RotaryEmbedding"] is matmul.SpyreRotaryEmbeddingMatmul
        assert op_registry_oot["Llama3RotaryEmbedding"] is matmul.SpyreLlama3RotaryEmbeddingMatmul
    finally:
        for n, cls in saved.items():
            if cls is not None:
                op_registry_oot[n] = cls
