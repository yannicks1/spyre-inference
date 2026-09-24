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

"""Tests for `spyre_inference/multimodal/granite4_vision.py`.

Each patch replaces an upstream tensor expression that cannot be laid out on device with
one that can, so the load-bearing tests are equivalence checks against the upstream
expression. Two of them earned their place the hard way: a strided-read rewrite of the
2x2 pooling and a `cat`-based newline append both lowered *silently wrong* on card and
were only caught by comparing against these references.

The patches are `getattr`-guarded, so an upstream rename turns one into a silent no-op --
hence the staleness tripwires. All host-side math; no card needed.
"""

from fractions import Fraction

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

granite4_vision = pytest.importorskip("vllm.model_executor.models.granite4_vision")
blip2 = pytest.importorskip("vllm.model_executor.models.blip2")

from spyre_inference.multimodal.granite4_vision import (  # noqa: E402
    _packed_row_index,
    _pool_matrix,
    _window_attention,
    patch_downsamplers,
    patch_feature_packing,
)

pytestmark = [pytest.mark.granite4_vision]

# granite-vision's real geometry: a 384/16 SigLIP grid downsampled by 4/8, anyres-tiled
# 2x3 plus the base tile, for a 728x840 image.
SIDE = 24
NEW_SIDE = 12
PATCHES = 7
GRID = (2, 3)
IMAGE_SIZE = (728, 840)
HIDDEN = 8


def test_pool_matrix_matches_area_interpolation():
    """`InterpolateDownsampler` is `F.interpolate(mode="area")`, i.e. an unimplemented
    `aten::_adaptive_avg_pool2d`; at an integer ratio it is the mean of each block."""
    features = torch.randn(3, SIDE * SIDE, HIDDEN, dtype=torch.float64)

    large = features.view(3, SIDE, SIDE, HIDDEN).permute(0, 3, 1, 2)
    small = torch.nn.functional.interpolate(large, size=(NEW_SIDE, NEW_SIDE), mode="area")
    expected = small.permute(0, 2, 3, 1).flatten(1, 2)

    matrix = _pool_matrix(SIDE, NEW_SIDE, None, torch.float64, torch.device("cpu"))
    assert torch.allclose(torch.matmul(matrix, features), expected)


@pytest.mark.parametrize("offset", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_pool_matrix_matches_spatial_offset_sampling(offset):
    """`SpatialOffsetDownsampler` picks one position per 2x2 block with a strided
    multi-axis index; the matrix selects the same one."""
    features = torch.randn(3, SIDE * SIDE, HIDDEN, dtype=torch.float64)

    sampler = granite4_vision.SpatialOffsetDownsampler.__new__(
        granite4_vision.SpatialOffsetDownsampler
    )
    sampler.orig_image_side = SIDE
    sampler.new_image_side = NEW_SIDE
    sampler.offset_h, sampler.offset_w = offset
    expected = sampler(features)

    matrix = _pool_matrix(SIDE, NEW_SIDE, offset, torch.float64, torch.device("cpu"))
    assert torch.equal(torch.matmul(matrix, features), expected)


def _upstream_packed(features, newline):
    """Upstream `_pack_and_unpad_image_features`, multi-tile branch, verbatim."""
    from transformers.models.llava_next.modeling_llava_next import unpad_image

    base, tiles = features[0], features[1:]
    tiles = tiles.view(*GRID, NEW_SIDE, NEW_SIDE, -1)
    tiles = tiles.permute(4, 0, 2, 1, 3).contiguous().flatten(1, 2).flatten(2, 3)
    tiles = unpad_image(tiles, torch.tensor(IMAGE_SIZE))
    tiles = torch.cat((tiles, newline[:, None, None].expand(*tiles.shape[:-1], 1)), dim=-1)
    return torch.cat((base, tiles.flatten(1, 2).transpose(0, 1)), dim=0)


def test_packed_row_index_matches_upstream_packing():
    """The gather reproduces upstream's view/permute/unpad/newline/flatten chain, whose
    `permute(...).contiguous()` is the expression with no device layout."""
    features = torch.randn(PATCHES, NEW_SIDE * NEW_SIDE, HIDDEN, dtype=torch.float64)
    newline = torch.randn(HIDDEN, dtype=torch.float64)

    rows = _packed_row_index(PATCHES, NEW_SIDE, GRID, IMAGE_SIZE)
    newline_row = PATCHES * NEW_SIDE * NEW_SIDE
    source = features.reshape(newline_row, HIDDEN)
    gathered = source.index_select(0, rows.clamp(max=newline_row - 1))
    gathered = torch.where((rows == newline_row).unsqueeze(-1), newline, gathered)

    assert torch.equal(gathered, _upstream_packed(features, newline))


def test_packed_row_index_single_tile():
    """One tile takes upstream's other branch: the features, then one newline row."""
    features = torch.randn(1, NEW_SIDE * NEW_SIDE, HIDDEN, dtype=torch.float64)
    newline = torch.randn(HIDDEN, dtype=torch.float64)

    rows = _packed_row_index(1, NEW_SIDE, GRID, IMAGE_SIZE)
    newline_row = NEW_SIDE * NEW_SIDE
    gathered = features.reshape(newline_row, HIDDEN).index_select(
        0, rows.clamp(max=newline_row - 1)
    )
    gathered = torch.where((rows == newline_row).unsqueeze(-1), newline, gathered)

    expected = torch.cat((features[0], newline[None]), dim=0)
    assert torch.equal(gathered, expected)


@pytest.mark.parametrize(("queries", "keys"), [(16, 64), (16, 16)])
def test_window_attention_matches_upstream_scores(queries, keys):
    """Both QFormer call sites: cross-attention over the 64-key window and the
    self-attention whose 16 keys are themselves padded (and so masked)."""
    heads, head_dim = 18, 64
    scale = head_dim**-0.5
    q = torch.randn(9, heads, queries, head_dim, dtype=torch.float64)
    k = torch.randn(9, heads, keys, head_dim, dtype=torch.float64)
    v = torch.randn(9, heads, keys, head_dim, dtype=torch.float64)

    expected = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1), v)
    assert torch.allclose(_window_attention(q, k, v, scale), expected)


def test_patched_upstream_attributes_still_exist():
    """Staleness tripwire: every patch resolves its target with `getattr`, so an upstream
    rename would silently skip it."""
    assert hasattr(granite4_vision, "InterpolateDownsampler")
    assert hasattr(granite4_vision, "SpatialOffsetDownsampler")
    assert hasattr(granite4_vision, "get_anyres_image_grid_shape")
    assert hasattr(blip2, "Blip2QFormerMultiHeadAttention")
    model_cls = granite4_vision.Granite4VisionForConditionalGeneration
    for name in ("_pack_and_unpad_image_features", "embed_input_ids", "forward"):
        assert callable(getattr(model_cls, name))


def test_downsamplers_expose_the_attributes_the_patch_reads():
    """`patch_downsamplers` keys off `orig_image_side`/`new_image_side` on both classes
    and `offset_h`/`offset_w` on the spatial one."""
    config = _downsample_config()
    interpolate = granite4_vision.InterpolateDownsampler(config)
    assert (interpolate.orig_image_side, interpolate.new_image_side) == (SIDE, NEW_SIDE)
    spatial = granite4_vision.SpatialOffsetDownsampler(config, offset=3)
    assert (spatial.orig_image_side, spatial.new_image_side) == (SIDE, NEW_SIDE)
    assert (spatial.offset_h, spatial.offset_w) == (1, 1)


def _downsample_config():
    from types import SimpleNamespace

    return SimpleNamespace(
        vision_config=SimpleNamespace(image_size=384, patch_size=16),
        downsample_rate=str(Fraction(NEW_SIDE, SIDE)),
    )


# ---------------------------------------------------------------------------
# On card. The host checks above prove the formulations, which were never the
# problem: both the 2x2-mean rewrite these replaced and a `cat`-appended newline row
# were *correct expressions that the device lowered wrong*. Only a device-vs-host
# comparison catches that class, so these run the patched code on the card.
# ---------------------------------------------------------------------------

# Loose enough for fp16 vs fp32 on these shapes, tight enough that a scrambled layout
# (which lands at O(1) relative error) fails.
DEVICE_TOLERANCE = 0.02


def _stub_model(newline: torch.Tensor | None):
    """The attributes the patched `_pack_and_unpad_image_features` reads off `self`."""
    from types import SimpleNamespace

    return SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=384, patch_size=16),
            # Enough of the real list that a 728x840 image selects 768x1152, i.e. the
            # 2x3 tile grid (+ base tile) the module was brought up against.
            image_grid_pinpoints=[
                [384, 384],
                [384, 768],
                [768, 384],
                [768, 768],
                [768, 1152],
            ],
        ),
        _downsample_rate=Fraction(NEW_SIDE, SIDE),
        image_newline=newline,
    )


def _relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    from spyre_inference.custom_ops.utils import convert

    got = convert(got, device="cpu").to(torch.float32)
    return float((got - want.to(torch.float32)).abs().max() / want.abs().max().clamp(min=1e-6))


@pytest.mark.parametrize("patches", [PATCHES, 1])
def test_pack_and_unpad_on_card_matches_upstream_on_host(patches):
    """Regression for the `cat` defect (see ISSUE_cat_reshaped_view.md): appending the
    newline row to a reshaped view of the `[tiles, tokens, hidden]` features returned
    silently wrong values on device while every host check passed."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from vllm.model_executor.models.granite4_vision import (
        Granite4VisionForConditionalGeneration as model_cls,
    )

    from spyre_inference.custom_ops.utils import convert

    patch_feature_packing()
    hidden = 128
    features = torch.randn(patches, NEW_SIDE * NEW_SIDE, hidden, dtype=torch.float16)
    newline = torch.randn(hidden, dtype=torch.float16)
    # 728x840 tiles 2x3 (+ base) = 7; a 384x384 image is the single-tile branch.
    image_sizes = torch.tensor([[728, 840]] if patches > 1 else [[384, 384]])

    (expected,) = model_cls._pack_and_unpad_image_features(
        _stub_model(newline.to(torch.float32)), (features.to(torch.float32),), image_sizes
    )
    (got,) = model_cls._pack_and_unpad_image_features(
        _stub_model(convert(newline, device="spyre")),
        (convert(features, device="spyre"),),
        image_sizes,
    )

    assert got.shape == expected.shape
    assert _relative_error(got, expected) < DEVICE_TOLERANCE


@pytest.mark.parametrize("offset", [None, (1, 0)])
def test_downsampler_on_card_matches_upstream_on_host(offset):
    """The patched pooling on card. A strided-read rewrite of the same math returned
    ~100% error here while being bit-exact on the host."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.utils import convert

    patch_downsamplers()
    cls = (
        granite4_vision.SpatialOffsetDownsampler
        if offset
        else granite4_vision.InterpolateDownsampler
    )
    sampler = cls(_downsample_config(), **({"offset": 2} if offset else {}))
    features = torch.randn(3, SIDE * SIDE, 128, dtype=torch.float16)

    expected = sampler(features.to(torch.float32))
    got = sampler(convert(features, device="spyre"))

    assert got.shape == expected.shape
    assert _relative_error(got, expected) < DEVICE_TOLERANCE


def test_window_attention_on_card_matches_upstream_on_host():
    """The QFormer windows on card: 16 queries over 64 keys, the shape whose transposed
    key operand has no device layout at all."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.utils import convert

    heads, head_dim, scale = 18, 64, 64**-0.5
    q = torch.randn(9, heads, 16, head_dim, dtype=torch.float16)
    k = torch.randn(9, heads, 64, head_dim, dtype=torch.float16)
    v = torch.randn(9, heads, 64, head_dim, dtype=torch.float16)

    expected = _window_attention(*(x.to(torch.float32) for x in (q, k, v)), scale)
    got = _window_attention(*(convert(x, device="spyre") for x in (q, k, v)), scale)

    assert got.shape == expected.shape
    assert _relative_error(got, expected) < DEVICE_TOLERANCE
