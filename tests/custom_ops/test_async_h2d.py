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

"""Non-blocking host-to-Spyre uploads through `convert`."""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops.utils import _async_source

POISON = -7


@pytest.mark.parametrize(
    "dtype,owned,copied",
    [
        (torch.int32, False, True),
        (torch.float32, False, True),
        (torch.int32, True, False),
        (torch.int64, False, False),
        (torch.float16, False, False),
    ],
)
def test_async_source_copies_only_unstaged_caller_buffers(dtype, owned, copied):
    src = torch.zeros(8, dtype=dtype)
    assert (_async_source(src, owned).data_ptr() != src.data_ptr()) == copied


@pytest.mark.parametrize(
    "source,device,dtype",
    [
        (torch.zeros(4, dtype=torch.int32), "cpu", None),
        (torch.zeros(4, dtype=torch.float32), "spyre", torch.float16),
        (torch.zeros(8, dtype=torch.int32)[::2], "spyre", None),
    ],
    ids=["cpu-target", "dtype-change", "strided"],
)
def test_layout_convert_falls_back_to_tensor_to(source, device, dtype, monkeypatch):
    from spyre_inference.custom_ops import utils

    def no_async(*args, **kwargs):
        raise AssertionError("must not take the async layout upload")

    layouts = []

    def record_to(tensor, *args, **kwargs):
        layouts.append(kwargs.get("device_layout"))
        return tensor

    monkeypatch.setattr(utils, "_to_layout_async", no_async)
    monkeypatch.setattr(torch.Tensor, "to", record_to)
    sentinel = object()
    utils.convert(source, device=device, dtype=dtype, device_layout=sentinel)
    assert layouts == [sentinel]


def test_async_source_keeps_pinned_copies_pinned():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    src = torch.zeros(8, dtype=torch.int32).pin_memory()
    copy = _async_source(src, owned=False)
    assert copy.is_pinned() and copy.data_ptr() != src.data_ptr()


@pytest.mark.parametrize(
    "layout,dtype,pinned",
    [
        pytest.param(False, torch.int32, False, id="convert-int32"),
        pytest.param(False, torch.float32, False, id="convert-fp32"),
        # Passed through uncloned: these fail if torch-spyre stops staging them.
        pytest.param(False, torch.int64, False, id="convert-int64"),
        pytest.param(False, torch.float16, False, id="convert-fp16"),
        pytest.param(True, torch.int32, False, id="layout-int32"),
        pytest.param(True, torch.float16, False, id="layout-fp16"),
        pytest.param(False, torch.int32, True, id="convert-int32-pinned"),
        pytest.param(True, torch.int32, True, id="layout-int32-pinned"),
    ],
)
def test_upload_ignores_source_rewrite_after_enqueue(layout, dtype, pinned):
    """Neither the caller rewriting its buffer nor a reuse of a freed copy may reach the card."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops import utils

    utils.register()
    dev = torch.device("spyre")
    a = torch.randn(2048, 2048, dtype=torch.float16).to(dev)
    mm = torch.compile(lambda x: x @ x, dynamic=False)
    mm(a)
    torch.spyre.synchronize()

    for _ in range(20):
        # Two whole sticks: a same-format upload of this size skips torch-spyre's staging.
        n = 128 // dtype.itemsize * 2
        src = torch.arange(n, dtype=dtype) + 40
        if pinned:
            src = src.pin_memory()
        want = src.clone()
        for _ in range(40):
            mm(a)
        if layout:
            dst = utils.convert(
                src.reshape(2, n // 2),
                device=dev,
                device_layout=utils.row_outermost_layout((2, n // 2), dtype),
            )
        else:
            dst = utils.convert(src, device=dev)
        src.fill_(POISON)
        # Same-size allocations can land on the block of a copy freed after enqueue.
        reused = [torch.full((n,), POISON, dtype=dtype, pin_memory=pinned) for _ in range(4)]
        torch.spyre.synchronize()
        assert torch.equal(dst.cpu().reshape(-1), want)
        del reused
