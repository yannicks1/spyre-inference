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

"""TorchSpyreWorker sizes the KV cache through upstream's Worker.determine_available_memory
(spyre-inference#901), with every block the measured memory pays for. Only the card is
faked: torch.accelerator's memory view, with free bytes the test sets, and the live
allocations in the allocator's regions.

The device cost per block comes from upstream's own KV cache grouping, and every layer's
own page counts towards it, as allocate_spyre_kv_caches allocates them."""

import itertools
import logging
import random
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import request_memory

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl
from spyre_inference.v1.worker import spyre_model_runner, spyre_worker
from spyre_inference.v1.worker.kv_cache_setup import allocate_spyre_kv_caches
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner
from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker, _tensors_fit

GiB = 1 << 30
MiB = 1 << 20
TOTAL = 96 * GiB
REGION = 16 * GiB
RESERVE = spyre_worker._SPYRE_EXTENT_RESERVE_BYTES
BLOCK_BYTES = 2 * 2 * 4 * 128 * 128  # K+V, fp16, 4 kv heads, head 128, block 128
BATCH_BLOCKS = 4 * (1024 // 128) + 1  # _worker's default max_num_seqs at max_model_len
NUM_LAYERS = 32


def _full(dtype: torch.dtype = torch.float16) -> FullAttentionSpec:
    return FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=dtype)


def _full_layers(dtype: torch.dtype = torch.float16, num_layers: int = NUM_LAYERS) -> dict:
    return {f"layers.{i}.attn": _full(dtype) for i in range(num_layers)}


def _worker(
    monkeypatch,
    *,
    free_after_load: int,
    max_num_seqs: int = 4,
    max_model_len: int = 1024,
    kv_cache_memory_bytes: int | None = None,
    dtype: torch.dtype = torch.float16,
    kv_cache_spec: dict | None = None,
    hybrid_kv_cache_manager: bool = False,
    allocations: dict | None = None,
) -> TorchSpyreWorker:
    """A worker that started on an empty card and whose model now leaves free_after_load.

    The model's bytes sit at the start of each of the six regions, spread evenly as the
    allocator spreads them, unless ``allocations`` (``{region: {offset: bytes}}``) says
    where they are.
    """
    card = SimpleNamespace(free=TOTAL, allocations={})
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (card.free, TOTAL))
    monkeypatch.setattr(spyre_worker, "_spyre_allocations", lambda: card.allocations)
    monkeypatch.setattr(torch.accelerator, "memory_stats", lambda *_: {})
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda *_: 0)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda *_: None)

    specs = kv_cache_spec if kv_cache_spec is not None else _full_layers(dtype)
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        max_model_len=max_model_len, multimodal_config=None, supports_kv_cache_grouping=False
    )
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.block_size = 128
    vllm_config.cache_config.kv_cache_memory_bytes = kv_cache_memory_bytes
    vllm_config.cache_config.gpu_memory_utilization = 0.9
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = not hybrid_kv_cache_manager
    worker = object.__new__(TorchSpyreWorker)
    worker.vllm_config = vllm_config
    worker.parallel_config = SimpleNamespace(data_parallel_size=1)
    worker.cache_config = vllm_config.cache_config
    worker.model_config = vllm_config.model_config
    worker.scheduler_config = vllm_config.scheduler_config
    worker.model_runner = SimpleNamespace(
        get_kv_cache_spec=lambda: dict(specs),
        profile_run=lambda **_: None,
        model_memory_usage=0,
    )
    worker.init_snapshot = MemorySnapshot(device=torch.device("cpu"))
    worker.requested_memory = request_memory(worker.init_snapshot, worker.cache_config)
    card.free = free_after_load
    used = TOTAL - free_after_load
    card.allocations = (
        allocations
        if allocations is not None
        else {r: {0: used // 6} for r in range(6) if used > 0}
    )
    return worker


def test_the_budget_is_the_requested_share_minus_what_loading_consumed(monkeypatch):
    """In whole blocks, as vLLM divides the budget by the bytes per block anyway, and
    past what a full batch needs: the prefix cache keeps the rest for reuse."""
    worker = _worker(monkeypatch, free_after_load=60 * GiB)

    blocks = worker.determine_available_memory() // (NUM_LAYERS * BLOCK_BYTES)

    assert blocks == (worker.requested_memory - 36 * GiB) // (NUM_LAYERS * BLOCK_BYTES)
    assert blocks > BATCH_BLOCKS


def test_an_explicit_kv_cache_memory_bytes_skips_the_measurement(monkeypatch):
    worker = _worker(monkeypatch, free_after_load=0, kv_cache_memory_bytes=3 * GiB)

    assert worker.determine_available_memory() == 3 * GiB


def test_loading_past_the_requested_share_leaves_nothing(monkeypatch):
    """vLLM then fails at init, before any allocation, naming gpu_memory_utilization."""
    worker = _worker(monkeypatch, free_after_load=5 * GiB)

    assert worker.determine_available_memory() <= 0


def test_an_fp8_cache_fits_twice_the_blocks(monkeypatch):
    """The budget is bytes and vLLM divides it by the spec's page size, so a 1-byte
    cache holds twice the tokens in the same memory."""
    blocks = {}
    for dtype in (torch.float16, torch.uint8):
        worker = _worker(monkeypatch, free_after_load=60 * GiB, dtype=dtype)
        page = next(iter(worker.model_runner.get_kv_cache_spec().values())).page_size_bytes
        blocks[dtype] = worker.determine_available_memory() // (NUM_LAYERS * page)

    assert blocks[torch.uint8] == 2 * blocks[torch.float16]


def _pack(
    tensor_bytes: list[int],
    allocations: dict,
    choose: Callable[[list[list[int]]], list[int]] | None = None,
) -> bool:
    """Place each tensor, largest first, as the allocator does: among the regions whose
    free bytes cover it (the most free first, or ``choose``'s pick), the first whose
    smallest fitting free range holds it."""
    regions: list[list[int]] = []
    for r in range(6):
        blocks = sorted(allocations.get(r, {}).items())
        gaps: list[int] = []
        end = 0
        for offset, size in blocks:
            gaps.append(offset - end)
            end = offset + size
        regions.append([*gaps, REGION - end])
    for nbytes in sorted(tensor_bytes, reverse=True):
        candidates = [g for g in regions if sum(g) >= nbytes and max(g) >= nbytes]
        if not candidates:
            return False
        region = choose(candidates) if choose else max(candidates, key=lambda g: sum(g))
        i = min((i for i, g in enumerate(region) if g >= nbytes), key=region.__getitem__)
        region[i] -= nbytes
    return True


def _shallow_worker(monkeypatch, allocations: dict) -> TorchSpyreWorker:
    """Four layers, so the default budget would make eight K/V tensors over half a region."""
    used = sum(sum(blocks.values()) for blocks in allocations.values())
    worker = _worker(
        monkeypatch,
        free_after_load=TOTAL - used,
        kv_cache_spec=_full_layers(num_layers=4),
        allocations=allocations,
    )
    worker.cache_config.gpu_memory_utilization = 0.92
    worker.requested_memory = request_memory(worker.init_snapshot, worker.cache_config)
    return worker


@pytest.mark.parametrize(
    "allocations",
    [
        pytest.param({0: {0: 1 * GiB}}, id="weights-in-one-region"),
        pytest.param({5: {0: 1 * GiB}}, id="weights-in-the-last-region"),
        pytest.param({r: {0: 400 * MiB} for r in range(6)}, id="weights-spread"),
    ],
)
def test_a_shallow_models_kv_tensors_fill_the_16_gib_regions(monkeypatch, allocations):
    """A tensor never spans two of the card's regions, so at the full budget one tensor
    fits each. Two per region fit at just under half a region each."""
    worker = _shallow_worker(monkeypatch, allocations)
    weights = sum(sum(blocks.values()) for blocks in allocations.values())
    extents = sorted(
        (REGION - sum(allocations.get(r, {}).values()) - RESERVE for r in range(6)), reverse=True
    )

    blocks = worker.determine_available_memory() // (4 * BLOCK_BYTES)

    assert blocks * 4 * BLOCK_BYTES < worker.requested_memory - weights
    # Two to a region in the four roomiest, as large as the extent reserve allows.
    assert blocks == extents[3] // 2 // (BLOCK_BYTES // 2)
    assert _pack([blocks * BLOCK_BYTES // 2] * 8, allocations)


def test_a_live_allocation_inside_a_region_splits_its_free_range(monkeypatch):
    """A tensor needs one contiguous range: an allocation in the middle of every region
    leaves two halves, each holding one tensor of just under half a region, not two."""
    allocations = {r: {8 * GiB: 1 * MiB} for r in range(6)}
    worker = _shallow_worker(monkeypatch, allocations)

    blocks = worker.determine_available_memory() // (4 * BLOCK_BYTES)

    assert blocks == (8 * GiB - 1 * MiB - RESERVE) // (BLOCK_BYTES // 2)
    assert _pack([blocks * BLOCK_BYTES // 2] * 8, allocations)


def test_bytes_no_live_tensor_holds_size_for_any_placement(monkeypatch):
    """Bytes the walk cannot locate could split any free range, so the tensors are sized
    as if every region held some: one largest tensor spared per other region."""
    seen = {r: {0: 400 * MiB} for r in range(6)}
    worker = _shallow_worker(monkeypatch, seen)
    free = TOTAL - 2400 * MiB - 2 * GiB
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (free, TOTAL))

    blocks = worker.determine_available_memory() // (4 * BLOCK_BYTES)

    assert blocks == free // ((8 + 5) * (BLOCK_BYTES // 2))


def test_an_alias_inside_an_allocation_is_not_counted_twice(monkeypatch):
    """A view's storage may start inside another allocation (a per-expert alias of an
    expert stack); the region's free range starts where the outer allocation ends."""
    allocations = {0: {0: 4 * GiB, 1 * GiB: 512 * MiB}}
    monkeypatch.setattr(spyre_worker, "_spyre_allocations", lambda: allocations)

    extents = spyre_worker._free_extents(TOTAL - 4 * GiB, TOTAL)

    assert extents is not None
    assert sorted(extents) == [12 * GiB - RESERVE] + [REGION - RESERVE] * 5


def _choices(tensor_bytes: list[int], allocations: dict):
    """Every sequence of region choices the allocator could make."""
    for picks in itertools.product(range(6), repeat=len(tensor_bytes)):
        it = iter(picks)

        def choose(candidates, it=it):
            return candidates[next(it) % len(candidates)]

        yield choose


def test_mixed_sizes_fit_whichever_region_the_allocator_picks():
    """_tensors_fit's bound is sound for every placement order among fitting regions."""
    rng = random.Random(0)
    checked = 0
    for _ in range(120):
        allocations = {r: {0: rng.randrange(0, 12) * GiB} for r in range(6) if rng.random() < 0.7}
        allocations = {r: b for r, b in allocations.items() if b[0]}
        sizes = rng.choice([[3, 4, 6, 8], [4, 4.25], [5, 5.5, 8]])
        tensors = [int(rng.choice(sizes) * GiB) for _ in range(rng.randrange(2, 5))]
        extents = [REGION - b[0] for b in allocations.values()]
        extents += [REGION] * (6 - len(allocations))
        if not _tensors_fit(tensors, extents):
            continue
        checked += 1
        for choose in _choices(tensors, allocations):
            assert _pack(tensors, allocations, choose), (tensors, allocations)
    assert checked > 20


def test_close_sizes_fit_rounded_up_to_the_larger():
    """Placing a slightly larger tensor may take two slots of the smaller size, but
    rounded up to the larger size every tensor takes one."""
    extents = [3270 * MiB] * 6

    assert _tensors_fit([1310 * MiB] * 6 + [1256 * MiB] * 2, extents)


def test_equal_sizes_fit_exactly_as_the_bound_says():
    """For one tensor size the bound is exact: one byte more and the allocator fails."""
    allocations = {r: {0: 400 * MiB} for r in range(6)}
    extents = [REGION - 400 * MiB] * 6
    size = (REGION - 400 * MiB) // 2

    assert _tensors_fit([size] * 12, extents)
    assert _pack([size] * 12, allocations)
    assert not _tensors_fit([size + 1] * 12, extents)
    assert not _pack([size + 1] * 12, allocations)


def _mamba_hybrid(num_mamba: int = 6) -> dict:
    page = _full().page_size_bytes
    mamba = MambaSpec(
        shapes=((3, 1024), (8, 64, 128)),
        dtypes=(torch.float16, torch.float16),
        block_size=8192,
        page_size_padded=page,
        mamba_cache_mode="none",
    )
    return {**{f"layers.{i}.mixer": mamba for i in range(num_mamba)}, "layers.99.attn": _full()}


def test_a_mamba_hybrid_is_sized_by_its_per_layer_device_cost(monkeypatch):
    """vLLM's pool needs one shared page per block for all groups, but Spyre gives every
    layer its own page: the budget converts the measured device bytes into blocks at
    Spyre's cost, so they fit on the card."""
    specs = _mamba_hybrid()
    worker = _worker(
        monkeypatch, free_after_load=20 * GiB, kv_cache_spec=specs, hybrid_kv_cache_manager=True
    )
    mamba_bytes = specs["layers.0.mixer"].real_page_size_bytes
    spyre_bytes_per_block = 6 * mamba_bytes + BLOCK_BYTES

    blocks = worker.determine_available_memory() // BLOCK_BYTES

    measured = worker.requested_memory - (TOTAL - 20 * GiB)
    assert blocks == measured // spyre_bytes_per_block
    assert blocks * spyre_bytes_per_block <= measured


@pytest.mark.parametrize(("spare_blocks", "warns"), [(-1, True), (0, False)])
def test_requests_queue_for_pages_only_below_the_batch_area(
    monkeypatch, caplog, spare_blocks, warns
):
    worker = _worker(monkeypatch, free_after_load=TOTAL)
    worker.requested_memory = (BATCH_BLOCKS + spare_blocks) * NUM_LAYERS * BLOCK_BYTES

    with caplog.at_level(logging.WARNING, logger="spyre_inference.v1.worker.spyre_worker"):
        caplog.clear()
        blocks = worker.determine_available_memory() // (NUM_LAYERS * BLOCK_BYTES)

    assert blocks == BATCH_BLOCKS + spare_blocks
    assert any("queue for pages" in r.getMessage() for r in caplog.records) == warns


@pytest.mark.parametrize(("override", "warns"), [(BATCH_BLOCKS - 1, True), (BATCH_BLOCKS, False)])
def test_the_batch_area_warning_checks_num_gpu_blocks_override(
    monkeypatch, caplog, override, warns
):
    """vLLM replaces the measured block count with the override later, so the override,
    not the card, decides whether a full batch fits."""
    worker = _worker(monkeypatch, free_after_load=TOTAL)
    worker.requested_memory = 2 * BATCH_BLOCKS * NUM_LAYERS * BLOCK_BYTES
    worker.cache_config.num_gpu_blocks_override = override

    with caplog.at_level(logging.WARNING, logger="spyre_inference.v1.worker.spyre_worker"):
        caplog.clear()
        worker.determine_available_memory()

    assert any("queue for pages" in r.getMessage() for r in caplog.records) == warns


def test_sizing_a_sliding_window_model_logs_no_vllm_warning(monkeypatch, caplog):
    """vLLM warns once that the hybrid manager is off for such a model, in
    get_kv_cache_configs; sizing must not repeat it on every worker."""
    specs = {
        "layers.0.attn": _full(),
        "layers.1.attn": SlidingWindowSpec(
            block_size=128, num_kv_heads=4, head_size=128, dtype=torch.float16, sliding_window=512
        ),
    }
    worker = _worker(monkeypatch, free_after_load=60 * GiB, kv_cache_spec=specs)
    kv_logger = logging.getLogger("vllm.v1.core.kv_cache_planning")
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)

    with caplog.at_level(logging.WARNING, logger="vllm.v1.core.kv_cache_planning"):
        caplog.clear()
        assert worker.determine_available_memory() > 0

    assert not [r for r in caplog.records if r.name == kv_logger.name]
    assert not kv_logger.disabled


def test_sizing_allocates_the_staging_buffers_and_runs_no_forward(monkeypatch):
    """On Spyre the first forward is the trace; run before the KV cache exists, it would
    trace no KV write and warmup would retrace every block. The staging buffers are
    allocated inside the measurement instead: they scale with max_num_batched_tokens, and
    allocated after the KV cache took the measured memory they would not fit."""

    def forward(*args, **kwargs):
        raise AssertionError("determine_available_memory ran a forward")

    staged = []
    monkeypatch.setattr(GPUModelRunner, "profile_run", forward)
    monkeypatch.setattr(TorchSpyreModelRunner, "_dummy_run", forward)
    monkeypatch.setattr(
        spyre_model_runner, "allocate_staging_buffers", lambda *args: staged.append(args)
    )
    worker = _worker(monkeypatch, free_after_load=60 * GiB)
    runner = object.__new__(TorchSpyreModelRunner)
    runner.model_memory_usage = 0
    runner.get_kv_cache_spec = worker.model_runner.get_kv_cache_spec
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner._spyre_device = torch.device("cpu")
    worker.model_runner = runner

    assert worker.determine_available_memory() > 0
    assert staged == [({}, torch.device("cpu"))]


def test_larger_pages_are_allocated_first(monkeypatch):
    """_region_safe_blocks sizes the cache for this order."""
    small = SlidingWindowSpec(
        block_size=128, num_kv_heads=2, head_size=128, dtype=torch.float16, sliding_window=512
    )
    specs = {f"layers.{i}.attn": small if i % 3 else _full() for i in range(6)}
    config = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=list(specs), kv_cache_spec=SimpleNamespace(kv_cache_specs=specs)
            )
        ],
        kv_cache_tensors=[SimpleNamespace(layers=list(specs))],
    )
    calls = []

    def allocate_pages(cls, num_blocks, spec, device):
        calls.append(spec.page_size_bytes)
        return object()

    monkeypatch.setattr(SpyreAttentionImpl, "allocate_pages", classmethod(allocate_pages))

    allocate_spyre_kv_caches(config, {}, {}, torch.device("cpu"))

    assert calls == [_full().page_size_bytes] * 2 + [small.page_size_bytes] * 4
