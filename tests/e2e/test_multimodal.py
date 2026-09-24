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

"""End-to-end multimodal tests: Pixtral vision + Ministral-3, and the Gemma 4 tower.

No upstream VLM generation test runs on Spyre (see the test_pixtral.py entry in
`upstream_tests.yaml`), so the end-to-end guarantee lives here. A synthetic image has
no meaningful reference string, so this only asserts the path runs and decodes text.
"""

import io
import sys
from pathlib import Path

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count

# Pixtral vision encoder + multimodal projector + Ministral-3 text decoder.
# The 14B is what this branch was brought up against — no smaller stand-in.
MODEL = "mistralai/Ministral-3-14B-Instruct-2512-BF16"

# Stock transformers vision tower + Gemma-4 MoE decoder, the second vision path here.
# `-it`: the base checkpoint ships no chat template.
GEMMA4_MODEL = "google/gemma-4-26B-A4B-it"
# The CI cache config pins this repo by commit sha, and a sha-pinned `snapshot_download`
# writes no `refs/main` — so the offline runner can only resolve it by that same sha.
# Keep in sync with .github/cache_config/hf_models_and_datasets.yaml.
GEMMA4_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"

# vLLM's own SigLIP tower + BLIP-2 QFormer projectors on a Granite decoder, the third
# vision path here -- and the only one whose decoder head_dim (64) is padded for stick
# alignment.
GRANITE4_MODEL = "ibm-granite/granite-vision-4.1-4b"
GRANITE4_REVISION = "37d591f06319e8f1638b5adcf58bdf50e0f84f7a"

MAX_MODEL_LEN = 4096
MAX_TOKENS = 16


def _synthetic_image_data_uri(size: int = 176, seed: int = 0) -> str:
    """A deterministic image built in-process — no network, no binary asset.

    Content does not matter; size does. 176x176 gives an 11x11 = 121 patch grid,
    the coprime-with-64 case the vision SDPA and conv patches exist for.
    """
    import base64

    from PIL import Image

    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                (x * 7 + seed) % 256,
                (y * 5 + seed) % 256,
                ((x + y) * 3 + seed) % 256,
            )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def _conversation(*uris: str):
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image_url", "image_url": {"url": u}} for u in uris),
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }
    ]


def _generate(
    conversations,
    enforce_eager: bool,
    images_per_prompt: int = 1,
    model: str = MODEL,
    config_format: str = "mistral",
    dtype: str = "float16",
    revision: str | None = None,
):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        # Explicit, not `auto`: auto's mistral probe is a live Hub call, so the
        # offline CI runner silently loads the unpatched HF tower instead.
        config_format=config_format,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=len(conversations),
        dtype=dtype,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": images_per_prompt},
    )
    outputs = llm.chat(
        conversations,
        SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    return [o.outputs[0].text for o in outputs]


@pytest.mark.multimodal
@pytest.mark.parametrize(
    "enforce_eager",
    [
        pytest.param(True, id="eager"),
        pytest.param(
            False,
            id="compiled",
            marks=pytest.mark.disable_co_optimizing_lx_planning,
        ),
    ],
)
@pytest.mark.uses_subprocess
def test_single_image_prompt_produces_output(enforce_eager, monkeypatch):
    """Smoke: the whole vision path (conv patch embed -> vision rope -> padded
    SDPA -> patch merger -> projector norm -> decoder) runs and decodes text.

    Both modes: `CompileOutermost` samples the mode at construction, so under
    `enforce_eager` every `compile_when_outermost` kernel falls through to eager and the
    compiled path — the repo default — goes uncovered.
    """
    # Not `spyre_available()`: it allocates on the card, opening /dev/vfio here, and
    # the `LLM` worker subprocess then cannot ("Device or resource busy").
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    # Building the graphs for a 14B two-tower model outruns the default timeout.
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    uri = _synthetic_image_data_uri()
    (text,) = _generate([_conversation(uri)], enforce_eager=enforce_eager)

    assert text.strip(), "empty generation from the multimodal path"


@pytest.mark.multimodal
@pytest.mark.uses_subprocess
def test_two_image_prompt_produces_output():
    """Two images make the vision mask non-trivial: a pair of strided sub-block writes
    rather than one full-range write, which is what `patch_block_attention_mask` is for.

    Eager only: the mask is built on CPU either way, so a compiled twin would cost a
    graph build for no new coverage.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    uris = (_synthetic_image_data_uri(seed=0), _synthetic_image_data_uri(seed=97))
    (text,) = _generate([_conversation(*uris)], enforce_eager=True, images_per_prompt=2)

    assert text.strip(), "empty generation from the two-image multimodal path"


@pytest.mark.multimodal
@pytest.mark.gemma4_vision
@pytest.mark.parametrize(
    "enforce_eager",
    [
        pytest.param(True, id="eager"),
        pytest.param(
            False,
            id="compiled",
            marks=pytest.mark.disable_co_optimizing_lx_planning,
        ),
    ],
)
@pytest.mark.uses_subprocess
def test_gemma4_single_image_prompt_produces_output(enforce_eager, monkeypatch):
    """The Gemma 4 tower on card, which the unit tests cannot reach: they check the
    rewrite on CPU, this checks that what it rewrites to actually lowers.

    Both modes, as for Pixtral above: compiled is the repo default, and the tower's
    rewrites (permutation-matmul rope, padded SDPA, the pooling matmul) have to lower
    inside a graph, not just run eagerly.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    # The 26B weights are prefetched only by the perf job, which never runs on a PR, so
    # the shared cache can legitimately not have them yet.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(GEMMA4_MODEL, revision=GEMMA4_REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"{GEMMA4_MODEL}@{GEMMA4_REVISION[:7]} not in the local HF cache")

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    uri = _synthetic_image_data_uri()
    (text,) = _generate(
        [_conversation(uri)],
        enforce_eager=enforce_eager,
        model=GEMMA4_MODEL,
        config_format="hf",
        revision=GEMMA4_REVISION,
    )

    assert text.strip(), "empty generation from the Gemma 4 vision path"


def _skip_without_cached_model(model: str, revision: str) -> None:
    """Skip when the shared HF cache has not prefetched all of `model`.

    Checked file by file, not just by `snapshot_download` succeeding: a cache populated
    with `allow_patterns` holds the config and the weights while the tokenizer and the
    processor are absent, and that combination raises mid-test instead of skipping.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(model, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError:
        pytest.skip(f"{model}@{revision[:7]} not in the local HF cache")
    missing = [
        name
        for name in ("tokenizer_config.json", "preprocessor_config.json")
        if not (path / name).exists()
    ]
    if missing:
        pytest.skip(f"{model}@{revision[:7]} is cached without {', '.join(missing)}")


@pytest.mark.multimodal
@pytest.mark.granite4_vision
@pytest.mark.parametrize(
    "enforce_eager",
    [
        pytest.param(True, id="eager"),
        pytest.param(
            False,
            id="compiled",
            marks=pytest.mark.disable_co_optimizing_lx_planning,
        ),
    ],
)
@pytest.mark.uses_subprocess
def test_granite4_single_image_prompt_produces_output(enforce_eager, monkeypatch):
    """The Granite 4 Vision encode path on card: SigLIP tower at a padded head_dim,
    the constant-matmul patch-grid pooling, the QFormer windows on padded shapes, the
    anyres row gather, and the deepstack features reaching the decoder.

    Both modes, as for the towers above: the patches have to lower inside a graph as
    well as eagerly, and compiled is the repo default.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")
    _skip_without_cached_model(GRANITE4_MODEL, GRANITE4_REVISION)

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    uri = _synthetic_image_data_uri()
    (text,) = _generate(
        [_conversation(uri)],
        enforce_eager=enforce_eager,
        model=GRANITE4_MODEL,
        config_format="hf",
        revision=GRANITE4_REVISION,
    )

    assert text.strip(), "empty generation from the Granite 4 vision path"


@pytest.mark.multimodal
@pytest.mark.granite4_vision
@pytest.mark.uses_subprocess
def test_granite4_two_image_prompt_produces_output(monkeypatch):
    """Two images make the deepstack merge concatenate per-image feature tensors, which
    one image leaves as a no-op, and give the anyres row gather a second geometry.

    Eager only: the gather index is built on the host either way, so a compiled twin
    would cost a graph build for no new coverage.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")
    _skip_without_cached_model(GRANITE4_MODEL, GRANITE4_REVISION)

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    uris = (_synthetic_image_data_uri(seed=0), _synthetic_image_data_uri(seed=97))
    (text,) = _generate(
        [_conversation(*uris)],
        enforce_eager=True,
        images_per_prompt=2,
        model=GRANITE4_MODEL,
        config_format="hf",
        revision=GRANITE4_REVISION,
    )

    assert text.strip(), "empty generation from the two-image Granite 4 vision path"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
