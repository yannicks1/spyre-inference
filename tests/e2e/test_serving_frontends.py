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

"""`vllm serve` through the Python and the Rust (`vllm-rs`) frontend.

Each frontend serves the model once, a fixed request set is sent, and the server is
stopped before any test looks at the responses, so only one engine holds the card at a
time. Python responses are checked against absolute expectations; Rust ones must equal
Python's, except where upstream's Rust frontend deliberately differs.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from spyre_testing_plugin.rust_frontend import require_rust_frontend
from spyre_testing_plugin.tags import record_serving_capability
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 4
STARTUP_TIMEOUT_S = 1500
LOGPROB_TOLERANCE = 1e-3

PROMPTS = ["The capital of France is", "def fibonacci(n):"]
CHAT = [{"role": "user", "content": "Name three primary colors, comma separated."}]
COLOR_SCHEMA = {
    "type": "object",
    "properties": {"color": {"enum": ["red", "green", "blue"]}, "bright": {"type": "boolean"}},
    "required": ["color", "bright"],
    "additionalProperties": False,
}


def _start_server(env: dict[str, str], log_path: Path) -> tuple[subprocess.Popen, str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL]
    cmd += ["--port", str(port), "--max-model-len", str(MAX_MODEL_LEN)]
    cmd += ["--max-num-seqs", str(MAX_NUM_SEQS)]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return proc, url
        except requests.RequestException:
            pass
        time.sleep(5)
    _stop_server(proc)
    tail = log_path.read_text(errors="replace")[-4000:]
    pytest.fail(f"server not ready (exit code {proc.returncode}); log tail:\n{tail}")


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=60)
    # The frontend and EngineCore share the server's session; reap whatever outlived it.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    assert wait_until_card_free(exclude_pids={os.getpid()}, timeout=60), (
        "Spyre devices were not released after the server stopped"
    )


def _post(url: str, path: str, body: dict, stream: bool = False) -> dict:
    r = requests.post(url + path, json={"model": MODEL, **body}, timeout=600)
    if r.status_code != 200:
        return {"status": r.status_code, "error": r.text}
    if not stream:
        return {"status": 200, "body": r.json()}
    chunks = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    return {"status": 200, "chunks": chunks}


def _completion(prompt: str, **kw) -> dict:
    return {"prompt": prompt, "temperature": 0, "max_tokens": 32, **kw}


def _chat(**kw) -> dict:
    return {"messages": CHAT, "temperature": 0, "max_tokens": 32, **kw}


def _overflow_prompt(url: str) -> str:
    """A prompt a few tokens short of max_model_len, so a clamped max_tokens decodes little."""
    unit = "The capital of France is Paris. "
    per_unit = _post(url, "/tokenize", {"prompt": unit * 8})["body"]["count"] / 8
    return unit * int((MAX_MODEL_LEN - 24) / per_unit)


def _collect(url: str) -> dict[str, dict]:
    out = {
        "tokenize_text": _post(url, "/tokenize", {"prompt": PROMPTS[0]}),
        "tokenize_chat": _post(url, "/tokenize", {"messages": CHAT, "add_generation_prompt": True}),
    }
    for i, p in enumerate(PROMPTS):
        out[f"greedy_{i}"] = _post(url, "/v1/completions", _completion(p, logprobs=5))
    out["stream"] = _post(url, "/v1/completions", _completion(PROMPTS[0], stream=True), True)
    out["chat"] = _post(url, "/v1/chat/completions", _chat(logprobs=True, top_logprobs=3))
    out["chat_stream"] = _post(url, "/v1/chat/completions", _chat(stream=True), True)
    out["stop_excl"] = _post(url, "/v1/completions", _completion(PROMPTS[1], stop=["\n\n", ":"]))
    out["stop_incl"] = _post(
        url,
        "/v1/completions",
        _completion(PROMPTS[1], stop=["\n\n", ":"], include_stop_str_in_output=True),
    )
    out["json_schema"] = _post(
        url,
        "/v1/chat/completions",
        _chat(
            messages=[{"role": "user", "content": "Pick a color as JSON."}],
            max_tokens=64,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "color", "schema": COLOR_SCHEMA},
            },
        ),
    )
    out["prompt_logprobs"] = _post(
        url, "/v1/completions", _completion(PROMPTS[0], max_tokens=4, prompt_logprobs=1)
    )
    out["n2"] = _post(url, "/v1/completions", _completion(PROMPTS[0], n=2, seed=1, temperature=0.8))
    out["max_tokens_overflow"] = _post(
        url, "/v1/completions", _completion(_overflow_prompt(url), max_tokens=100_000)
    )
    return out


def _serve_and_collect(frontend: str, tmp_path_factory) -> dict[str, dict]:
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_ENGINE_READY_TIMEOUT_S=str(STARTUP_TIMEOUT_S))
    env.pop("VLLM_USE_RUST_FRONTEND", None)
    if frontend == "rust":
        env.update(VLLM_USE_RUST_FRONTEND="1", VLLM_RUST_FRONTEND_PATH=require_rust_frontend())
    else:
        env.pop("VLLM_RUST_FRONTEND_PATH", None)
    log_path = tmp_path_factory.mktemp(f"serve-{frontend}") / "server.log"
    proc, url = _start_server(env, log_path)
    try:
        return _collect(url)
    finally:
        _stop_server(proc)


@pytest.fixture(scope="module")
def python_responses(tmp_path_factory) -> dict[str, dict]:
    return _serve_and_collect("python", tmp_path_factory)


@pytest.fixture(scope="module")
def rust_responses(tmp_path_factory) -> dict[str, dict]:
    return _serve_and_collect("rust", tmp_path_factory)


def _ok(responses: dict[str, dict], name: str) -> dict:
    r = responses[name]
    assert r["status"] == 200, f"{name}: HTTP {r['status']}: {r.get('error', '')[:500]}"
    return r


def _text(r: dict) -> str:
    if "chunks" in r:
        return "".join(
            ch.get("text") or (ch.get("delta") or {}).get("content") or ""
            for c in r["chunks"]
            for ch in c.get("choices", [])
        )
    choice = r["body"]["choices"][0]
    return choice["text"] if "text" in choice else choice["message"]["content"]


def _finish_reason(r: dict) -> str | None:
    if "chunks" in r:
        reasons = [ch.get("finish_reason") for c in r["chunks"] for ch in c.get("choices", [])]
        return next((f for f in reversed(reasons) if f), None)
    return r["body"]["choices"][0]["finish_reason"]


def _assert_close(where: str, ref: float, value: float) -> None:
    assert abs(ref - value) <= LOGPROB_TOLERANCE, f"{where}: python {ref} vs rust {value}"


def _assert_logprobs_match(name: str, ref: dict, got: dict) -> None:
    ref_choice, choice = ref["body"]["choices"][0], got["body"]["choices"][0]
    ref_lp, lp = ref_choice.get("logprobs"), choice.get("logprobs")
    if ref_lp and "token_logprobs" in ref_lp:
        assert lp["tokens"] == ref_lp["tokens"], name
        pairs = zip(ref_lp["token_logprobs"], lp["token_logprobs"], strict=True)
        for i, (a, b) in enumerate(pairs):
            _assert_close(f"{name} token {i}", a, b)
        # Rust renders a special token in the top-k as "" (e.g. <|end_of_text|>).
        for i, (a, b) in enumerate(zip(ref_lp["top_logprobs"], lp["top_logprobs"], strict=True)):
            where = f"{name} step {i}"
            renamed = [t for t in a if t not in b]
            assert len(renamed) == len(set(b) - set(a)) and set(b) - set(a) <= {""}, (
                f"{where}: top-k {sorted(a)} vs {sorted(b)}"
            )
            assert all(t.startswith("<|") for t in renamed), f"{where}: {renamed} missing"
            for tok in a:
                _assert_close(f"{where} {tok!r}", a[tok], b[tok if tok in b else ""])
    elif ref_lp:
        # Chat top-k is compared by value, for the same special-token rendering.
        for i, (a, b) in enumerate(zip(ref_lp["content"], lp["content"], strict=True)):
            assert a["token"] == b["token"], f"{name} step {i}: {a['token']!r} vs {b['token']!r}"
            _assert_close(f"{name} step {i}", a["logprob"], b["logprob"])
            for j, (x, y) in enumerate(zip(a["top_logprobs"], b["top_logprobs"], strict=True)):
                _assert_close(f"{name} step {i} rank {j}", x["logprob"], y["logprob"])
    ref_plp = ref_choice.get("prompt_logprobs")
    if ref_plp:
        plp = choice["prompt_logprobs"]
        assert len(plp) == len(ref_plp), name
        for i, (a, b) in enumerate(zip(ref_plp[1:], plp[1:])):
            assert a.keys() == b.keys(), f"{name} prompt position {i + 1}"
            for tok in a:
                _assert_close(f"{name} prompt {i + 1}", a[tok]["logprob"], b[tok]["logprob"])


def _assert_json_schema_output(r: dict) -> None:
    value = json.loads(_text(r))
    assert set(value) == {"color", "bright"}, value
    assert value["color"] in COLOR_SCHEMA["properties"]["color"]["enum"], value
    assert isinstance(value["bright"], bool), value


def _record(record_property, frontend: str) -> None:
    record_serving_capability(
        record_property,
        MODEL,
        "served",
        1,
        MAX_MODEL_LEN,
        MAX_NUM_SEQS,
        task="generate",
        frontend=frontend,
    )


@pytest.mark.uses_subprocess
def test_python_frontend_serves(python_responses, record_property) -> None:
    _record(record_property, "python")
    r = python_responses
    assert _ok(r, "tokenize_text")["body"]["tokens"] and _ok(r, "tokenize_chat")["body"]["tokens"]
    assert _text(_ok(r, "greedy_0")).startswith(" Paris")
    assert _text(_ok(r, "stream")) == _text(r["greedy_0"])
    assert _finish_reason(r["stream"]) == "length"
    assert _text(_ok(r, "chat")) and _text(_ok(r, "chat")) == _text(_ok(r, "chat_stream"))
    assert _finish_reason(_ok(r, "stop_excl")) == "stop"
    assert _text(_ok(r, "stop_incl")) == _text(r["stop_excl"]) + ":"
    _assert_json_schema_output(_ok(r, "json_schema"))
    assert len(_ok(r, "prompt_logprobs")["body"]["choices"][0]["prompt_logprobs"]) == len(
        r["tokenize_text"]["body"]["tokens"]
    )
    assert len(_ok(r, "n2")["body"]["choices"]) == 2
    overflow = r["max_tokens_overflow"]
    assert overflow["status"] == 400 and "max_model_len" in overflow["error"], overflow


@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "name",
    [
        "tokenize_text",
        "tokenize_chat",
        "greedy_0",
        "greedy_1",
        "stream",
        "chat",
        "chat_stream",
        "stop_excl",
        "stop_incl",
        "json_schema",
        "prompt_logprobs",
    ],
)
def test_rust_frontend_matches_python(
    name, python_responses, rust_responses, record_property
) -> None:
    _record(record_property, "rust")
    ref, got = _ok(python_responses, name), _ok(rust_responses, name)
    if name.startswith("tokenize"):
        assert got["body"]["tokens"] == ref["body"]["tokens"]
        return
    assert _text(got) == _text(ref)
    assert _finish_reason(got) == _finish_reason(ref)
    if "body" in ref:
        _assert_logprobs_match(name, ref, got)
    if name == "json_schema":
        _assert_json_schema_output(got)


@pytest.mark.uses_subprocess
def test_rust_frontend_known_differences(rust_responses, record_property) -> None:
    """Upstream's Rust frontend serves n=1 only and clamps max_tokens instead of a 400."""
    _record(record_property, "rust")
    n2 = rust_responses["n2"]
    assert n2["status"] == 400 and "Only n=1 is supported" in n2["error"], n2
    overflow = _ok(rust_responses, "max_tokens_overflow")
    usage = overflow["body"]["usage"]
    assert _finish_reason(overflow) == "length", overflow["body"]["choices"][0]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == MAX_MODEL_LEN, usage
