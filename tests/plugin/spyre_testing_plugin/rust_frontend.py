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

"""Locate vLLM's Rust frontend binary (``vllm-rs``) for tests that serve through it.

``vllm-rs`` is an optional setuptools-rust extension: on an image without a Rust
toolchain ``uv sync`` builds vLLM without it and says nothing, so tests that need it
skip instead of failing.
"""

from __future__ import annotations

import importlib.util
import os

import pytest


def _candidate() -> str:
    """The path vLLM would run, as ``vllm.envs._resolve_rust_cli_path`` resolves it."""
    raw = os.environ.get("VLLM_RUST_FRONTEND_PATH", "auto")
    if raw.lower() not in ("auto", "1", "true"):
        return raw
    spec = importlib.util.find_spec("vllm")
    pkg_dir = os.path.dirname(spec.origin) if spec and spec.origin else ""
    return os.path.join(pkg_dir, "vllm-rs")


def require_rust_frontend() -> str:
    """Return the ``vllm-rs`` path, or skip the calling test when there is no binary."""
    path = _candidate()
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        pytest.skip(
            f"no vllm-rs binary at {path}: vLLM was built without the Rust frontend "
            "(no Rust toolchain on the image); set VLLM_RUST_FRONTEND_PATH to one"
        )
    return path
