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

"""`vllm serve` under the Rust frontend refuses a config it cannot serve, before warmup."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest
from spyre_testing_plugin.rust_frontend import require_rust_frontend
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

POOLING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


@pytest.mark.uses_subprocess
def test_rust_frontend_refuses_pooling_model() -> None:
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        VLLM_USE_RUST_FRONTEND="1",
        VLLM_RUST_FRONTEND_PATH=require_rust_frontend(),
    )
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", POOLING_MODEL]
    # check_and_update_config raises in the supervisor, before an engine claims the card.
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        # The server came up instead: take its engine down with it, so the card is freed.
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        assert wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
        pytest.fail(f"server started instead of refusing the pooling model:\n{output[-4000:]}")
    assert proc.returncode != 0, output[-4000:]
    assert "serves no pooling routes" in output, output[-4000:]
