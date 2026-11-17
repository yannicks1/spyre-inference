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

"""The host RoPE cache must come out bit-identical in every process."""

import hashlib
import os
import subprocess
import sys

import spyre_inference

_COS_TABLE = """
import hashlib, torch
torch.set_num_threads(8)
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 256, 2, dtype=torch.float) / 256))
freqs = torch.einsum("i,j -> ij", torch.arange(32768, dtype=torch.float), inv_freq)
print(hashlib.sha1(freqs.cos().numpy().tobytes()).hexdigest())
"""


def test_register_defaults_mkl_cbwr(monkeypatch):
    monkeypatch.delenv("MKL_CBWR", raising=False)
    spyre_inference.register()
    assert os.environ["MKL_CBWR"] == "COMPATIBLE"


def test_register_keeps_an_explicit_mkl_cbwr(monkeypatch):
    monkeypatch.setenv("MKL_CBWR", "AVX2")
    spyre_inference.register()
    assert os.environ["MKL_CBWR"] == "AVX2"


def test_host_cos_table_is_identical_across_processes():
    env = {**os.environ, "MKL_CBWR": "COMPATIBLE"}
    digests = {
        subprocess.run(
            [sys.executable, "-c", _COS_TABLE], env=env, capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(digests) == 1, digests
    assert all(len(d) == len(hashlib.sha1().hexdigest()) for d in digests)
