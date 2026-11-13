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

"""Put vLLM's Rust frontend binary (``vllm-rs``) next to the installed ``vllm`` package.

The images have no Rust toolchain, so ``uv sync`` builds vLLM without ``vllm-rs``.
This takes it from the upstream x86_64 PyPI wheel of the installed vLLM version (the
engine protocol is version-coupled) and writes it where ``VLLM_RUST_FRONTEND_PATH=auto``
looks. Only the binary's zip member is fetched, by HTTP range reads (~60 MB of ~320 MB).
Run it with the project venv's interpreter. Drop it once the images build ``vllm-rs``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import sys
import urllib.request
import zipfile

MEMBER = "vllm/vllm-rs"


class _RangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main() -> int:
    if platform.machine() != "x86_64":
        print(f"no vllm-rs wheel for {platform.machine()}; Rust-frontend tests will skip")
        return 0
    version = importlib.metadata.version("vllm").split("+")[0]
    with urllib.request.urlopen(f"https://pypi.org/pypi/vllm/{version}/json", timeout=60) as r:
        files = json.load(r)["urls"]
    url = next(f["url"] for f in files if f["filename"].endswith("manylinux_2_28_x86_64.whl"))
    spec = importlib.util.find_spec("vllm")
    assert spec and spec.origin, "vllm is not importable from this interpreter"
    out = os.path.join(os.path.dirname(spec.origin), "vllm-rs")
    # ZipFile.read checks the member's CRC-32.
    data = zipfile.ZipFile(io.BufferedReader(_RangeFile(url), buffer_size=8 << 20)).read(MEMBER)
    with open(out, "wb") as f:
        f.write(data)
    os.chmod(out, 0o755)
    print(f"wrote {out} ({len(data)} bytes) from {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
