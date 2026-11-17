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

"""Attention over a cache that stores each KV head as ``head_split`` narrower page heads.

A layer whose head is wider than its pages' (gemma-4's 512-wide global heads in a slot of
256-wide sliding pages) keeps part ``e`` of KV head ``h``, dims ``[e * d, (e + 1) * d)``, in
page head ``e * num_kv_heads + h``. A Spyre tensor's stick axis cannot be re-viewed, so the
kernels contract each part on its own: the score sums the parts' dot products and the
output concatenates the parts' ``probs @ V``. Part-major, so a part's heads are a contiguous
slice of the page-head axis; a head-major order reads them at a stride the backend rejects.
"""

import torch


def split_scores(q: torch.Tensor, k_page: torch.Tensor, head_split: int, kv_axis: int):
    """``q @ k_page^T`` for ``k_page`` holding ``head_split`` page heads per KV head.

    ``kv_axis`` is ``k_page``'s page-head axis; ``q``'s last axis is the full head.
    """
    kv = k_page.shape[kv_axis] // head_split
    d = q.shape[-1] // head_split
    # Per part, then summed: concatenating the parts back to a full-width K does not
    # compile in the decode kernels ("store value of unexpected type Constant").
    scores = None
    for e in range(head_split):
        k = k_page.narrow(kv_axis, e * kv, kv)
        part = torch.matmul(q[..., e * d : (e + 1) * d], k.transpose(-2, -1))
        scores = part if scores is None else scores + part
    return scores


def split_pv(probs: torch.Tensor, v_page: torch.Tensor, head_split: int, kv_axis: int):
    """``probs @ v_page`` with the page heads of one KV head concatenated back to its width."""
    kv = v_page.shape[kv_axis] // head_split
    return torch.cat(
        [torch.matmul(probs, v_page.narrow(kv_axis, e * kv, kv)) for e in range(head_split)],
        dim=-1,
    )
