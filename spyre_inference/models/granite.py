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

"""Granite adaptations: per-rank MLP width padding and the TP4 prefill work split.

Port of hf-adapters ``super/granite-tp4-stable-20261020`` (hf-adapters #700). Each
rank's gate/up halves are zero-padded to a multiple of four sticks so the 32-core
pointwise chain splits evenly; down_proj still reads only the original channels.
Granite 3.3 8B at TP4 additionally gets ``spyre_hint`` work divisions on the
512-row prefill MLP.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models.granite import GraniteForCausalLM, GraniteMLP

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Four fp16 sticks.
_WIDTH_MULTIPLE = 256
_PREFILL_ROWS = 512


def _forced_division_config(config, tp_size: int) -> bool:
    """Only the Granite 3.3 8B TP4 shapes hf-adapters validated."""
    import os

    return (
        os.environ.get("JR_EXP_HINTS", "1") == "1"
        and tp_size == 4
        and config.hidden_size == 4096
        and config.intermediate_size == 12800
        and config.num_hidden_layers == 40
    )


class SpyreGraniteMLP(GraniteMLP):
    """Upstream MLP over a gate_up whose per-rank halves may be padded."""

    _spyre_width: int | None = None
    _spyre_forced_division: bool = False

    def pad_width(self, multiple: int) -> None:
        """Zero-pad each per-rank gate/up half of the loaded weight to ``multiple``."""
        proj = self.gate_up_proj
        if not isinstance(proj.quant_method, UnquantizedLinearMethod) or proj.bias is not None:
            return
        weight = proj.weight.data
        width = weight.shape[0] // 2
        padded = -(-width // multiple) * multiple
        if padded == width:
            return
        gate, up = weight.split(width)
        rows = (0, 0, 0, padded - width)
        proj.weight = Parameter(
            torch.cat([F.pad(gate, rows), F.pad(up, rows)]), requires_grad=False
        )
        self._spyre_width = width

    def forward(self, x):
        if self._spyre_width is None:
            return super().forward(x)
        if self._spyre_forced_division and tuple(x.shape) == (_PREFILL_ROWS, x.shape[-1]):
            return self._forward_forced(x)
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up)[..., : self._spyre_width])
        return x

    def _forward_forced(self, x):
        from torch_spyre._inductor.propagate_hints import spyre_hint

        with spyre_hint(
            named_dims=["granite_mlp_rows", "granite_mlp_gate_up"],
            work_div={"granite_mlp_rows": 8, "granite_mlp_gate_up": 4},
        ):
            gate_up, _ = self.gate_up_proj(x)
        with spyre_hint(
            named_dims=["granite_mlp_rows", "granite_mlp_channels"],
            work_div={"granite_mlp_rows": 8, "granite_mlp_channels": 4},
        ):
            product = self.act_fn(gate_up)
        with spyre_hint(
            named_dims=["granite_mlp_rows", "granite_mlp_hidden"],
            work_div={"granite_mlp_rows": 8, "granite_mlp_hidden": 4},
        ):
            x, _ = self.down_proj(product[..., : self._spyre_width])
        return x


class SpyreGraniteForCausalLM(GraniteForCausalLM):
    """Retypes every decoder MLP to ``SpyreGraniteMLP`` and pads it after loading."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        forced = _forced_division_config(self.config, get_tensor_model_parallel_world_size())
        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if type(mlp) is not GraniteMLP:
                raise RuntimeError(f"Expected GraniteMLP, got {type(mlp).__name__}")
            mlp.__class__ = SpyreGraniteMLP
            mlp._spyre_forced_division = forced

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        padded = 0
        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, SpyreGraniteMLP):
                mlp.pad_width(_WIDTH_MULTIPLE)
                padded += mlp._spyre_width is not None
        if padded:
            logger.info(
                "Padded %d Granite MLPs to a per-rank multiple of %d (forced prefill split: %s).",
                padded,
                _WIDTH_MULTIPLE,
                _forced_division_config(self.config, get_tensor_model_parallel_world_size()),
            )
        return loaded
