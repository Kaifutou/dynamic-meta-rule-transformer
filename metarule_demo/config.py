from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class DemoConfig:
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "runs/metarule_demo"

    vocab_symbols: int = 32
    support_size: int = 4
    num_rules: int = 16
    phase_rule_count: int = 4
    phase_sequence: tuple[int, ...] = (0, 1, 2, 0, 3, 1)

    batch_size: int = 32
    eval_batch_size: int = 32
    online_batch_size: int = 32
    train_steps: int = 320
    eval_interval: int = 40
    phase_cycle: int = 20
    val_episodes_per_phase: int = 12
    online_episodes_per_phase: int = 20
    online_cycles: int = 2
    adaptation_window: int = 5

    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 4
    lora_rank: int = 4
    d_memory: int = 64
    fast_slots: int = 8
    slow_slots: int = 16
    dropout: float = 0.1
    dynamic_scale: float = 0.05
    lambda_fast: float = 0.95

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dynamic_reg_weight: float = 1e-3
    gate_reg_weight: float = 1e-3

    @property
    def vocab_size(self) -> int:
        return self.vocab_symbols + 3

    @property
    def seq_len(self) -> int:
        return self.support_size * 3 + 2

    @property
    def phase_count(self) -> int:
        return len(self.phase_sequence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DemoConfig":
        if "phase_sequence" in data and isinstance(data["phase_sequence"], list):
            data = dict(data)
            data["phase_sequence"] = tuple(data["phase_sequence"])
        return cls(**data)

    def ensure_output_dir(self) -> Path:
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        return output
