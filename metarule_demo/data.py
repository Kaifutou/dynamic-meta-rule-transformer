from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import DemoConfig


PAD_ID = 0
SEP_ID = 1
QRY_ID = 2
SYMBOL_OFFSET = 3


@dataclass
class EpisodeBatch:
    tokens: torch.Tensor
    targets: torch.Tensor
    episode_meta: dict[str, torch.Tensor | int | bool]


class RuleShiftEpisodeDataset:
    def __init__(self, config: DemoConfig):
        self.config = config
        self._rngs = {
            "train": torch.Generator().manual_seed(config.seed + 1),
            "val": torch.Generator().manual_seed(config.seed + 2),
            "test": torch.Generator().manual_seed(config.seed + 3),
        }
        self.rule_bank = self._build_rule_bank()
        self.phase_to_rules, self.phase_is_repeat = self._build_phase_rules()

    def _build_rule_bank(self) -> torch.Tensor:
        rules = []
        g = torch.Generator().manual_seed(self.config.seed + 101)
        for _ in range(self.config.num_rules):
            rules.append(torch.randperm(self.config.vocab_symbols, generator=g))
        return torch.stack(rules, dim=0)

    def _build_phase_rules(self) -> tuple[list[list[int]], list[bool]]:
        group_count = max(self.config.phase_sequence) + 1
        required = group_count * self.config.phase_rule_count
        if required > self.config.num_rules:
            raise ValueError(
                "num_rules is too small for phase_sequence and phase_rule_count. "
                f"Need >= {required}, got {self.config.num_rules}."
            )

        group_to_rules: dict[int, list[int]] = {}
        cursor = 0
        for group_id in range(group_count):
            group_to_rules[group_id] = list(range(cursor, cursor + self.config.phase_rule_count))
            cursor += self.config.phase_rule_count

        phase_to_rules = []
        phase_is_repeat = []
        seen_groups: set[int] = set()
        for group_id in self.config.phase_sequence:
            phase_to_rules.append(group_to_rules[group_id])
            phase_is_repeat.append(group_id in seen_groups)
            seen_groups.add(group_id)

        return phase_to_rules, phase_is_repeat

    def sample_train_phase(self, step: int) -> int:
        return (step // self.config.phase_cycle) % self.config.phase_count

    def get_online_schedule(self, cycles: int | None = None) -> list[int]:
        cycles = cycles if cycles is not None else self.config.online_cycles
        schedule = []
        for _ in range(cycles):
            schedule.extend(range(self.config.phase_count))
        return schedule

    def sample_batch(
        self,
        split: str,
        phase_id: int,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> EpisodeBatch:
        if split not in self._rngs:
            raise ValueError(f"Unknown split: {split}")
        if phase_id < 0 or phase_id >= self.config.phase_count:
            raise ValueError(f"phase_id out of range: {phase_id}")

        batch_size = batch_size or self.config.batch_size
        device = device or self.config.device
        g = self._rngs[split]

        phase_rules = self.phase_to_rules[phase_id]
        rule_choice = torch.randint(len(phase_rules), size=(batch_size,), generator=g)
        rule_ids = torch.tensor(phase_rules, dtype=torch.long)[rule_choice]
        rule_maps = self.rule_bank[rule_ids]

        supports, queries = self._sample_symbols(batch_size=batch_size, generator=g)
        mapped_supports = rule_maps.gather(1, supports)
        mapped_queries = rule_maps.gather(1, queries.unsqueeze(1)).squeeze(1)

        tokens = torch.full(
            (batch_size, self.config.seq_len),
            fill_value=PAD_ID,
            dtype=torch.long,
        )

        for idx in range(self.config.support_size):
            pos = idx * 3
            tokens[:, pos] = supports[:, idx] + SYMBOL_OFFSET
            tokens[:, pos + 1] = mapped_supports[:, idx] + SYMBOL_OFFSET
            tokens[:, pos + 2] = SEP_ID

        tokens[:, -2] = QRY_ID
        tokens[:, -1] = queries + SYMBOL_OFFSET

        targets = mapped_queries + SYMBOL_OFFSET

        batch = EpisodeBatch(
            tokens=tokens.to(device),
            targets=targets.to(device),
            episode_meta={
                "phase_id": phase_id,
                "rule_ids": rule_ids.to(device),
                "is_repeat_phase": self.phase_is_repeat[phase_id],
            },
        )
        return batch

    def _sample_symbols(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support = torch.empty(batch_size, self.config.support_size, dtype=torch.long)
        queries = torch.empty(batch_size, dtype=torch.long)

        for i in range(batch_size):
            draw = torch.randperm(self.config.vocab_symbols, generator=generator)
            queries[i] = draw[0]
            support[i] = draw[1 : 1 + self.config.support_size]

        return support, queries
