from __future__ import annotations

from metarule_demo.config import DemoConfig
from metarule_demo.data import RuleShiftEpisodeDataset


def test_sample_batch_shapes() -> None:
    cfg = DemoConfig(
        vocab_symbols=12,
        support_size=3,
        num_rules=8,
        phase_rule_count=2,
        phase_sequence=(0, 1, 0),
        batch_size=5,
    )
    ds = RuleShiftEpisodeDataset(cfg)
    batch = ds.sample_batch(split="train", phase_id=0, batch_size=5, device="cpu")

    assert batch.tokens.shape == (5, cfg.seq_len)
    assert batch.targets.shape == (5,)
    assert batch.episode_meta["phase_id"] == 0
