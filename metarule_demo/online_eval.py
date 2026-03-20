from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import DemoConfig
from .data import RuleShiftEpisodeDataset
from .model import MetaRuleTransformer, VariantFlags


def _load_variant_from_checkpoint(checkpoint: dict) -> VariantFlags:
    variant = checkpoint["variant"]
    return VariantFlags(**variant)


def _merge_eval_overrides(train_cfg: DemoConfig, runtime_cfg: DemoConfig) -> DemoConfig:
    merged = DemoConfig.from_dict(train_cfg.to_dict())
    merged.output_dir = runtime_cfg.output_dir
    merged.device = runtime_cfg.device
    merged.seed = runtime_cfg.seed
    merged.online_episodes_per_phase = runtime_cfg.online_episodes_per_phase
    merged.online_cycles = runtime_cfg.online_cycles
    merged.online_batch_size = runtime_cfg.online_batch_size
    merged.adaptation_window = runtime_cfg.adaptation_window
    return merged


def _summarize_online(
    rows: list[dict[str, float | int | str | bool]],
    adaptation_window: int,
) -> dict[str, dict[str, float | dict[str, float]]]:
    by_variant: dict[str, list[dict[str, float | int | str | bool]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)

    summary: dict[str, dict[str, float | dict[str, float]]] = {}

    for variant, variant_rows in by_variant.items():
        occ: dict[tuple[int, int], list[dict[str, float | int | str | bool]]] = defaultdict(list)
        for row in variant_rows:
            occ[(int(row["cycle_id"]), int(row["phase_id"]))].append(row)

        phase_group_first_early: dict[int, float] = {}
        recovery_values: list[float] = []
        shift_early_values: list[float] = []
        cold_start_gains: list[float] = []
        group_gains: dict[int, list[float]] = defaultdict(list)

        for key in sorted(occ.keys()):
            part = occ[key]
            acc = [float(r["accuracy"]) for r in part]
            early = sum(acc[:adaptation_window]) / max(min(len(acc), adaptation_window), 1)
            late = sum(acc[-adaptation_window:]) / max(min(len(acc), adaptation_window), 1)
            recovery_values.append(late - early)
            shift_early_values.append(early)

            group = int(part[0]["rule_group"])
            if group not in phase_group_first_early:
                phase_group_first_early[group] = early
            else:
                gain = early - phase_group_first_early[group]
                cold_start_gains.append(gain)
                group_gains[group].append(gain)

        summary[variant] = {
            "mean_shift_early_acc": float(sum(shift_early_values) / max(len(shift_early_values), 1)),
            "mean_shift_recovery_gain": float(sum(recovery_values) / max(len(recovery_values), 1)),
            "mean_cold_start_gain": float(sum(cold_start_gains) / max(len(cold_start_gains), 1)),
            "cold_start_gain_by_group": {
                str(k): float(sum(v) / max(len(v), 1)) for k, v in group_gains.items()
            },
        }

    return summary


def run_online_eval(
    checkpoint_dir: str | Path,
    config: DemoConfig,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)

    checkpoints = sorted(checkpoint_dir.glob("*_best.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No '*_best.pt' checkpoints found in {checkpoint_dir}")

    first_payload = torch.load(checkpoints[0], map_location="cpu")
    train_cfg = DemoConfig.from_dict(first_payload["config"])
    eval_cfg = _merge_eval_overrides(train_cfg=train_cfg, runtime_cfg=config)

    output_dir = eval_cfg.ensure_output_dir()
    online_csv = output_dir / "online_metrics.csv"
    online_summary_json = output_dir / "online_summary.json"

    dataset = RuleShiftEpisodeDataset(eval_cfg)
    device = torch.device(eval_cfg.device)
    rows: list[dict[str, float | int | str | bool]] = []

    for ckpt_path in checkpoints:
        payload = torch.load(ckpt_path, map_location=device)
        variant = _load_variant_from_checkpoint(payload)

        model = MetaRuleTransformer(config=eval_cfg, variant=variant).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        state = model.init_state(eval_cfg.online_batch_size, device=device)
        global_episode = 0

        for cycle_id in range(eval_cfg.online_cycles):
            for phase_id in range(eval_cfg.phase_count):
                for episode_in_phase in range(eval_cfg.online_episodes_per_phase):
                    batch = dataset.sample_batch(
                        split="test",
                        phase_id=phase_id,
                        batch_size=eval_cfg.online_batch_size,
                        device=device,
                    )
                    with torch.no_grad():
                        logits, state, _ = model(
                            batch.tokens,
                            state,
                            feedback={"targets": batch.targets},
                        )

                    loss = F.cross_entropy(logits, batch.targets)
                    acc = (logits.argmax(dim=-1) == batch.targets).float().mean()
                    rule_group = eval_cfg.phase_sequence[phase_id]

                    row = {
                        "variant": variant.name,
                        "global_episode": global_episode,
                        "cycle_id": cycle_id,
                        "phase_id": phase_id,
                        "rule_group": rule_group,
                        "episode_in_phase": episode_in_phase,
                        "is_repeat_phase": bool(dataset.phase_is_repeat[phase_id] or cycle_id > 0),
                        "loss": float(loss.item()),
                        "accuracy": float(acc.item()),
                    }
                    rows.append(row)
                    global_episode += 1

    with online_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "global_episode",
                "cycle_id",
                "phase_id",
                "rule_group",
                "episode_in_phase",
                "is_repeat_phase",
                "loss",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize_online(rows, adaptation_window=eval_cfg.adaptation_window)
    with online_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return online_summary_json
