from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import DemoConfig
from .data import RuleShiftEpisodeDataset
from .model import MetaRuleTransformer, VariantFlags


VARIANT_SPECS = {
    "full": VariantFlags(
        name="full",
        dynamic_enabled=True,
        use_fast_memory=True,
        use_slow_memory=True,
        allow_memory_write=True,
    ),
    "static": VariantFlags(
        name="static",
        dynamic_enabled=False,
        use_fast_memory=False,
        use_slow_memory=False,
        allow_memory_write=False,
    ),
    "fast_only": VariantFlags(
        name="fast_only",
        dynamic_enabled=True,
        use_fast_memory=True,
        use_slow_memory=False,
        allow_memory_write=True,
    ),
    "memory_no_dynamic": VariantFlags(
        name="memory_no_dynamic",
        dynamic_enabled=False,
        use_fast_memory=True,
        use_slow_memory=True,
        allow_memory_write=True,
    ),
}


def evaluate_model(
    model: MetaRuleTransformer,
    dataset: RuleShiftEpisodeDataset,
    config: DemoConfig,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    state = model.init_state(config.eval_batch_size, device=device)

    with torch.no_grad():
        for phase_id in range(config.phase_count):
            for _ in range(config.val_episodes_per_phase):
                batch = dataset.sample_batch(
                    split="val",
                    phase_id=phase_id,
                    batch_size=config.eval_batch_size,
                    device=device,
                )
                logits, state, _ = model(
                    batch.tokens,
                    state,
                    feedback={"targets": batch.targets},
                )
                loss = F.cross_entropy(logits, batch.targets)
                acc = (logits.argmax(dim=-1) == batch.targets).float().mean()

                total_loss += float(loss.item())
                total_acc += float(acc.item())
                total_count += 1

    return total_loss / max(total_count, 1), total_acc / max(total_count, 1)


def run_train(config: DemoConfig) -> Path:
    output_dir = config.ensure_output_dir()
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    dataset = RuleShiftEpisodeDataset(config)
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    train_csv = output_dir / "train_metrics.csv"
    val_csv = output_dir / "val_metrics.csv"

    with train_csv.open("w", newline="", encoding="utf-8") as train_f, val_csv.open(
        "w", newline="", encoding="utf-8"
    ) as val_f:
        train_writer = csv.DictWriter(
            train_f,
            fieldnames=[
                "variant",
                "step",
                "phase_id",
                "loss",
                "task_loss",
                "acc",
                "delta_norm",
                "alpha_mean",
                "rho_mean",
            ],
        )
        train_writer.writeheader()

        val_writer = csv.DictWriter(
            val_f,
            fieldnames=["variant", "step", "val_loss", "val_acc"],
        )
        val_writer.writeheader()

        summary: dict[str, dict[str, float | str]] = {}

        for variant_name, variant in VARIANT_SPECS.items():
            model = MetaRuleTransformer(config=config, variant=variant).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            state = model.init_state(config.batch_size, device=device)
            best_val_acc = -1.0
            best_ckpt = checkpoints_dir / f"{variant_name}_best.pt"

            for step in range(1, config.train_steps + 1):
                model.train()
                phase_id = dataset.sample_train_phase(step - 1)
                batch = dataset.sample_batch(
                    split="train",
                    phase_id=phase_id,
                    batch_size=config.batch_size,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)
                logits, next_state, aux = model(
                    batch.tokens,
                    state,
                    feedback={"targets": batch.targets},
                )

                task_loss = F.cross_entropy(logits, batch.targets)
                delta_norm = aux["delta_norm"]
                alpha_mean = aux["alpha"].mean()
                rho_mean = aux["rho"].mean()

                loss = (
                    task_loss
                    + config.dynamic_reg_weight * delta_norm
                    + config.gate_reg_weight * (alpha_mean + rho_mean)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                state = next_state.detach()

                acc = (logits.argmax(dim=-1) == batch.targets).float().mean().item()
                train_writer.writerow(
                    {
                        "variant": variant_name,
                        "step": step,
                        "phase_id": phase_id,
                        "loss": float(loss.item()),
                        "task_loss": float(task_loss.item()),
                        "acc": float(acc),
                        "delta_norm": float(delta_norm.item()),
                        "alpha_mean": float(alpha_mean.item()),
                        "rho_mean": float(rho_mean.item()),
                    }
                )

                if step % config.eval_interval == 0 or step == config.train_steps:
                    val_loss, val_acc = evaluate_model(model, dataset, config, device)
                    val_writer.writerow(
                        {
                            "variant": variant_name,
                            "step": step,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                        }
                    )

                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        torch.save(
                            {
                                "config": config.to_dict(),
                                "variant": asdict(variant),
                                "model_state": model.state_dict(),
                                "step": step,
                                "val_acc": val_acc,
                            },
                            best_ckpt,
                        )

            last_ckpt = checkpoints_dir / f"{variant_name}_last.pt"
            torch.save(
                {
                    "config": config.to_dict(),
                    "variant": asdict(variant),
                    "model_state": model.state_dict(),
                    "step": config.train_steps,
                    "val_acc": best_val_acc,
                },
                last_ckpt,
            )
            summary[variant_name] = {
                "best_val_acc": best_val_acc,
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
            }

    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return output_dir
