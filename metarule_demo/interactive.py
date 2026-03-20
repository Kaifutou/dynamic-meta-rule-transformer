from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .config import DemoConfig
from .data import QRY_ID, SEP_ID, SYMBOL_OFFSET, RuleShiftEpisodeDataset
from .memory import MemoryState
from .model import MetaRuleTransformer, VariantFlags


@dataclass
class LoadedModel:
    name: str
    model: MetaRuleTransformer
    state: MemoryState


def _parse_supports(text: str, expected: int, vocab_symbols: int) -> list[tuple[int, int]]:
    items = [part.strip() for part in text.split(",") if part.strip()]
    if len(items) != expected:
        raise ValueError(f"Need exactly {expected} support pairs, got {len(items)}")

    pairs: list[tuple[int, int]] = []
    for item in items:
        if "->" in item:
            left, right = item.split("->", maxsplit=1)
        elif ":" in item:
            left, right = item.split(":", maxsplit=1)
        else:
            raise ValueError(f"Bad pair '{item}'. Use x->y")

        x = int(left.strip())
        y = int(right.strip())
        if not (0 <= x < vocab_symbols and 0 <= y < vocab_symbols):
            raise ValueError(f"Pair '{item}' out of range [0, {vocab_symbols - 1}]")
        pairs.append((x, y))

    return pairs


def _build_tokens(config: DemoConfig, pairs: list[tuple[int, int]], query: int, device: torch.device) -> torch.Tensor:
    if len(pairs) != config.support_size:
        raise ValueError(f"pairs length must be {config.support_size}")

    tokens = torch.zeros((1, config.seq_len), dtype=torch.long, device=device)
    for idx, (x, y) in enumerate(pairs):
        pos = idx * 3
        tokens[:, pos] = x + SYMBOL_OFFSET
        tokens[:, pos + 1] = y + SYMBOL_OFFSET
        tokens[:, pos + 2] = SEP_ID

    tokens[:, -2] = QRY_ID
    tokens[:, -1] = query + SYMBOL_OFFSET
    return tokens


def _decode_tokens(config: DemoConfig, token_row: torch.Tensor) -> tuple[list[tuple[int, int]], int]:
    pairs: list[tuple[int, int]] = []
    for idx in range(config.support_size):
        pos = idx * 3
        x = int(token_row[pos].item()) - SYMBOL_OFFSET
        y = int(token_row[pos + 1].item()) - SYMBOL_OFFSET
        pairs.append((x, y))
    query = int(token_row[-1].item()) - SYMBOL_OFFSET
    return pairs, query


def _load_models(
    output_dir: Path,
    checkpoint_kind: str,
    device: torch.device,
    variants: Iterable[str] | None = None,
) -> tuple[DemoConfig, dict[str, LoadedModel]]:
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    suffix = "_best.pt" if checkpoint_kind == "best" else "_last.pt"
    checkpoint_paths = sorted(ckpt_dir.glob(f"*{suffix}"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints with suffix '{suffix}' in {ckpt_dir}")

    wanted = set(variants) if variants else None
    loaded: dict[str, LoadedModel] = {}
    base_config: DemoConfig | None = None

    for ckpt_path in checkpoint_paths:
        payload = torch.load(ckpt_path, map_location=device)
        variant = VariantFlags(**payload["variant"])
        if wanted is not None and variant.name not in wanted:
            continue

        config = DemoConfig.from_dict(payload["config"])
        config.device = str(device)

        if base_config is None:
            base_config = config

        model = MetaRuleTransformer(config=config, variant=variant).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        state = model.init_state(batch_size=1, device=device)
        loaded[variant.name] = LoadedModel(name=variant.name, model=model, state=state)

    if not loaded:
        raise ValueError("No model loaded. Check variants filter and checkpoints.")
    if base_config is None:
        raise RuntimeError("Internal error: missing config.")

    return base_config, loaded


def _run_and_print(
    models: dict[str, LoadedModel],
    tokens: torch.Tensor,
    target: int | None,
    device: torch.device,
) -> None:
    feedback = None
    if target is not None:
        feedback = {"targets": torch.tensor([target + SYMBOL_OFFSET], device=device)}

    print("\nvariant              pred  conf    alpha   rho     delta_norm  correct")
    print("---------------------------------------------------------------------")
    with torch.no_grad():
        for name in sorted(models.keys()):
            loaded = models[name]
            logits, next_state, aux = loaded.model(tokens, loaded.state, feedback=feedback)
            probs = torch.softmax(logits, dim=-1)
            pred_token = int(logits.argmax(dim=-1).item())
            pred = pred_token - SYMBOL_OFFSET
            conf = float(probs.max().item())
            alpha = float(aux["alpha"].mean().item())
            rho = float(aux["rho"].mean().item())
            delta_norm = float(aux["delta_norm"].item())
            correct = "-" if target is None else ("Y" if pred == target else "N")

            print(
                f"{name:<20} {pred:>4}  {conf:>0.4f}  {alpha:>0.4f}  {rho:>0.4f}  {delta_norm:>0.6f}    {correct}"
            )
            loaded.state = next_state.detach()
    print()


def interactive_demo(
    output_dir: str | Path,
    checkpoint_kind: str = "best",
    device: str = "cpu",
    variants: list[str] | None = None,
) -> None:
    output_dir = Path(output_dir)
    torch_device = torch.device(device)

    config, models = _load_models(
        output_dir=output_dir,
        checkpoint_kind=checkpoint_kind,
        device=torch_device,
        variants=variants,
    )
    dataset = RuleShiftEpisodeDataset(config)
    sample_phase_cursor = 0

    print("=== Interactive Meta-Rule Demo ===")
    print(f"Loaded variants: {', '.join(models.keys())}")
    print(f"support_size={config.support_size}, symbol_range=[0, {config.vocab_symbols - 1}]")
    print("Input format example: 1->3, 5->2, 7->7, 9->0")
    print("Commands: help, reset, sample, :sample [phase], exit")

    while True:
        try:
            supports_text = input("supports> ").strip()
        except EOFError:
            print("\nBye.")
            break

        cmd = supports_text.lower()
        if cmd in {"exit", "quit", "q"}:
            print("Bye.")
            break
        if cmd == "help":
            print("Enter support pairs, then query, then optional target.")
            print("supports format: x->y, x->y, ... (exactly support_size pairs)")
            print("sample or :sample [phase]: generate one in-distribution sample and evaluate.")
            continue
        if cmd == "reset":
            for loaded in models.values():
                loaded.state = loaded.model.init_state(batch_size=1, device=torch_device)
            print("Memory state reset for all variants.")
            continue
        if not supports_text:
            continue

        try:
            if cmd.startswith("sample") or cmd.startswith(":sample"):
                parts = supports_text.replace(":", "").split()
                if len(parts) >= 2:
                    phase_id = int(parts[1])
                else:
                    phase_id = sample_phase_cursor
                    sample_phase_cursor = (sample_phase_cursor + 1) % config.phase_count

                if not (0 <= phase_id < config.phase_count):
                    raise ValueError(f"phase_id out of range [0, {config.phase_count - 1}]")

                batch = dataset.sample_batch(split="test", phase_id=phase_id, batch_size=1, device=torch_device)
                pairs, query = _decode_tokens(config, batch.tokens[0])
                target = int(batch.targets[0].item()) - SYMBOL_OFFSET

                print(
                    f"Auto sample (phase={phase_id}, group={config.phase_sequence[phase_id]}, "
                    f"repeat_phase={dataset.phase_is_repeat[phase_id]})"
                )
                print(f"supports: {', '.join([f'{x}->{y}' for x, y in pairs])}")
                print(f"query x: {query}")
                print(f"target y: {target}")

                _run_and_print(models=models, tokens=batch.tokens, target=target, device=torch_device)
                continue

            pairs = _parse_supports(
                supports_text,
                expected=config.support_size,
                vocab_symbols=config.vocab_symbols,
            )

            query_raw = input("query x> ").strip()
            query = int(query_raw)
            if not (0 <= query < config.vocab_symbols):
                raise ValueError(f"query out of range [0, {config.vocab_symbols - 1}]")

            target_raw = input("target y (optional)> ").strip()
            target = None
            if target_raw:
                target = int(target_raw)
                if not (0 <= target < config.vocab_symbols):
                    raise ValueError(f"target out of range [0, {config.vocab_symbols - 1}]")

            tokens = _build_tokens(config=config, pairs=pairs, query=query, device=torch_device)
            _run_and_print(models=models, tokens=tokens, target=target, device=torch_device)
        except Exception as exc:
            print(f"Input error: {exc}")
            print("Try 'help' for format details.\n")
