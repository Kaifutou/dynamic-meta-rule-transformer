from __future__ import annotations

import argparse
from pathlib import Path

from metarule_demo.config import DemoConfig
from metarule_demo.online_eval import run_online_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online evaluation for Dynamic Meta-Rule Transformer demo")
    parser.add_argument("--output-dir", default="runs/metarule_demo")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online-episodes-per-phase", type=int, default=20)
    parser.add_argument("--online-cycles", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = DemoConfig(
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        online_episodes_per_phase=args.online_episodes_per_phase,
        online_cycles=args.online_cycles,
        online_batch_size=args.batch_size,
    )

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else Path(args.output_dir) / "checkpoints"
    )
    run_online_eval(checkpoint_dir=checkpoint_dir, config=config)


if __name__ == "__main__":
    main()
