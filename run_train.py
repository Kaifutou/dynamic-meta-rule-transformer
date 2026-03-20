from __future__ import annotations

import argparse

from metarule_demo.config import DemoConfig
from metarule_demo.train import run_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dynamic Meta-Rule Transformer demo")
    parser.add_argument("--output-dir", default="runs/metarule_demo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps", type=int, default=320)
    parser.add_argument("--eval-interval", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    config = DemoConfig(
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_batch_size=args.batch_size,
        online_batch_size=args.batch_size,
    )
    run_train(config)


if __name__ == "__main__":
    main()
