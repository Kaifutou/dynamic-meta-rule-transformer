from __future__ import annotations

import argparse

from metarule_demo.interactive import interactive_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive dialog demo for Dynamic Meta-Rule Transformer")
    parser.add_argument("--output-dir", default="runs/metarule_demo")
    parser.add_argument("--checkpoint-kind", choices=["best", "last"], default="best")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant names, e.g. full,static",
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()] or None
    interactive_demo(
        output_dir=args.output_dir,
        checkpoint_kind=args.checkpoint_kind,
        device=args.device,
        variants=variants,
    )


if __name__ == "__main__":
    main()
