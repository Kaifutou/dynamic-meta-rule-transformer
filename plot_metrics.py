from __future__ import annotations

import argparse

from metarule_demo.plotting import plot_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics for Dynamic Meta-Rule Transformer demo")
    parser.add_argument("--log-dir", default="runs/metarule_demo")
    args = parser.parse_args()

    plot_metrics(args.log_dir)


if __name__ == "__main__":
    main()
