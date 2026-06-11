from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.viability.config import load_config
from src.viability.doe import generate_doe
from src.viability.evaluator import evaluate_design, evaluate_designs_parallel
from src.viability.io import run_output_dir, write_config_resolved, write_table
from src.viability.policy import PolicyDesign


def main() -> int:
    parser = argparse.ArgumentParser(description="Viability analysis prototype commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate-design", help="Evaluate one policy design")
    evaluate_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    evaluate_parser.add_argument(
        "--design-json",
        required=True,
        help="JSON object with constant policy values",
    )

    doe_parser = subparsers.add_parser("generate-doe", help="Generate input combinations only")
    doe_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    doe_parser.add_argument("--n", type=int, default=None, help="Override doe.n_initial")
    doe_parser.add_argument("--method", default=None, help="Override doe.method")
    doe_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for doe.csv (default: run output dir from config)",
    )
    doe_parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print CSV to stdout in addition to writing output files",
    )

    run_doe_parser = subparsers.add_parser("run-doe", help="Generate and evaluate input combinations")
    run_doe_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    run_doe_parser.add_argument("--n", type=int, default=None, help="Override doe.n_initial")
    run_doe_parser.add_argument("--method", default=None, help="Override doe.method")
    run_doe_parser.add_argument("--workers", type=int, default=None, help="Override run.workers")
    run_doe_parser.add_argument(
        "--output-dir",
        default=None,
        help="Run output directory (default: run output dir from config)",
    )
    run_doe_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Flush evaluation checkpoint batches every N completed designs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "evaluate-design":
        design_values = json.loads(args.design_json)
        design = PolicyDesign.from_mapping(design_values, config.policy)
        result = evaluate_design(design, config)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.status == "ok" else 1

    output_dir = Path(args.output_dir) if args.output_dir else run_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "generate-doe":
        df = generate_doe(config, n=args.n, method=args.method)
        write_config_resolved(config, output_dir)
        write_table(df, output_dir / "doe.csv", prefer_parquet=False)
        if args.stdout:
            print(df.to_csv(index=False), end="")
        return 0

    if args.command == "run-doe":
        write_config_resolved(config, output_dir)
        designs = generate_doe(config, n=args.n, method=args.method)
        write_table(designs, output_dir / "doe.csv", prefer_parquet=False)
        results = evaluate_designs_parallel(
            designs,
            config,
            workers=args.workers,
            checkpoint_dir=output_dir / "checkpoints",
            checkpoint_every=args.checkpoint_every,
        )
        write_table(results, output_dir / "evaluations.parquet")
        all_ok = bool(len(results) > 0 and (results["status"] == "ok").all())
        return 0 if all_ok else 1

    raise ValueError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
