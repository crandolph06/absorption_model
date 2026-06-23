from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.viability.active_learning import run_active_learning_from_files
from src.viability.config import load_config
from src.viability.doe import generate_doe
from src.viability.dynamic_search import (
    run_dynamic_policy_diagnostic,
    run_dynamic_policy_refinement,
    run_dynamic_policy_search,
)
from src.viability.dynamic_bound_relaxation import run_dynamic_bound_relaxation_study
from src.viability.dynamic_ipug import run_dynamic_ipug_diagnostic
from src.viability.dynamic_trajectory_artifacts import (
    run_dynamic_trajectory_artifacts,
)
from src.viability.evaluator import evaluate_design, evaluate_designs_parallel
from src.viability.io import run_output_dir, write_config_resolved, write_table
from src.viability.plots import run_envelope_plots_from_files
from src.viability.policy import PolicyDesign
from src.viability.relaxation import (
    DEFAULT_RELAXATION_CONSTRAINTS,
    run_dynamic_relaxation_study,
)
from src.viability.report import write_viability_report_from_files
from src.viability.search import run_surrogate_search_from_files, verify_candidates_from_file
from src.viability.surrogate import (
    fit_surrogates_from_file,
    run_gpr_convergence_from_file,
    write_holdout_selection_from_file,
    write_gpr_prediction_overlay_plot,
)


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
    run_doe_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override run.workers; direct physics batches should normally use multiple workers",
    )
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
    fit_parser = subparsers.add_parser("fit-surrogate", help="Fit baseline viability surrogates")
    fit_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    fit_parser.add_argument("--evaluations", required=True, help="Evaluations parquet or CSV path")
    fit_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for surrogate artifacts (default: directory containing evaluations)",
    )
    fit_parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.1,
        help="Absolute phi threshold for boundary MAE metric",
    )
    fit_parser.add_argument(
        "--no-gpr",
        action="store_true",
        help="Skip the optional phi Gaussian process model",
    )
    converge_parser = subparsers.add_parser(
        "converge-surrogate",
        help="Run a fixed-holdout GPR convergence study from evaluated rows",
    )
    converge_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    converge_parser.add_argument("--evaluations", required=True, help="Evaluations parquet or CSV path")
    converge_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for convergence artifacts (default: directory containing evaluations)",
    )
    converge_parser.add_argument(
        "--train-sizes",
        default=None,
        help="Comma-separated training sizes; default uses powers of two plus max train size",
    )
    converge_parser.add_argument("--holdout-fraction", type=float, default=0.2)
    converge_parser.add_argument("--target-r2", type=float, default=None)
    converge_parser.add_argument("--target-normalized-mae", type=float, default=None)
    converge_parser.add_argument("--target-normalized-rmse", type=float, default=None)
    converge_parser.add_argument("--boundary-threshold", type=float, default=0.1)
    converge_parser.add_argument(
        "--holdout-evaluations",
        default=None,
        help="Optional fixed holdout evaluations parquet/CSV used for validation",
    )
    holdout_parser = subparsers.add_parser(
        "select-holdout",
        help="Select and save a fixed holdout evaluations table",
    )
    holdout_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    holdout_parser.add_argument("--evaluations", required=True, help="Evaluations parquet or CSV path")
    holdout_parser.add_argument("--output", required=True, help="Output holdout parquet or CSV path")
    holdout_parser.add_argument("--holdout-fraction", type=float, default=0.2)
    overlay_parser = subparsers.add_parser(
        "plot-gpr-overlay",
        help="Overlay holdout prediction-vs-truth scatters from convergence runs",
    )
    overlay_parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Convergence run directory; repeat in draw order",
    )
    overlay_parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Legend label; repeat once per run-dir",
    )
    overlay_parser.add_argument(
        "--color",
        action="append",
        default=None,
        help="Matplotlib color; repeat once per run-dir",
    )
    overlay_parser.add_argument(
        "--alpha",
        action="append",
        type=float,
        default=None,
        help="Point alpha; repeat once per run-dir",
    )
    overlay_parser.add_argument(
        "--zorder",
        action="append",
        type=float,
        default=None,
        help="Point zorder; repeat once per run-dir",
    )
    overlay_parser.add_argument("--output", required=True, help="Output PNG path")
    active_parser = subparsers.add_parser(
        "active-learn",
        help="Run Sobol candidate-pool active learning against a fixed holdout",
    )
    active_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    active_parser.add_argument(
        "--evaluations",
        required=True,
        help="Training evaluations parquet or CSV path",
    )
    active_parser.add_argument(
        "--holdout-evaluations",
        required=True,
        help="Fixed holdout evaluations parquet or CSV path",
    )
    active_parser.add_argument(
        "--output-dir",
        required=True,
        help="Active-learning output directory",
    )
    active_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/state.json",
    )
    active_parser.add_argument("--boundary-threshold", type=float, default=0.1)
    search_parser = subparsers.add_parser(
        "search",
        help="Screen policy candidates with the signed-RAP constraint surrogate",
    )
    search_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    search_parser.add_argument("--surrogate", required=True, help="Signed-RAP GPR bundle path")
    search_parser.add_argument("--output-dir", required=True, help="Search output directory")
    verify_parser = subparsers.add_parser(
        "verify-candidates",
        help="Run the direct evaluator on surrogate-selected candidate policies",
    )
    verify_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    verify_parser.add_argument("--candidates", required=True, help="candidate_policies.csv path")
    verify_parser.add_argument("--output-dir", required=True, help="Verification output directory")
    verify_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override run.workers; direct physics batches should normally use multiple workers",
    )
    verify_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Flush verification checkpoint batches every N completed designs",
    )
    dynamic_parser = subparsers.add_parser(
        "dynamic-search",
        help="Run structured finite-horizon dynamic-policy search with direct physics verification",
    )
    dynamic_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    dynamic_parser.add_argument("--output-dir", required=True, help="Dynamic search output directory")
    dynamic_parser.add_argument("--epochs", type=int, default=3, help="Number of policy epochs")
    dynamic_parser.add_argument(
        "--initial-samples",
        type=int,
        default=64,
        help="Initial Sobol schedules to direct-evaluate before surrogate search",
    )
    dynamic_parser.add_argument(
        "--optimizer-pool-size",
        type=int,
        default=4096,
        help="Sobol pool scored by the dynamic policy surrogate",
    )
    dynamic_parser.add_argument(
        "--verify-top",
        type=int,
        default=32,
        help="Number of optimizer-proposed schedules to direct-verify",
    )
    dynamic_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override run.workers for direct dynamic evaluations",
    )
    dynamic_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Flush dynamic evaluation checkpoints every N completed schedules",
    )
    refine_parser = subparsers.add_parser(
        "dynamic-refine",
        help="Refine a structured dynamic-policy search around direct nearest misses",
    )
    refine_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    refine_parser.add_argument(
        "--previous-evaluations",
        required=True,
        help="Previous dynamic all_evaluations parquet or CSV path",
    )
    refine_parser.add_argument("--output-dir", required=True, help="Refinement output directory")
    refine_parser.add_argument("--epochs", type=int, default=3, help="Number of policy epochs")
    refine_parser.add_argument(
        "--local-samples",
        type=int,
        default=512,
        help="Sobol perturbations around nearest-miss anchor schedules",
    )
    refine_parser.add_argument(
        "--optimizer-pool-size",
        type=int,
        default=4096,
        help="Global Sobol pool scored by the refinement surrogate",
    )
    refine_parser.add_argument(
        "--verify-top",
        type=int,
        default=32,
        help="Number of refinement candidates to direct-verify",
    )
    refine_parser.add_argument(
        "--diagnostic-sensitivity",
        default=None,
        help="Optional local_sensitivity.csv path used for diagnostic-informed moves",
    )
    refine_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override run.workers for direct refinement evaluations",
    )
    refine_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Flush refinement checkpoints every N completed schedules",
    )
    diagnose_parser = subparsers.add_parser(
        "dynamic-diagnose",
        help="Run local finite-difference control diagnostics around the best dynamic schedule",
    )
    diagnose_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    diagnose_parser.add_argument(
        "--evaluations",
        required=True,
        help="Dynamic all_evaluations.parquet path",
    )
    diagnose_parser.add_argument("--output-dir", required=True, help="Diagnostic output directory")
    diagnose_parser.add_argument("--epochs", type=int, default=3, help="Number of policy epochs")
    diagnose_parser.add_argument(
        "--perturbation-fraction",
        type=float,
        default=0.05,
        help="Fraction of each control range used for finite differences",
    )
    diagnose_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override run.workers for direct diagnostic evaluations",
    )
    diagnose_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Flush diagnostic checkpoints every N completed schedules",
    )
    envelope_parser = subparsers.add_parser(
        "plot-envelope",
        help="Generate fixed and projected feasible-envelope plots",
    )
    envelope_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    envelope_parser.add_argument("--surrogate", required=True, help="Signed-RAP GPR bundle path")
    envelope_parser.add_argument("--evaluations", required=True, help="Direct evaluations parquet or CSV path")
    envelope_parser.add_argument(
        "--verified-candidates",
        required=True,
        help="Verified candidates parquet or CSV path",
    )
    envelope_parser.add_argument("--output-dir", required=True, help="Envelope output directory")
    report_parser = subparsers.add_parser(
        "make-report",
        help="Write a Markdown viability report from explicit artifacts",
    )
    report_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    report_parser.add_argument("--evaluations", required=True, help="Direct evaluations parquet or CSV path")
    report_parser.add_argument(
        "--verified-candidates",
        required=True,
        help="Verified candidates parquet or CSV path",
    )
    report_parser.add_argument("--search-summary", required=True, help="search_summary.json path")
    report_parser.add_argument("--verification-summary", required=True, help="verification_summary.json path")
    report_parser.add_argument("--envelope-summary", required=True, help="envelope_summary.json path")
    report_parser.add_argument("--output", required=True, help="Output report.md path")
    relaxation_parser = subparsers.add_parser(
        "dynamic-relaxation-study",
        help="Summarize observed direct dynamic evaluations as requirement relaxations",
    )
    relaxation_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    relaxation_parser.add_argument(
        "--evaluations",
        action="append",
        required=True,
        help="Dynamic evaluations parquet/CSV path; repeat to combine result bundles",
    )
    relaxation_parser.add_argument("--output-dir", required=True, help="Relaxation study output directory")
    relaxation_parser.add_argument(
        "--constraints",
        default=",".join(DEFAULT_RELAXATION_CONSTRAINTS),
        help="Comma-separated constraint names to include in the relaxation study",
    )
    relaxation_parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of nearest-under-relaxation policies to write",
    )
    bound_parser = subparsers.add_parser(
        "dynamic-bound-relaxation-study",
        help="Run one-at-a-time direct input-bound relaxation sweeps around the best dynamic schedule",
    )
    bound_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    bound_parser.add_argument(
        "--evaluations",
        required=True,
        help="Dynamic evaluations parquet/CSV path used to select the base schedule",
    )
    bound_parser.add_argument("--output-dir", required=True, help="Bound-relaxation output directory")
    bound_parser.add_argument("--epochs", type=int, default=3, help="Number of policy epochs")
    bound_parser.add_argument(
        "--sweep-points",
        type=int,
        default=5,
        help="Fixed-shape sweep points per relaxed upper-bound value",
    )
    bound_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override run.workers for direct bound-relaxation evaluations",
    )
    bound_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Flush bound-relaxation checkpoints every N completed schedules",
    )
    ipug_parser = subparsers.add_parser(
        "dynamic-ipug-diagnostic",
        help="Run direct all-epoch IPUG counterfactuals around the best dynamic schedule",
    )
    ipug_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    ipug_parser.add_argument(
        "--evaluations",
        required=True,
        help="Dynamic evaluations parquet/CSV path used to select the base schedule",
    )
    ipug_parser.add_argument("--output-dir", required=True, help="IPUG diagnostic output directory")
    ipug_parser.add_argument("--epochs", type=int, default=3, help="Number of policy epochs")
    ipug_parser.add_argument(
        "--ipug-values",
        default="0,2,5,8,10,15,20",
        help="Comma-separated all-epoch IPUG quota values to direct-evaluate",
    )
    ipug_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override run.workers for direct IPUG evaluations",
    )
    ipug_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Flush IPUG diagnostic checkpoints every N completed schedules",
    )
    trajectory_parser = subparsers.add_parser(
        "dynamic-trajectory-artifacts",
        help="Rerun selected dynamic schedules and write trajectory/trade-space figures",
    )
    trajectory_parser.add_argument("--config", required=True, help="Path to viability YAML config")
    trajectory_parser.add_argument("--output-dir", required=True, help="Trajectory artifact output directory")
    trajectory_parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        help="Dynamic evaluations parquet/CSV path; repeat for multiple result bundles",
    )
    trajectory_parser.add_argument(
        "--epoch-count",
        action="append",
        type=int,
        required=True,
        help="Epoch count corresponding to each --evaluation; repeat in the same order",
    )
    trajectory_parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label corresponding to each --evaluation; repeat in the same order",
    )
    args = parser.parse_args()

    if args.command == "plot-gpr-overlay":
        result = write_gpr_prediction_overlay_plot(
            args.run_dir,
            args.output,
            labels=args.label,
            colors=args.color,
            alphas=args.alpha,
            zorders=args.zorder,
        )
        print(json.dumps({"plot_path": str(result.plot_path), "point_counts": result.point_counts}))
        return 0

    config = load_config(args.config)
    if args.command == "evaluate-design":
        design_values = json.loads(args.design_json)
        design = PolicyDesign.from_mapping(design_values, config.policy)
        result = evaluate_design(design, config)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.status == "ok" else 1

    if args.command == "select-holdout":
        result = write_holdout_selection_from_file(
            args.evaluations,
            config,
            args.output,
            holdout_fraction=args.holdout_fraction,
        )
        print(
            json.dumps(
                {
                    "holdout_path": str(result.holdout_path),
                    "n_rows_total": result.n_rows_total,
                    "holdout_size": result.holdout_size,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "make-report":
        result = write_viability_report_from_files(
            config=config,
            evaluations_path=args.evaluations,
            verified_candidates_path=args.verified_candidates,
            search_summary_path=args.search_summary,
            verification_summary_path=args.verification_summary,
            envelope_summary_path=args.envelope_summary,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "report_path": str(result.report_path),
                    "verified_count": result.verified_count,
                    "feasible_count": result.feasible_count,
                    "best_candidate_id": result.best_candidate_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

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
        worker_count = config.run.workers if args.workers is None else args.workers
        results = evaluate_designs_parallel(
            designs,
            config,
            workers=args.workers,
            checkpoint_dir=output_dir / "checkpoints",
            checkpoint_every=args.checkpoint_every,
        )
        write_table(results, output_dir / "evaluations.parquet")
        all_ok = bool(len(results) > 0 and (results["status"] == "ok").all())
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "design_count": int(len(designs)),
                    "evaluated_count": int(len(results)),
                    "ok_count": int((results["status"] == "ok").sum()),
                    "workers": worker_count,
                    "phase_backend": config.model.phase_backend,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if all_ok else 1

    if args.command == "fit-surrogate":
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.evaluations).parent
        result = fit_surrogates_from_file(
            args.evaluations,
            config,
            output_dir,
            boundary_threshold=args.boundary_threshold,
            fit_gpr=not args.no_gpr,
        )
        print(json.dumps(result.metrics, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "converge-surrogate":
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.evaluations).parent
        write_config_resolved(config, output_dir)
        train_sizes = _parse_train_sizes(args.train_sizes)
        result = run_gpr_convergence_from_file(
            args.evaluations,
            config,
            output_dir,
            train_sizes=train_sizes,
            holdout_fraction=args.holdout_fraction,
            target_r2=args.target_r2,
            target_normalized_mae=args.target_normalized_mae,
            target_normalized_rmse=args.target_normalized_rmse,
            boundary_threshold=args.boundary_threshold,
            holdout_path=args.holdout_evaluations,
        )
        print(result.metrics_table.to_string(index=False))
        print(json.dumps({"converged": result.converged}, sort_keys=True))
        return 0

    if args.command == "active-learn":
        result = run_active_learning_from_files(
            evaluations_path=args.evaluations,
            holdout_path=args.holdout_evaluations,
            config=config,
            output_dir=args.output_dir,
            resume=args.resume,
            boundary_threshold=args.boundary_threshold,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "metrics_path": str(result.metrics_path),
                    "state_path": str(result.state_path),
                    "latest_training_path": str(result.latest_training_path),
                    "latest_model_path": str(result.latest_model_path),
                    "plot_paths": {name: str(path) for name, path in result.plot_paths.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "search":
        result = run_surrogate_search_from_files(
            surrogate_path=args.surrogate,
            config=config,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "candidates_path": str(result.candidates_path),
                    "scored_path": str(result.scored_path),
                    "summary_path": str(result.summary_path),
                    "candidate_count": result.candidate_count,
                    "scored_count": result.scored_count,
                    "plot_paths": {name: str(path) for name, path in result.plot_paths.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "verify-candidates":
        result = verify_candidates_from_file(
            candidates_path=args.candidates,
            config=config,
            output_dir=args.output_dir,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "verified_path": str(result.verified_path),
                    "summary_path": str(result.summary_path),
                    "verified_count": result.verified_count,
                    "feasible_count": result.feasible_count,
                    "plot_paths": {name: str(path) for name, path in result.plot_paths.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "dynamic-search":
        result = run_dynamic_policy_search(
            config=config,
            output_dir=args.output_dir,
            epoch_count=args.epochs,
            initial_samples=args.initial_samples,
            optimizer_pool_size=args.optimizer_pool_size,
            verify_top=args.verify_top,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "initial_schedules_path": str(result.initial_schedules_path),
                    "initial_evaluations_path": str(result.initial_evaluations_path),
                    "optimizer_candidates_path": str(result.optimizer_candidates_path),
                    "optimizer_evaluations_path": str(result.optimizer_evaluations_path),
                    "all_evaluations_path": str(result.all_evaluations_path),
                    "summary_path": str(result.summary_path),
                    "evaluated_count": result.evaluated_count,
                    "feasible_count": result.feasible_count,
                    "best_phi": result.best_phi,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "dynamic-refine":
        result = run_dynamic_policy_refinement(
            config=config,
            previous_evaluations_path=args.previous_evaluations,
            output_dir=args.output_dir,
            diagnostic_sensitivity_path=args.diagnostic_sensitivity,
            epoch_count=args.epochs,
            local_samples=args.local_samples,
            optimizer_pool_size=args.optimizer_pool_size,
            verify_top=args.verify_top,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "refinement_candidates_path": str(result.refinement_candidates_path),
                    "refinement_evaluations_path": str(result.refinement_evaluations_path),
                    "all_evaluations_path": str(result.all_evaluations_path),
                    "summary_path": str(result.summary_path),
                    "candidate_count": result.candidate_count,
                    "evaluated_count": result.evaluated_count,
                    "feasible_count": result.feasible_count,
                    "best_phi": result.best_phi,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "dynamic-diagnose":
        result = run_dynamic_policy_diagnostic(
            config=config,
            evaluations_path=args.evaluations,
            output_dir=args.output_dir,
            epoch_count=args.epochs,
            perturbation_fraction=args.perturbation_fraction,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "diagnostic_schedules_path": str(result.diagnostic_schedules_path),
                    "diagnostic_evaluations_path": str(result.diagnostic_evaluations_path),
                    "sensitivity_path": str(result.sensitivity_path),
                    "report_path": str(result.report_path),
                    "evaluated_count": result.evaluated_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "dynamic-relaxation-study":
        result = run_dynamic_relaxation_study(
            config=config,
            evaluation_paths=args.evaluations,
            output_dir=args.output_dir,
            constraints=_parse_constraints(args.constraints),
            top_n=args.top_n,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "nearest_path": str(result.nearest_path),
                    "pareto_path": str(result.pareto_path),
                    "relaxation_sets_path": str(result.relaxation_sets_path),
                    "summary_path": str(result.summary_path),
                    "report_path": str(result.report_path),
                    "evaluated_count": result.evaluated_count,
                    "feasible_count": result.feasible_count,
                    "best_phi": result.best_phi,
                    "best_linf_relaxation": result.best_linf_relaxation,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "dynamic-bound-relaxation-study":
        result = run_dynamic_bound_relaxation_study(
            config=config,
            evaluations_path=args.evaluations,
            output_dir=args.output_dir,
            epoch_count=args.epochs,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
            sweep_points=args.sweep_points,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "candidates_path": str(result.candidates_path),
                    "evaluations_path": str(result.evaluations_path),
                    "best_by_experiment_path": str(result.best_by_experiment_path),
                    "summary_path": str(result.summary_path),
                    "report_path": str(result.report_path),
                    "evaluated_count": result.evaluated_count,
                    "feasible_count": result.feasible_count,
                    "best_phi": result.best_phi,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "dynamic-ipug-diagnostic":
        result = run_dynamic_ipug_diagnostic(
            config=config,
            evaluations_path=args.evaluations,
            output_dir=args.output_dir,
            epoch_count=args.epochs,
            ipug_values=_parse_float_list(args.ipug_values),
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "candidates_path": str(result.candidates_path),
                    "evaluations_path": str(result.evaluations_path),
                    "summary_path": str(result.summary_path),
                    "report_path": str(result.report_path),
                    "evaluated_count": result.evaluated_count,
                    "feasible_count": result.feasible_count,
                    "best_phi": result.best_phi,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "dynamic-trajectory-artifacts":
        specs = _parse_evaluation_specs(args.evaluation, args.epoch_count, args.label)
        result = run_dynamic_trajectory_artifacts(
            config=config,
            evaluation_specs=specs,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "summary_path": str(result.summary_path),
                    "selected_policies_path": str(result.selected_policies_path),
                    "trajectory_paths": {name: str(path) for name, path in result.trajectory_paths.items()},
                    "figure_paths": {name: str(path) for name, path in result.figure_paths.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "plot-envelope":
        result = run_envelope_plots_from_files(
            surrogate_path=args.surrogate,
            evaluations_path=args.evaluations,
            verified_candidates_path=args.verified_candidates,
            config=config,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "summary_path": str(result.summary_path),
                    "anchor_design_id": result.anchor_design_id,
                    "anchor_phi": result.anchor_phi,
                    "plots_skipped": result.plots_skipped,
                    "plots_skipped_reason": result.plots_skipped_reason,
                    "plot_paths": {name: str(path) for name, path in result.plot_paths.items()},
                    "grid_paths": {name: str(path) for name, path in result.grid_paths.items()},
                    "de_comparison_paths": {
                        name: str(path)
                        for name, path in result.de_comparison_paths.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    raise ValueError(f"Unhandled command {args.command!r}")


def _parse_train_sizes(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_constraints(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return list(DEFAULT_RELAXATION_CONSTRAINTS)
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one numeric value")
    return parsed


def _parse_evaluation_specs(
    evaluations: list[str],
    epoch_counts: list[int],
    labels: list[str] | None,
) -> list[tuple[str, int, str]]:
    if len(evaluations) != len(epoch_counts):
        raise ValueError("--evaluation and --epoch-count must be supplied the same number of times")
    if labels is None:
        labels = []
    if labels and len(labels) != len(evaluations):
        raise ValueError("--label must be omitted or supplied once per --evaluation")
    specs = []
    for index, (evaluation, epoch_count) in enumerate(zip(evaluations, epoch_counts, strict=True)):
        label = labels[index] if labels else f"run_{index + 1}_{epoch_count}epoch"
        specs.append((evaluation, int(epoch_count), label))
    return specs


if __name__ == "__main__":
    raise SystemExit(main())
