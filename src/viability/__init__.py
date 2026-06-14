"""Prototype viability analysis layer for the absorption model."""

from src.viability.active_learning import (
    ActiveLearningResult,
    run_active_learning,
    run_active_learning_from_files,
)
from src.viability.config import STANDARD_POLICY_VARIABLES, ViabilityConfig, load_config
from src.viability.design_space import DesignSpace
from src.viability.doe import generate_doe
from src.viability.evaluator import EvaluationResult, evaluate_design, evaluate_designs_parallel
from src.viability.io import run_output_dir, write_config_resolved, write_table
from src.viability.policy import PolicyDesign
from src.viability.surrogate import (
    GPRPredictionOverlayResult,
    HoldoutSelectionResult,
    SurrogateConvergenceResult,
    SurrogateFitResult,
    fit_surrogates,
    fit_surrogates_from_file,
    run_gpr_convergence,
    run_gpr_convergence_from_file,
    write_holdout_selection_from_file,
    write_gpr_prediction_overlay_plot,
)

__all__ = [
    "STANDARD_POLICY_VARIABLES",
    "ActiveLearningResult",
    "DesignSpace",
    "EvaluationResult",
    "GPRPredictionOverlayResult",
    "HoldoutSelectionResult",
    "PolicyDesign",
    "SurrogateConvergenceResult",
    "SurrogateFitResult",
    "ViabilityConfig",
    "evaluate_design",
    "evaluate_designs_parallel",
    "fit_surrogates",
    "fit_surrogates_from_file",
    "generate_doe",
    "load_config",
    "run_gpr_convergence",
    "run_gpr_convergence_from_file",
    "run_active_learning",
    "run_active_learning_from_files",
    "run_output_dir",
    "write_holdout_selection_from_file",
    "write_gpr_prediction_overlay_plot",
    "write_config_resolved",
    "write_table",
]
