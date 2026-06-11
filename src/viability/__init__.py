"""Prototype viability analysis layer for the absorption model."""

from src.viability.config import STANDARD_POLICY_VARIABLES, ViabilityConfig, load_config
from src.viability.design_space import DesignSpace
from src.viability.doe import generate_doe
from src.viability.evaluator import EvaluationResult, evaluate_design, evaluate_designs_parallel
from src.viability.io import run_output_dir, write_config_resolved, write_table
from src.viability.policy import PolicyDesign

__all__ = [
    "STANDARD_POLICY_VARIABLES",
    "DesignSpace",
    "EvaluationResult",
    "PolicyDesign",
    "ViabilityConfig",
    "evaluate_design",
    "evaluate_designs_parallel",
    "generate_doe",
    "load_config",
    "run_output_dir",
    "write_config_resolved",
    "write_table",
]
