from dataclasses import dataclass

from src.models import PHASE_DAYS_PER_NOTIONAL_MONTH

DEFAULT_PHASE_LENGTH_DAYS = 120
DEBUG_PHASE_LENGTH_DAYS = 30


@dataclass(frozen=True)
class SimulationConfig:
    """Fleet-wide simulation settings; shared by all squadrons in a run."""

    # phase_length_days: int = DEFAULT_PHASE_LENGTH_DAYS
    phase_length_days_debug: int = DEBUG_PHASE_LENGTH_DAYS
    allocation_noise: float = 0.0

    @property
    def phase_length_months(self) -> float:
        return max(0.0, float(self.phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH)
