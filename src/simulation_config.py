from dataclasses import dataclass
from typing import Optional

from src.models import PHASE_DAYS_PER_NOTIONAL_MONTH

DEFAULT_PHASE_LENGTH_DAYS = 120
DEBUG_PHASE_LENGTH_DAYS = 30


@dataclass(frozen=True)
class SimulationConfig:
    """Fleet-wide simulation settings; shared by all squadrons in a run."""

    phase_length_days: int = DEFAULT_PHASE_LENGTH_DAYS
    allocation_noise: float = 0.0
    # Max single-ship CT sorties allocated per pilot per notional month (extra iron stays unallocated).
    single_ship_monthly_cap: float = 1.0
    # When set (0–1), sortie iron is split: syllabus vs CT each get a reserved share.
    upgrade_sortie_fraction: Optional[float] = None
    # When set, pilots are assigned to UTCs and RAP is prioritized by UTC; when false, allocator attempts to prioritize equity
    utc_wise_allocation: bool = False

    @property
    def phase_length_months(self) -> float:
        return max(0.0, float(self.phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH)

    def phase_sortie_budgets(self, total_iron: int) -> tuple[int, Optional[int]]:
        """
        Return (upgrade_sortie_cap, ct_sortie_cap).

        ``ct_sortie_cap`` is None when fraction is unset (legacy: CT uses leftover iron).
        """
        if self.upgrade_sortie_fraction is None:
            return total_iron, None
        frac = max(0.0, min(1.0, float(self.upgrade_sortie_fraction)))
        upgrade_cap = int(total_iron * frac)
        return upgrade_cap, total_iron - upgrade_cap
