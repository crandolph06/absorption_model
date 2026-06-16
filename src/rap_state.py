from src.models import (
    Qual,
    Upgrade,
    monthly_sortie_rap_target,
    monthly_sim_rap_target,
)

_COHORT_QUAL = {"WG": Qual.WG, "FL": Qual.FL, "IP": Qual.IP}
_SHORTFALL_BIT = {"WG": 1, "FL": 2, "IP": 4}


def _rap_cohort_groups(pilots):
    """WG / FL / IP cohorts for RAP only. MQT excluded (no sortie or sim RAP requirement)."""
    return {
        "WG": [p for p in pilots if p.qual == Qual.WG and p.upgrade != Upgrade.MQT],
        "FL": [p for p in pilots if p.qual == Qual.FL],
        "IP": [p for p in pilots if p.qual == Qual.IP]
    }


def rap_assess(pilots):
    """
    Sortie-based RAP shortfall for WG, FL, and IP only (same cohorts as ``sim_rap_metrics``).

    For observed MQT flying only (not RAP), use ``mqt_observed_sortie_metrics``.
    """
    groups = _rap_cohort_groups(pilots)

    rap_dict = {}
    blue_rap_dict = {}
    red_dict = {}

    for group_name, group_pilots in groups.items():
        # No pilots in this cohort → no shortfall (don't treat avg=0 as below RAP).
        if not group_pilots:
            rap_dict[group_name] = [0, 0]
            blue_rap_dict[group_name] = [0, 0]
            red_dict[group_name] = [0, 0]
            continue

        avg_sorties = sum(p.sortie_rap_monthly for p in group_pilots) / len(group_pilots)
        avg_blue_sorties = sum(p.sortie_blue_monthly for p in group_pilots) / len(group_pilots)
        avg_red_sorties = sum(p.sortie_red_monthly for p in group_pilots) / len(group_pilots)

        qual = _COHORT_QUAL[group_name]
        rap_req = monthly_sortie_rap_target(qual)
        bit_mask = _SHORTFALL_BIT[group_name]

        rap_dict[group_name] = [bit_mask if avg_sorties < rap_req else 0, avg_sorties]
        blue_rap_dict[group_name] = [bit_mask if avg_blue_sorties < rap_req else 0, avg_blue_sorties]
        red_dict[group_name] = [avg_red_sorties / avg_sorties if avg_sorties > 0 else 0, avg_red_sorties]

    return rap_dict, blue_rap_dict, red_dict


def mqt_observed_sortie_metrics(pilots):
    """
    Observed sortie rates for MQT students only (not RAP; MQT has no sortie RAP requirement).
    """
    mqts = [p for p in pilots if p.upgrade == Upgrade.MQT]
    if not mqts:
        return {"sortie_mo": 0.0, "sortie_blue_mo": 0.0, "sortie_red_mo": 0.0}
    n = len(mqts)
    return {
        "sortie_mo": sum(p.sortie_monthly for p in mqts) / n,
        "sortie_blue_mo": sum(p.sortie_blue_monthly for p in mqts) / n,
        "sortie_red_mo": sum(p.sortie_red_monthly for p in mqts) / n,
    }


def mqt_observed_sim_metrics(pilots):
    """
    Observed sim rates for MQT students only (not RAP; MQT has no sim RAP requirement).
    """
    mqts = [p for p in pilots if p.upgrade == Upgrade.MQT]
    if not mqts:
        return {"sim_mo": 0.0, "sim_rap_shortfall": 0.0}
    n = len(mqts)
    return {
        "sim_mo": sum(p.sim_monthly for p in mqts) / n,
        "sim_rap_shortfall": sum(p.sim_rap_shortfall for p in mqts) / n,
    }


def rap_state_code(rap_dict):
    """Bitmask from ``rap_assess`` shortfall flags (WG=1, FL=2, IP=4)."""
    rap_code = 0
    for k, v in rap_dict.items():
        if k in ("WG", "FL", "IP"):
            rap_code += v[0]
    return rap_code


def rap_state_label(code: int) -> str:
    labels = {
        0: "All Make RAP",
        1: "WG Shortfall",
        2: "FL Shortfall",
        3: "WG + FL Shortfall",
        4: "IP Shortfall",
        5: "WG + IP Shortfall",
        6: "FL + IP Shortfall",
        7: "WG + FL + IP Shortfall",
    }
    return labels.get(code, f"RAP code {code}")


def sim_rap_metrics(pilots):
    """
    Per-cohort sim utilization for WG, FL, and IP only (same cohorts as ``rap_assess``).

    Uses ``Pilot.sim_monthly`` and ``Pilot.sim_rap_shortfall`` vs ``monthly_sim_rap_target``.
    MQT is excluded from RAP; for observed MQT sim only, use ``mqt_observed_sim_metrics``.
    """
    groups = _rap_cohort_groups(pilots)
    out = {}
    for name, group_pilots in groups.items():
        qual = _COHORT_QUAL[name]
        sim_req = monthly_sim_rap_target(qual)
        if not group_pilots:
            out[name] = {
                "sim_mo": sim_req,
                "sim_rap_shortfall": 0.0,
                "sim_req_mo": sim_req,
            }
            continue
        n = len(group_pilots)
        out[name] = {
            "sim_mo": sum(p.sim_monthly for p in group_pilots) / n,
            "sim_rap_shortfall": sum(p.sim_rap_shortfall for p in group_pilots) / n,
            "sim_req_mo": sim_req,
        }
    return out


def sim_rap_state_code(sim_metrics: dict) -> int:
    """Same 3-bit mask as ``rap_state_code`` (WG=1, FL=2, IP=4); cohorts exclude MQT."""
    code = 0
    for key, bit in (("WG", 1), ("FL", 2), ("IP", 4)):
        if sim_metrics[key]["sim_mo"] < monthly_sim_rap_target(_COHORT_QUAL[key]):
            code |= bit
    return code


def sim_rap_state_label(code: int) -> str:
    """Same wording as ``rap_state_label`` (sortie and sim RAP share labels)."""
    return rap_state_label(code)
