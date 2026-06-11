"""
Automated checks for syllabus upgrade allocation and continuation training (CT).

Run: python -m unittest tests.test_upgrade_logic -v
"""
import contextlib
import io
import unittest

from src.engine import (
    create_pilots,
    phase_upgrade_metrics,
    process_syllabus_event,
    run_phase_simulation,
    select_upgrade_students,
    total_phase_capacity,
)
from src.models import Qual, SquadronConfig, Upgrade
from src.simulation_config import SimulationConfig
from src.syllabi import (
    TEST_MQT_SYLLABUS,
    TEST_FLUG_SYLLABUS,
    TEST_IPUG_SYLLABUS,
    TEST_CONTINUATION_PROFILE,
    incomplete_burden,
    syllabus_burden_per_student,
    syllabus_line_sortie_burden,
)


def _make_roster(
    wg: int = 5,
    fl: int = 2,
    ip: int = 2,
    mqt: int = 0,
    *,
    ute: float = 3.0,
    paa: int = 9,
    sq_id: int = 1,
):
    total = wg + fl + ip
    exp = fl + ip
    cfg = SquadronConfig(
        ute=ute,
        paa=paa,
        id=sq_id,
        total_pilots=total,
        ip_qty=ip,
        experience_ratio=exp / total,
    )
    pilots = create_pilots(cfg)
    wgs = [p for p in pilots if p.qual == Qual.WG]
    for i in range(mqt):
        wgs[i].upgrade = Upgrade.MQT
    return cfg, pilots


def _run_phase(cfg, pilots, *, phase_days: int = 30, pre_seed: bool = True):
    sim = SimulationConfig(phase_length_days=phase_days, allocation_noise=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_phase_simulation(
            cfg,
            pilots,
            allocation_noise=0.0,
            sim_config=sim,
            pre_seed_upgrades=pre_seed,
            mqt_syllabus=TEST_MQT_SYLLABUS,
            flug_syllabus=TEST_FLUG_SYLLABUS,
            ipug_syllabus=TEST_IPUG_SYLLABUS,
            continuation_profile=TEST_CONTINUATION_PROFILE,
        )


def _mqt_students(pilots):
    return [p for p in pilots if p.upgrade == Upgrade.MQT]


def _total_sorties(pilots):
    return int(round(sum(p.sortie_phase for p in pilots)))


class TestSyllabusOrdering(unittest.TestCase):
    def test_event_first_before_next_event(self):
        """Each syllabus row runs for all students before the next row starts."""
        cfg, pilots = _make_roster(mqt=2, ute=100.0, paa=100)
        students = _mqt_students(pilots)
        cap = 999_999

        with contextlib.redirect_stdout(io.StringIO()):
            process_syllabus_event(
                TEST_MQT_SYLLABUS[0], students, pilots, Upgrade.MQT,
                0.0, cfg, 30, cap,
            )
        self.assertEqual([p.sortie_phase for p in students], [1.0, 1.0])

        with contextlib.redirect_stdout(io.StringIO()):
            process_syllabus_event(
                TEST_MQT_SYLLABUS[1], students, pilots, Upgrade.MQT,
                0.0, cfg, 30, cap,
            )
        self.assertEqual([p.sortie_phase for p in students], [2.0, 2.0])


class TestCapacityDeferral(unittest.TestCase):
    def test_two_mqt_defers_one_dca_line_at_27_capacity(self):
        """2 MQT @ ute×paa=27: OBFM+ACM complete; one DCA line (8 slots) deferred."""
        cfg, pilots = _make_roster(mqt=2, ute=3.0, paa=9)
        _run_phase(cfg, pilots)

        self.assertEqual(_total_sorties(pilots), 27)
        self.assertEqual(cfg.deferred_sortie_burden, 8)
        self.assertAlmostEqual(cfg.mqt_sortie_carry, 8 / 14)

        incomplete = [
            p.incomplete_syllabus_items
            for p in _mqt_students(pilots)
            if p.incomplete_syllabus_items
        ]
        self.assertEqual(len(incomplete), 1)
        self.assertEqual(incomplete[0][0].name, "DCA")
        self.assertEqual(incomplete_burden(incomplete[0])[0], 8)

    def test_metrics_match_squadron_deferred_fields(self):
        cfg, pilots = _make_roster(mqt=2, ute=3.0, paa=9)
        _run_phase(cfg, pilots)
        metrics = phase_upgrade_metrics(
            pilots,
            mqt_syllabus=TEST_MQT_SYLLABUS,
            flug_syllabus=TEST_MQT_SYLLABUS,
            ipug_syllabus=TEST_MQT_SYLLABUS,
        )

        self.assertEqual(metrics["deferred_mqt_sorties"], cfg.deferred_sortie_burden)
        self.assertEqual(metrics["held_back_mqt"], 1)
        self.assertAlmostEqual(
            metrics["remaining_mqt_syllabi_sorties_only"],
            cfg.mqt_sortie_carry,
        )

    def test_one_mqt_defers_dca_at_12_capacity(self):
        cfg, pilots = _make_roster(mqt=1, ute=2.0, paa=6)
        _run_phase(cfg, pilots)

        self.assertEqual(_total_sorties(pilots), 12)
        self.assertEqual(cfg.deferred_sortie_burden, 8)
        student = _mqt_students(pilots)[0]
        self.assertEqual([e.name for e in student.incomplete_syllabus_items], ["DCA"])


class TestGraduation(unittest.TestCase):
    def test_completed_mqt_graduates_to_wg(self):
        cfg, pilots = _make_roster(mqt=1, ute=3.0, paa=9)
        _run_phase(cfg, pilots)

        self.assertEqual(len(_mqt_students(pilots)), 0)
        self.assertEqual(cfg.deferred_sortie_burden, 0)
        grads = [p for p in pilots if p.upgrade == Upgrade.NONE and p.qual == Qual.WG]
        self.assertEqual(len(grads), 5)


class TestCapacityAccounting(unittest.TestCase):
    def test_sorties_never_exceed_phase_capacity(self):
        cfg, pilots = _make_roster(mqt=2, ute=3.0, paa=9)
        sim = SimulationConfig(phase_length_days=30, allocation_noise=0.0)
        gross = int(total_phase_capacity(cfg) * sim.phase_length_months)
        _run_phase(cfg, pilots)
        self.assertLessEqual(_total_sorties(pilots), gross)

    def test_deferred_burden_reduces_next_phase_budget(self):
        cfg, pilots = _make_roster(mqt=1, ute=2.0, paa=6)
        _run_phase(cfg, pilots, phase_days=30)
        self.assertEqual(cfg.deferred_sortie_burden, 8)

        cfg.ute = 3.0
        cfg.paa = 9
        gross = int(total_phase_capacity(cfg) * 1.0)
        effective = gross - cfg.deferred_sortie_burden
        self.assertEqual(effective, 19)


class TestCarryoverRetry(unittest.TestCase):
    def test_phase_two_completes_deferred_dca_without_syllabus_rerun(self):
        """Carryover retries DCA in step 3; step 4 does not re-fly OBFM/ACM."""
        cfg, pilots = _make_roster(mqt=1, ute=2.0, paa=6)
        _run_phase(cfg, pilots, phase_days=30)
        self.assertEqual(cfg.deferred_sortie_burden, 8)

        cfg.ute = 3.0
        cfg.paa = 9
        _run_phase(cfg, pilots, phase_days=30, pre_seed=True)

        self.assertEqual(len(_mqt_students(pilots)), 0)
        self.assertEqual(cfg.deferred_sortie_burden, 0)
        self.assertEqual(cfg.mqt_sortie_carry, 0.0)

    def test_carryover_student_sortie_count_is_dca_only_in_phase_two(self):
        """Phase 2 resets counters; carryover flies one student sortie (DCA), not three."""
        cfg, pilots = _make_roster(mqt=1, ute=2.0, paa=6)
        _run_phase(cfg, pilots, phase_days=30)
        student = _mqt_students(pilots)[0]
        self.assertEqual(student.sortie_phase, 2.0)  # OBFM + ACM in phase 1

        cfg.ute = 3.0
        cfg.paa = 9
        _run_phase(cfg, pilots, phase_days=30, pre_seed=True)

        self.assertEqual(student.sortie_phase, 1.0)  # DCA only in phase 2 (not 3 from re-run)
        self.assertEqual(student.upgrade, Upgrade.NONE)


class TestSelectUpgradeStudents(unittest.TestCase):
    def test_carryover_mqt_not_reselected(self):
        cfg, pilots = _make_roster(mqt=1, wg=4, fl=1, ip=1)
        carryover = _mqt_students(pilots)[0]
        cfg.mqt_students = 1  # one new MQT slot; carryover already counted separately

        selected = select_upgrade_students(pilots, Upgrade.MQT, cfg.mqt_students)

        self.assertEqual(len(_mqt_students(pilots)), 2)
        self.assertTrue(all(p is not carryover for p in selected))
        self.assertEqual(len(selected), 1)


class TestContinuationTraining(unittest.TestCase):
    def test_mqt_students_do_not_receive_ct_sorties(self):
        """CT pool excludes MQT; MQT sorties come from syllabus only (not CT top-up)."""
        cfg, pilots = _make_roster(mqt=2, ute=3.0, paa=9, wg=5, fl=2, ip=2)
        _run_phase(cfg, pilots, phase_days=30)

        sortie_student_lines = sum(
            e.num_student
            for e in TEST_MQT_SYLLABUS
            if e.event_type.name == "SORTIE"
        )
        for student in _mqt_students(pilots):
            # Deferred students may have fewer than a full syllabus; never more than sortie rows.
            self.assertLessEqual(student.sortie_phase, float(sortie_student_lines))
            self.assertGreater(student.sortie_phase, 0.0)

    def test_wg_only_roster_tags_ct_as_single_ship(self):
        """With no FL/IP, FL CT buckets cannot staff; WG CT flies single-ship."""
        cfg = SquadronConfig(
            ute=10.0, paa=5, id=1, total_pilots=8, ip_qty=0, experience_ratio=0.0,
        )
        pilots = create_pilots(cfg)
        _run_phase(cfg, pilots, phase_days=30, pre_seed=True)

        ct_sorties = sum(p.sortie_phase for p in pilots)
        single_ship = sum(p.sortie_single_ship for p in pilots)
        self.assertGreater(ct_sorties, 0)
        self.assertEqual(single_ship, ct_sorties)

    def test_ct_uses_leftover_capacity_after_syllabus(self):
        cfg, pilots = _make_roster(mqt=2, ute=3.0, paa=9)
        sim = SimulationConfig(phase_length_days=30, allocation_noise=0.0)
        gross = int(total_phase_capacity(cfg) * sim.phase_length_months)
        _run_phase(cfg, pilots)

        self.assertEqual(_total_sorties(pilots), gross)


class TestSyllabusBurdenMath(unittest.TestCase):
    def test_test_mqt_sortie_burden_per_student(self):
        sortie_b, sim_b = syllabus_burden_per_student(TEST_MQT_SYLLABUS)
        self.assertEqual(sortie_b, 14)
        self.assertEqual(sim_b, 4)

    def test_dca_line_is_eight_sortie_slots(self):
        dca = next(e for e in TEST_MQT_SYLLABUS if e.name == "DCA" and e.event_type.name == "SORTIE")
        self.assertEqual(syllabus_line_sortie_burden(dca), 8)


if __name__ == "__main__":
    unittest.main()
