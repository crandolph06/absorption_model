from dataclasses import dataclass
from enum import Enum
from typing import List
from src.models import EventType, Qual

# ----------------------
# Syllabus Bucket
# ----------------------
@dataclass(frozen=True)
class SyllabusEvent: 
    name: str
    event_type: EventType
    seat_type: Qual
    num_student: int = 1
    num_instructor: int = 1
    num_blue_wg: int = 0
    num_blue_fl: int = 0
    num_red_wg: int = 0
    num_red_fl: int = 0

    def total_slots(self):
        return self.num_student + self.num_instructor + self.num_blue_wg + self.num_blue_fl + self.num_red_wg + self.num_red_fl

@dataclass
class UpgradeProgram:
    name: str # "MQT", "FLUG", "IPUG"
    syllabus: List[SyllabusEvent]
    student_qual: Qual
    num_students: int

# ----------------------
# Tiny Test Syllabi
# ----------------------
TEST_MQT_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.WG, num_student=1, num_instructor=1),
    SyllabusEvent("DBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0)
    # SyllabusEvent("ACM", EventType.SORTIE, Qual.WG, num_blue_wg=1,)
]

TEST_FLUG_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.WG, num_student=1, num_instructor=1),
    # SyllabusEvent("ACM", EventType.SORTIE, Qual.WG, num_blue_wg=1)
]

TEST_IPUG_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.FL, num_student=1, num_instructor=1),
    SyllabusEvent("ACM", EventType.SORTIE, Qual.FL, 1,1,0,0,1,1)
]

# ----------------------
# MQT Syllabus
# ----------------------
MQT_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("DBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("HABFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("2vX TI", EventType.SIM, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("ACM", EventType.SORTIE, Qual.WG, 1,1,0,0,1,1),
    SyllabusEvent("4vX TI", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SORTIE, Qual.WG, 1,1,1,1,2,2),
    SyllabusEvent("U-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("D-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("U-SEAD", EventType.SORTIE, Qual.WG, 1,1,2,2,0,0),
    SyllabusEvent("O-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("O-SEAD", EventType.SORTIE, Qual.WG, 1,1,2,2,2,2),
    SyllabusEvent("O-AI", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("CAS", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("BSA", EventType.SORTIE, Qual.WG, 1,1,1,1,0,0),
]

# ----------------------
# FLUG Syllabus (same as MQT)
# ----------------------
FLUG_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("DBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("HABFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("2vX TI", EventType.SIM, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("ACM", EventType.SORTIE, Qual.WG, 1,1,0,0,1,1),
    SyllabusEvent("4vX TI", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SORTIE, Qual.WG, 1,1,1,1,2,2),
    SyllabusEvent("U-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("D-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("U-SEAD", EventType.SORTIE, Qual.WG, 1,1,2,2,0,0),
    SyllabusEvent("O-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("O-SEAD", EventType.SORTIE, Qual.WG, 1,1,2,2,2,2),
    SyllabusEvent("O-AI", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("CAS", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("BSA", EventType.SORTIE, Qual.WG, 1,1,1,1,0,0),
]

# ----------------------
# IPUG Syllabus
# ----------------------
IPUG_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("DBFM", EventType.SORTIE, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("HABFM", EventType.SORTIE, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("2vX TI", EventType.SIM, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("ACM", EventType.SORTIE, Qual.FL, 1,1,0,0,1,1),
    SyllabusEvent("4vX TI", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SORTIE, Qual.FL, 1,1,1,1,2,2),
    SyllabusEvent("U-SEAD", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("D-SEAD", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("U-SEAD", EventType.SORTIE, Qual.FL, 1,1,2,2,0,0),
    SyllabusEvent("O-SEAD", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("O-SEAD", EventType.SORTIE, Qual.FL, 1,1,2,2,2,2),
    SyllabusEvent("O-AI", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("CAS", EventType.SORTIE, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("BSA", EventType.SORTIE, Qual.FL, 1,1,1,1,0,0),
]

# ----------------------
# Continuation Profile
# ----------------------
@dataclass(frozen=True)
class ContinuationBucket:
    name: str
    min_qual: Qual
    side: str
    fraction: float

@dataclass
class ContinuationProfile:
    name: str
    buckets: List[ContinuationBucket]

CONTINUATION_PROFILE = ContinuationProfile(
    name="Continuation Training",
    buckets=[
        ContinuationBucket("Blue FL", Qual.FL, "Blue", 0.425),
        ContinuationBucket("Blue WG", Qual.WG, "Blue", 0.425),
        ContinuationBucket("Red FL", Qual.FL, "Red", 0.075),
        ContinuationBucket("Red WG", Qual.WG, "Red", 0.075),
    ]
)