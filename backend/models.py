"""
데이터 스키마 정의 모듈

고장이력 레코드의 Enum 상수 및 dataclass를 정의한다.
SQLite 테이블 DDL도 이 파일에서 관리한다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enum 상수
# ---------------------------------------------------------------------------

class 제대구분(str, Enum):
    """정비 제대 분류"""
    부대정비 = "부대정비"   # 1제대
    야전정비 = "야전정비"   # 2제대
    창정비   = "창정비"     # 3제대


class 체계명(str, Enum):
    """익명화된 가상 체계"""
    체계A = "체계-A"
    체계B = "체계-B"
    체계C = "체계-C"


class 고장유형(str, Enum):
    전기적    = "전기적"
    기계적    = "기계적"
    소프트웨어 = "소프트웨어"
    환경적    = "환경적"


class 처리상태(str, Enum):
    수리완료 = "수리완료"
    수리중   = "수리중"
    미해결   = "미해결"


class 임무영향구분(str, Enum):
    """
    고장이 임무 수행에 미치는 영향 (effect_class).
    RAM 분석에서 임무기반 MTBF(mtbf_mission) 분자 계수 기준으로 사용된다.
    """
    EFF  = "EFF"   # Mission-Effective failure (임무영향 고장)
    NEFF = "NEFF"  # Non-Effective failure (비임무영향 고장)


class 데이터출처(str, Enum):
    """
    레코드 출처 (sources / provenance).
    R6 검증용 배지 문구를 데이터 성격에 따라 동적 전환하는 근거가 된다.
    """
    샘플   = "sample"      # data/sample_data.csv 시드
    합성   = "synthetic"   # Phase 1 합성 생성기 산출
    임포트 = "import"      # 외부 CSV/Excel import


# LRU 명칭 목록 (LRU-001 ~ LRU-010)
LRU_목록: list[str] = [f"LRU-{i:03d}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# 고장이력 dataclass
# ---------------------------------------------------------------------------

@dataclass
class 고장이력:
    """
    고장이력 단일 레코드.

    - 필수 입력 필드: 발생일시, 체계명, LRU명, 고장유형, 고장증상, 제대구분, 처리상태
    - 선택 입력 필드: 수리시작시간, 수리완료시간, 정비조치내용
    - 자동 계산 필드: id (DB 자동증가), 등록일시 (레코드 생성 시각)
    - 파생 지표: MTTR은 수리시작/완료시간으로부터 계산 (crud.py에서 처리)
    """

    # --- 식별자 (DB 저장 후 채워짐) ---
    id: Optional[int] = field(default=None)

    # --- 필수 입력 ---
    제대구분:   str = field(default="")        # 제대구분 Enum 값
    발생일시:   datetime = field(default_factory=datetime.now)
    체계명:     str = field(default="")        # 체계명 Enum 값
    LRU명:      str = field(default="")        # LRU_목록 값
    고장유형:   str = field(default="")        # 고장유형 Enum 값
    고장증상:   str = field(default="")        # 자유 텍스트

    # --- 선택 입력 (수리 전 미확정 가능) ---
    수리시작시간:   Optional[datetime] = field(default=None)
    수리완료시간:   Optional[datetime] = field(default=None)
    정비조치내용:   Optional[str]      = field(default=None)  # 자유 텍스트

    # --- 상태 ---
    처리상태: str = field(default=처리상태.미해결.value)

    # --- 메타 ---
    등록일시: datetime = field(default_factory=datetime.now)  # 레코드 최초 입력 시각

    # --- Phase 0-2 확장 필드 (RAM·공통원인 분석용, 모두 선택 / NULL 허용) ---
    op_hours_at_failure: Optional[float] = field(default=None)  # 고장 시점 누적 운용시간
    aldt_hours:          Optional[float] = field(default=None)  # 행정·군수지연시간 (ALDT)
    mdt_hours:           Optional[float] = field(default=None)  # 정비중단시간 (MDT)
    lcn:                 Optional[str]   = field(default=None)  # 군수관리번호 (예: A.01.03.02)
    effect_class:        Optional[str]   = field(default=None)  # EFF / NEFF (임무영향구분)
    narrative:           Optional[str]   = field(default=None)  # 자유서술 정비기록
    system_id:           Optional[str]   = field(default=None)  # systems.system_id 링크
    source:              str = field(default=데이터출처.샘플.value)  # 레코드 출처

    def mttr_hours(self) -> Optional[float]:
        """
        MTTR(평균수리시간) 계산용 수리 소요 시간(시간 단위) 반환.
        수리시작시간 또는 수리완료시간이 없으면 None 반환.
        """
        if self.수리시작시간 is None or self.수리완료시간 is None:
            return None
        delta = self.수리완료시간 - self.수리시작시간
        return delta.total_seconds() / 3600


# ---------------------------------------------------------------------------
# SQLite 테이블 DDL
# ---------------------------------------------------------------------------

# 고장이력 테이블 생성 SQL.
# database.py의 init_db()에서 이 상수를 사용한다.
CREATE_고장이력_TABLE = """
CREATE TABLE IF NOT EXISTS 고장이력 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    제대구분        TEXT    NOT NULL,                  -- 부대정비 / 야전정비 / 창정비
    발생일시        TEXT    NOT NULL,                  -- ISO 8601 형식 (YYYY-MM-DD HH:MM:SS)
    체계명          TEXT    NOT NULL,                  -- 체계-A / 체계-B / 체계-C
    LRU명           TEXT    NOT NULL,                  -- LRU-001 ~ LRU-010
    고장유형        TEXT    NOT NULL,                  -- 전기적 / 기계적 / 소프트웨어 / 환경적
    고장증상        TEXT    NOT NULL,
    수리시작시간    TEXT,                              -- NULL 허용 (수리 전 미확정)
    수리완료시간    TEXT,                              -- NULL 허용 (수리 미완료)
    정비조치내용    TEXT,                              -- NULL 허용 (미조치 상태)
    처리상태        TEXT    NOT NULL DEFAULT '미해결', -- 수리완료 / 수리중 / 미해결
    등록일시        TEXT    NOT NULL,                  -- 레코드 최초 입력 시각
    -- ── Phase 0-2 확장 필드 (모두 NULL 허용 → 구 CSV 로드 호환 유지) ──
    op_hours_at_failure REAL,                          -- 고장 시점 누적 운용시간
    aldt_hours          REAL,                          -- 행정·군수지연시간 (ALDT)
    mdt_hours           REAL,                          -- 정비중단시간 (MDT)
    lcn                 TEXT,                          -- 군수관리번호 (예: A.01.03.02)
    effect_class        TEXT,                          -- EFF / NEFF
    narrative           TEXT,                          -- 자유서술 정비기록
    system_id           TEXT,                          -- systems.system_id 링크
    source              TEXT    DEFAULT 'sample'       -- 레코드 출처 (sample/synthetic/import)
);
"""

# ---------------------------------------------------------------------------
# 확장 테이블 dataclass (Phase 0-2)
# ---------------------------------------------------------------------------

@dataclass
class 장비:
    """
    systems 테이블 레코드 — 장비 모집단(모수).
    고장률 분모(Σ누적 운용시간) 정의에 사용된다.
    """
    system_id:           str            = ""
    system_type:         str            = ""            # 체계-A / 체계-B / 체계-C 매핑
    echelon:             Optional[str]  = None          # 배치 제대
    commissioned_at:     Optional[datetime] = None      # 운용 개시 일시
    cumulative_op_hours: float          = 0.0           # 누적 운용시간


@dataclass
class LCN노드:
    """lcn_tree 테이블 레코드 — LCN 계층 구조(구조거리 계산용)."""
    lcn:         str           = ""
    parent_lcn:  Optional[str] = None                   # 루트는 None
    level:       int           = 0                       # 계층 깊이 (루트=0)
    description: Optional[str] = None


@dataclass
class BIT이벤트:
    """
    bit_events 테이블 레코드 — Phase 5 조인용.
    ※ R3: 탐지 채널 입력이 아니다. LCN·시각 윈도우 조인까지만 사용한다.
    """
    event_id:    Optional[int]      = None
    system_id:   Optional[str]      = None
    lcn:         Optional[str]      = None
    bit_code:    str                = ""
    occurred_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# 확장 테이블 DDL (Phase 0-2)
# ---------------------------------------------------------------------------

CREATE_systems_TABLE = """
CREATE TABLE IF NOT EXISTS systems (
    system_id           TEXT    PRIMARY KEY,            -- 개별 장비 식별자 (예: SYS-A-001)
    system_type         TEXT    NOT NULL,               -- 체계-A / 체계-B / 체계-C
    echelon             TEXT,                           -- 배치 제대
    commissioned_at     TEXT,                           -- 운용 개시 일시 (ISO 8601)
    cumulative_op_hours REAL    NOT NULL DEFAULT 0      -- 누적 운용시간 (고장률 분모)
);
"""

CREATE_lcn_tree_TABLE = """
CREATE TABLE IF NOT EXISTS lcn_tree (
    lcn         TEXT    PRIMARY KEY,                     -- 계층 문자열 (예: A.01.03.02)
    parent_lcn  TEXT,                                    -- 상위 LCN (루트는 NULL)
    level       INTEGER NOT NULL,                        -- 계층 깊이 (루트=0)
    description TEXT
);
"""

CREATE_bit_events_TABLE = """
CREATE TABLE IF NOT EXISTS bit_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    system_id   TEXT,                                    -- systems.system_id 참조
    lcn         TEXT,                                    -- lcn_tree.lcn 참조
    bit_code    TEXT    NOT NULL,
    occurred_at TEXT    NOT NULL                         -- 발생 일시 (ISO 8601)
);
"""

# init_db()에서 순회 생성하는 확장 테이블 DDL 목록
CREATE_확장테이블_SQL: list[str] = [
    CREATE_systems_TABLE,
    CREATE_lcn_tree_TABLE,
    CREATE_bit_events_TABLE,
]

# 자주 사용하는 조회 조건에 대한 인덱스
# (고장이력.lcn 인덱스는 마이그레이션으로 컬럼 추가 후 생성되어야 하므로
#  database.init_db()에서 마이그레이션 이후에 실행한다.)
CREATE_인덱스_SQL: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_제대구분  ON 고장이력(제대구분);",
    "CREATE INDEX IF NOT EXISTS idx_체계명    ON 고장이력(체계명);",
    "CREATE INDEX IF NOT EXISTS idx_발생일시  ON 고장이력(발생일시);",
    "CREATE INDEX IF NOT EXISTS idx_처리상태  ON 고장이력(처리상태);",
    "CREATE INDEX IF NOT EXISTS idx_고장이력_lcn ON 고장이력(lcn);",
    "CREATE INDEX IF NOT EXISTS idx_bit_lcn      ON bit_events(lcn);",
    "CREATE INDEX IF NOT EXISTS idx_bit_occurred ON bit_events(occurred_at);",
    "CREATE INDEX IF NOT EXISTS idx_bit_system   ON bit_events(system_id);",
    "CREATE INDEX IF NOT EXISTS idx_lcn_parent   ON lcn_tree(parent_lcn);",
]

# 고장이력 테이블의 Phase 0-2 확장 컬럼 (마이그레이션 시 존재 여부 확인용).
# {컬럼명: SQL 타입 선언}. 구 DB에 없는 컬럼만 ALTER TABLE로 추가한다.
고장이력_확장컬럼: dict[str, str] = {
    "op_hours_at_failure": "REAL",
    "aldt_hours":          "REAL",
    "mdt_hours":           "REAL",
    "lcn":                 "TEXT",
    "effect_class":        "TEXT",
    "narrative":           "TEXT",
    "system_id":           "TEXT",
    "source":              "TEXT DEFAULT 'sample'",
}
