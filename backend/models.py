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
    운용부대 = "운용부대"   # 1제대
    정비부대 = "정비부대"   # 2제대
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
    제대구분        TEXT    NOT NULL,                  -- 운용부대 / 정비부대 / 창정비
    발생일시        TEXT    NOT NULL,                  -- ISO 8601 형식 (YYYY-MM-DD HH:MM:SS)
    체계명          TEXT    NOT NULL,                  -- 체계-A / 체계-B / 체계-C
    LRU명           TEXT    NOT NULL,                  -- LRU-001 ~ LRU-010
    고장유형        TEXT    NOT NULL,                  -- 전기적 / 기계적 / 소프트웨어 / 환경적
    고장증상        TEXT    NOT NULL,
    수리시작시간    TEXT,                              -- NULL 허용 (수리 전 미확정)
    수리완료시간    TEXT,                              -- NULL 허용 (수리 미완료)
    정비조치내용    TEXT,                              -- NULL 허용 (미조치 상태)
    처리상태        TEXT    NOT NULL DEFAULT '미해결', -- 수리완료 / 수리중 / 미해결
    등록일시        TEXT    NOT NULL                   -- 레코드 최초 입력 시각
);
"""

# 자주 사용하는 조회 조건에 대한 인덱스
CREATE_인덱스_SQL: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_제대구분  ON 고장이력(제대구분);",
    "CREATE INDEX IF NOT EXISTS idx_체계명    ON 고장이력(체계명);",
    "CREATE INDEX IF NOT EXISTS idx_발생일시  ON 고장이력(발생일시);",
    "CREATE INDEX IF NOT EXISTS idx_처리상태  ON 고장이력(처리상태);",
]
