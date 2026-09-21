"""
Phase 0-2 스키마 재설계 검증 테스트.

검증 대상:
  - 확장 테이블(systems / lcn_tree / bit_events) 생성
  - 고장이력 확장 컬럼(op_hours_at_failure 등) 존재
  - 구 스키마 → 신 스키마 마이그레이션 (멱등, 기존 행 보존)
  - 구 CSV 시드 호환 (source 기본값 'sample')
  - 신규 테이블 CRUD 왕복 (systems / lcn_tree / bit_events)
  - 고장이력 확장 필드 저장·조회 왕복

원칙 #7(기존 UI/배포/구 CSV 무손상)을 코드로 보증한다.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import crud, database  # noqa: E402
from backend.models import (  # noqa: E402
    BIT이벤트,
    LCN노드,
    고장이력,
    고장이력_확장컬럼,
    장비,
)

# 구(舊) 스키마 — Phase 0-2 이전의 원본 고장이력 테이블 (확장 컬럼 없음)
OLD_DDL = """
CREATE TABLE 고장이력 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    제대구분 TEXT, 발생일시 TEXT, 체계명 TEXT, LRU명 TEXT,
    고장유형 TEXT, 고장증상 TEXT, 수리시작시간 TEXT, 수리완료시간 TEXT,
    정비조치내용 TEXT, 처리상태 TEXT, 등록일시 TEXT
);
"""


def _컬럼목록(table: str) -> set:
    with database.get_connection() as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


@pytest.fixture
def 임시DB(tmp_path, monkeypatch):
    """임시 경로에 새 DB를 만들어 init_db()로 전체 스키마를 구성한다."""
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    return tmp_path


# ---------------------------------------------------------------------------
# 스키마 생성
# ---------------------------------------------------------------------------

def test_확장테이블_생성(임시DB):
    with database.get_connection() as conn:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"고장이력", "systems", "lcn_tree", "bit_events"} <= tables


def test_고장이력_확장컬럼_존재(임시DB):
    assert set(고장이력_확장컬럼) <= _컬럼목록("고장이력")


def test_샘플시드_source_기본값(임시DB):
    # 구 CSV(100건, source 컬럼 없음)가 로드되고 source 기본값이 채워져야 한다
    df = crud.고장이력_전체조회()
    assert len(df) == 100
    assert (df["source"] == "sample").all()


# ---------------------------------------------------------------------------
# 마이그레이션 (구 스키마 → 신 스키마)
# ---------------------------------------------------------------------------

def test_마이그레이션_멱등_및_행보존(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "old.db")

    # 구 스키마 테이블 생성 + 기존 행 1건
    with database.get_connection() as conn:
        conn.execute(OLD_DDL)
        conn.execute(
            "INSERT INTO 고장이력 "
            "(제대구분, 발생일시, 체계명, LRU명, 고장유형, 고장증상, 처리상태, 등록일시) "
            "VALUES ('부대정비','2025-01-01 00:00:00','체계-A','LRU-001',"
            "'전기적','증상','미해결','2025-01-01 00:00:00')"
        )

    assert "lcn" not in _컬럼목록("고장이력")  # 마이그레이션 전

    # 마이그레이션 1회
    with database.get_connection() as conn:
        database._migrate_고장이력_schema(conn)
    assert set(고장이력_확장컬럼) <= _컬럼목록("고장이력")

    # 기존 행이 보존되어야 함
    df = crud.고장이력_전체조회()
    assert len(df) == 1
    assert df.iloc[0]["제대구분"] == "부대정비"

    # 멱등: 다시 실행해도 예외 없이 그대로여야 함
    with database.get_connection() as conn:
        database._migrate_고장이력_schema(conn)
    assert set(고장이력_확장컬럼) <= _컬럼목록("고장이력")


# ---------------------------------------------------------------------------
# 신규 테이블 CRUD 왕복
# ---------------------------------------------------------------------------

def test_systems_crud_upsert(임시DB):
    n = crud.장비_저장([
        장비(system_id="SYS-A-001", system_type="체계-A",
             echelon="부대정비", cumulative_op_hours=1200.0),
        장비(system_id="SYS-A-002", system_type="체계-A",
             echelon="야전정비", cumulative_op_hours=800.0),
    ])
    assert n == 2

    df = crud.장비_전체조회().set_index("system_id")
    assert len(df) == 2
    assert df.loc["SYS-A-001", "cumulative_op_hours"] == 1200.0

    # 동일 PK 재저장 → UPSERT (건수 불변, 값 갱신)
    crud.장비_저장(장비(system_id="SYS-A-001", system_type="체계-A",
                       cumulative_op_hours=1500.0))
    df2 = crud.장비_전체조회().set_index("system_id")
    assert len(df2) == 2
    assert df2.loc["SYS-A-001", "cumulative_op_hours"] == 1500.0


def test_lcn_tree_crud(임시DB):
    crud.LCN트리_저장([
        LCN노드(lcn="A",    parent_lcn=None, level=0, description="체계-A 루트"),
        LCN노드(lcn="A.01", parent_lcn="A",  level=1, description="구동부"),
    ])
    df = crud.LCN트리_전체조회()
    assert set(df["lcn"]) == {"A", "A.01"}


def test_bit_events_crud(임시DB):
    n = crud.BIT이벤트_저장([
        BIT이벤트(system_id="SYS-A-001", lcn="A.01",
                 bit_code="BIT-001", occurred_at=datetime(2025, 1, 1, 0, 0, 0)),
    ])
    assert n == 1
    df = crud.BIT이벤트_전체조회()
    assert len(df) == 1
    assert df.iloc[0]["bit_code"] == "BIT-001"


# ---------------------------------------------------------------------------
# 고장이력 확장 필드 저장·조회 왕복
# ---------------------------------------------------------------------------

def test_고장이력_확장필드_왕복(임시DB):
    rec = 고장이력(
        제대구분="부대정비", 발생일시=datetime(2025, 3, 1, 0, 0, 0),
        체계명="체계-A", LRU명="LRU-001", 고장유형="전기적",
        고장증상="증상", 처리상태="수리완료", 등록일시=datetime(2025, 3, 1, 0, 0, 0),
        op_hours_at_failure=345.6, aldt_hours=2.0, mdt_hours=5.0,
        lcn="A.01.03.02", effect_class="EFF", narrative="정비 서술",
        system_id="SYS-A-001", source="synthetic",
    )
    new_id = crud.고장이력_저장(rec)
    row = crud.고장이력_단건조회(new_id)

    assert row["lcn"] == "A.01.03.02"
    assert row["effect_class"] == "EFF"
    assert row["source"] == "synthetic"
    assert row["op_hours_at_failure"] == 345.6
