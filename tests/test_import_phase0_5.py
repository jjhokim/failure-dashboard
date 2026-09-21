"""
Phase 0-5 CSV/Excel import 어댑터 검증.

  - 표준화: 영어/한국어 컬럼 매핑, 필수 누락 감지, 미매핑 컬럼 반환
  - 고장이력_DataFrame저장: 표준 DataFrame 일괄 저장 (source='import')
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import adapters, crud, database  # noqa: E402


@pytest.fixture
def 임시DB(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    return tmp_path


def test_표준화_영어컬럼_매핑():
    df = pd.DataFrame({
        "echelon": ["부대정비"], "occurred_at": ["2025-01-01 00:00:00"],
        "system": ["체계-A"], "lru": ["LRU-001"], "fault_type": ["전기적"],
        "symptom": ["증상"], "status": ["수리완료"],
    })
    표준df, 누락, 미매핑 = adapters.표준화(df)
    assert 누락 == []
    assert 미매핑 == []
    assert set(["제대구분", "발생일시", "체계명", "LRU명",
                "고장유형", "고장증상", "처리상태"]) <= set(표준df.columns)
    assert 표준df.iloc[0]["제대구분"] == "부대정비"


def test_표준화_필수누락_감지():
    df = pd.DataFrame({"echelon": ["부대정비"], "system": ["체계-A"]})
    _, 누락, _ = adapters.표준화(df)
    assert "고장증상" in 누락
    assert "발생일시" in 누락


def test_표준화_미매핑_컬럼_반환():
    df = pd.DataFrame({
        "제대구분": ["부대정비"], "발생일시": ["2025-01-01 00:00:00"],
        "체계명": ["체계-A"], "LRU명": ["LRU-001"], "고장유형": ["전기적"],
        "고장증상": ["증상"], "처리상태": ["수리완료"],
        "알수없는컬럼": ["x"],
    })
    표준df, 누락, 미매핑 = adapters.표준화(df)
    assert 누락 == []
    assert "알수없는컬럼" in 미매핑
    assert "알수없는컬럼" not in 표준df.columns


def test_DataFrame저장_왕복(임시DB):
    df = pd.DataFrame({
        "echelon": ["부대정비", "야전정비"],
        "occurred_at": ["2025-01-01 00:00:00", "2025-02-01 00:00:00"],
        "system": ["체계-A", "체계-B"], "lru": ["LRU-001", "LRU-002"],
        "fault_type": ["전기적", "기계적"], "symptom": ["증상1", "증상2"],
        "status": ["수리완료", "수리중"], "lcn": ["A.01", "B.02"],
    })
    표준df, 누락, _ = adapters.표준화(df)
    assert 누락 == []

    before = len(crud.고장이력_전체조회())
    n = crud.고장이력_DataFrame저장(표준df, source="import")
    assert n == 2

    after = crud.고장이력_전체조회()
    assert len(after) == before + 2
    imported = after[after["source"] == "import"]
    assert len(imported) == 2
    assert set(imported["lcn"]) == {"A.01", "B.02"}
