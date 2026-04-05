"""
SQLite 연결 및 테이블 초기화 모듈

주요 역할:
  - DB 파일 경로 관리 (data/failure.db)
  - 테이블 및 인덱스 생성 (최초 1회)
  - 샘플 데이터 seed (DB가 비어 있을 때만 실행)
  - 커넥션 컨텍스트 매니저 제공 (crud.py에서 사용)
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from backend.models import CREATE_고장이력_TABLE, CREATE_인덱스_SQL

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------

# 프로젝트 루트: backend/ 의 부모 디렉터리
_ROOT = Path(__file__).resolve().parent.parent

# SQLite DB 파일 경로
DB_PATH = _ROOT / "data" / "failure.db"

# 샘플 데이터 CSV 경로
SAMPLE_CSV_PATH = _ROOT / "data" / "sample_data.csv"


# ---------------------------------------------------------------------------
# 커넥션 관리
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """
    SQLite 커넥션 컨텍스트 매니저.

    사용 예:
        with get_connection() as conn:
            df = pd.read_sql("SELECT * FROM 고장이력", conn)
    """
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    # 딕셔너리 형태로 row 접근 가능하도록 설정
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    DB 파일과 테이블을 초기화한다.

    - DB 파일이 없으면 자동 생성된다 (SQLite 특성).
    - 테이블이 이미 존재하면 CREATE IF NOT EXISTS로 건너뛴다.
    - 고장이력 테이블이 비어 있으면 샘플 데이터를 seed한다.
    """
    # data/ 디렉터리가 없으면 생성
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        cursor = conn.cursor()

        # 테이블 생성
        cursor.execute(CREATE_고장이력_TABLE)

        # 인덱스 생성
        for sql in CREATE_인덱스_SQL:
            cursor.execute(sql)

    # 테이블이 비어 있으면 샘플 데이터 삽입
    _seed_if_empty()


def _seed_if_empty() -> None:
    """고장이력 테이블이 비어 있을 때만 샘플 CSV를 DB에 삽입한다."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM 고장이력")
        count = cursor.fetchone()[0]

        if count > 0:
            return  # 이미 데이터 존재 → seed 생략

        if not SAMPLE_CSV_PATH.exists():
            print(f"[경고] 샘플 데이터 파일 없음: {SAMPLE_CSV_PATH}")
            return

        df = pd.read_csv(SAMPLE_CSV_PATH, encoding="utf-8-sig")

        # id 컬럼이 CSV에 포함된 경우 제거 (DB AUTOINCREMENT 사용)
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        df.to_sql("고장이력", conn, if_exists="append", index=False)
        print(f"[정보] 샘플 데이터 {len(df)}건 삽입 완료")


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def reset_db() -> None:
    """
    DB를 완전히 초기화한다 (테이블 삭제 후 재생성 + seed).
    개발/테스트 용도로만 사용할 것.
    """
    with get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS 고장이력")

    init_db()
    print("[정보] DB 초기화 완료")


def get_db_stats() -> dict:
    """DB 상태 요약 반환 (대시보드 디버그 화면 등에서 사용)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM 고장이력")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM 고장이력 WHERE 처리상태 != '수리완료'"
        )
        미완료 = cursor.fetchone()[0]

    return {
        "총_고장건수": total,
        "미완료_건수": 미완료,
        "db_경로": str(DB_PATH),
    }
