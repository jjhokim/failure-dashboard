"""
CRUD 및 분석 지표 계산 모듈

구성:
  1. 입력     - 고장이력 단건 저장, 기존 레코드 수정
  2. 조회     - 전체/필터 조회, 단건 조회
  3. 삭제     - 단건 삭제
  4. 분석     - MTBF, MTTR, Ao(가용도) 계산
  5. 집계     - 파레토, 월별 트렌드, 제대별 비교, KPI 요약

반환 타입:
  - 조회·집계 함수는 모두 pandas DataFrame 반환 (Streamlit에서 바로 사용)
  - KPI 요약은 dict 반환
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from backend.database import get_connection
from backend.models import 고장이력


# ===========================================================================
# 1. 입력
# ===========================================================================

def 고장이력_저장(record: 고장이력) -> int:
    """
    고장이력 단건을 DB에 저장하고, 생성된 id를 반환한다.

    Parameters
    ----------
    record : 고장이력
        models.py의 고장이력 dataclass 인스턴스.
        id 필드는 무시된다 (AUTOINCREMENT).

    Returns
    -------
    int
        DB에서 할당된 새 레코드의 id.
    """
    sql = """
        INSERT INTO 고장이력 (
            제대구분, 발생일시, 체계명, LRU명, 고장유형, 고장증상,
            수리시작시간, 수리완료시간, 정비조치내용, 처리상태, 등록일시
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        record.제대구분,
        _fmt_dt(record.발생일시),
        record.체계명,
        record.LRU명,
        record.고장유형,
        record.고장증상,
        _fmt_dt(record.수리시작시간),
        _fmt_dt(record.수리완료시간),
        record.정비조치내용,
        record.처리상태,
        _fmt_dt(record.등록일시),
    )
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        return cursor.lastrowid


def 고장이력_수정(id: int, 수정내용: dict) -> None:
    """
    기존 레코드를 부분 수정한다.

    Parameters
    ----------
    id : int
        수정할 레코드의 id.
    수정내용 : dict
        {컬럼명: 새값} 형태. 예) {"처리상태": "수리완료", "수리완료시간": "2026-01-01 12:00:00"}
    """
    if not 수정내용:
        return

    # datetime 객체가 들어오면 문자열로 변환
    normalized = {k: (_fmt_dt(v) if isinstance(v, datetime) else v)
                  for k, v in 수정내용.items()}

    set_clause = ", ".join(f"{col} = ?" for col in normalized)
    values     = list(normalized.values()) + [id]

    with get_connection() as conn:
        conn.execute(f"UPDATE 고장이력 SET {set_clause} WHERE id = ?", values)


# ===========================================================================
# 2. 조회
# ===========================================================================

def 고장이력_전체조회(
    제대구분:   Optional[str]  = None,
    체계명:     Optional[str]  = None,
    고장유형:   Optional[str]  = None,
    처리상태:   Optional[str]  = None,
    날짜_시작:  Optional[str]  = None,   # "YYYY-MM-DD"
    날짜_종료:  Optional[str]  = None,   # "YYYY-MM-DD"
) -> pd.DataFrame:
    """
    필터 조건에 맞는 고장이력을 DataFrame으로 반환한다.
    모든 필터는 선택적이며, 지정하지 않으면 전체 조회된다.

    Returns
    -------
    pd.DataFrame
        발생일시 내림차순 정렬. 수리소요시간_h 파생 컬럼 포함.
    """
    sql    = "SELECT * FROM 고장이력 WHERE 1=1"
    params: list = []

    if 제대구분:
        sql += " AND 제대구분 = ?"
        params.append(제대구분)
    if 체계명:
        sql += " AND 체계명 = ?"
        params.append(체계명)
    if 고장유형:
        sql += " AND 고장유형 = ?"
        params.append(고장유형)
    if 처리상태:
        sql += " AND 처리상태 = ?"
        params.append(처리상태)
    if 날짜_시작:
        sql += " AND 발생일시 >= ?"
        params.append(f"{날짜_시작} 00:00:00")
    if 날짜_종료:
        sql += " AND 발생일시 <= ?"
        params.append(f"{날짜_종료} 23:59:59")

    sql += " ORDER BY 발생일시 DESC"

    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)

    return _add_파생컬럼(df)


def 고장이력_단건조회(id: int) -> Optional[pd.Series]:
    """id로 단건을 조회해 Series로 반환한다. 없으면 None."""
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT * FROM 고장이력 WHERE id = ?", conn, params=[id]
        )
    if df.empty:
        return None
    return _add_파생컬럼(df).iloc[0]


def 미해결_고장_조회(limit: int = 10) -> pd.DataFrame:
    """
    처리상태가 '수리중' 또는 '미해결'인 건을 발생일시 오름차순으로 반환한다.
    대시보드 요약 패널의 '최근 미해결 목록'에 사용된다.
    """
    sql = """
        SELECT * FROM 고장이력
        WHERE 처리상태 IN ('수리중', '미해결')
        ORDER BY 발생일시 ASC
        LIMIT ?
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=[limit])
    return _add_파생컬럼(df)


# ===========================================================================
# 3. 삭제
# ===========================================================================

def 고장이력_삭제(id: int) -> None:
    """id에 해당하는 레코드를 삭제한다."""
    with get_connection() as conn:
        conn.execute("DELETE FROM 고장이력 WHERE id = ?", [id])


# ===========================================================================
# 4. 분석 지표 계산
# ===========================================================================

def MTTR_계산(df: pd.DataFrame) -> float:
    """
    평균수리시간(MTTR) 계산.

    수리완료 건의 수리소요시간_h 평균을 반환한다.
    유효 데이터가 없으면 0.0 반환.

    Parameters
    ----------
    df : pd.DataFrame
        고장이력_전체조회() 결과.
    """
    완료건 = df[df["처리상태"] == "수리완료"]["수리소요시간_h"].dropna()
    if 완료건.empty:
        return 0.0
    return round(완료건.mean(), 2)


def MTBF_계산(df: pd.DataFrame, 관측기간_h: Optional[float] = None) -> float:
    """
    평균고장간격(MTBF) 계산.

    MTBF = 관측기간 / 고장건수
    관측기간을 지정하지 않으면 df의 첫 발생일시 ~ 마지막 발생일시 구간을 사용한다.

    Parameters
    ----------
    df : pd.DataFrame
        고장이력_전체조회() 결과.
    관측기간_h : float, optional
        명시적 관측 기간(시간). 미지정 시 데이터 범위로 자동 산출.
    """
    건수 = len(df)
    if 건수 == 0:
        return 0.0

    if 관측기간_h is None:
        dates = pd.to_datetime(df["발생일시"], errors="coerce").dropna()
        if len(dates) < 2:
            return 0.0
        관측기간_h = (dates.max() - dates.min()).total_seconds() / 3600

    if 관측기간_h <= 0:
        return 0.0

    return round(관측기간_h / 건수, 2)


def 가용도_계산(mtbf: float, mttr: float) -> float:
    """
    운용 가용도(Ao) 계산.

    Ao = MTBF / (MTBF + MTTR)
    MTBF + MTTR == 0 이면 0.0 반환.
    """
    분모 = mtbf + mttr
    if 분모 == 0:
        return 0.0
    return round(mtbf / 분모, 4)  # 소수점 4자리 (예: 0.8734 → 87.34%)


# ===========================================================================
# 5. 집계
# ===========================================================================

def KPI_요약(df: pd.DataFrame, 관측기간_h: Optional[float] = None) -> dict:
    """
    대시보드 KPI 카드용 요약 지표를 반환한다.

    Returns
    -------
    dict
        {
          "총_고장건수": int,
          "미완료_건수": int,
          "MTBF_h":      float,   # 시간 단위
          "MTTR_h":      float,   # 시간 단위
          "가용도_Ao":   float,   # 0~1 사이 소수
        }
    """
    mtbf = MTBF_계산(df, 관측기간_h)
    mttr = MTTR_계산(df)
    ao   = 가용도_계산(mtbf, mttr)

    미완료 = df[df["처리상태"] != "수리완료"]

    return {
        "총_고장건수": len(df),
        "미완료_건수": len(미완료),
        "MTBF_h":      mtbf,
        "MTTR_h":      mttr,
        "가용도_Ao":   ao,
    }


def 월별_고장건수(df: pd.DataFrame) -> pd.DataFrame:
    """
    월별 고장 트렌드 집계.

    Returns
    -------
    pd.DataFrame
        columns: ["연월", "고장건수", "제대구분"]
        연월 형식: "YYYY-MM"
    """
    if df.empty:
        return pd.DataFrame(columns=["연월", "고장건수", "제대구분"])

    tmp = df.copy()
    tmp["연월"] = pd.to_datetime(tmp["발생일시"], errors="coerce").dt.strftime("%Y-%m")
    result = (
        tmp.groupby(["연월", "제대구분"], as_index=False)
           .size()
           .rename(columns={"size": "고장건수"})
           .sort_values("연월")
    )
    return result


def 파레토_분석(df: pd.DataFrame, 기준컬럼: str = "고장유형") -> pd.DataFrame:
    """
    파레토 분석 — 고장유형(또는 LRU명)별 빈도 및 누적 비율 계산.

    Parameters
    ----------
    df : pd.DataFrame
        고장이력_전체조회() 결과.
    기준컬럼 : str
        "고장유형" 또는 "LRU명".

    Returns
    -------
    pd.DataFrame
        columns: [기준컬럼, "건수", "비율_%", "누적비율_%"]
        건수 내림차순 정렬.
    """
    if df.empty or 기준컬럼 not in df.columns:
        return pd.DataFrame(columns=[기준컬럼, "건수", "비율_%", "누적비율_%"])

    freq = (
        df[기준컬럼]
        .value_counts()
        .reset_index()
        .rename(columns={"index": 기준컬럼, "count": "건수",
                         기준컬럼: 기준컬럼, "proportion": "비율_%"})
    )
    # pandas 버전 호환: value_counts() 결과 컬럼명 정규화
    if "count" not in freq.columns and freq.shape[1] == 2:
        freq.columns = [기준컬럼, "건수"]

    freq = freq.sort_values("건수", ascending=False).reset_index(drop=True)
    전체 = freq["건수"].sum()
    freq["비율_%"]    = (freq["건수"] / 전체 * 100).round(1)
    freq["누적비율_%"] = freq["비율_%"].cumsum().round(1)
    return freq


def 체계별_MTTR(df: pd.DataFrame) -> pd.DataFrame:
    """
    체계명 × 제대구분별 평균 수리소요시간 집계.

    Returns
    -------
    pd.DataFrame
        columns: ["체계명", "제대구분", "평균MTTR_h", "고장건수"]
    """
    완료건 = df[df["처리상태"] == "수리완료"].copy()
    if 완료건.empty:
        return pd.DataFrame(columns=["체계명", "제대구분", "평균MTTR_h", "고장건수"])

    result = (
        완료건.groupby(["체계명", "제대구분"], as_index=False)
              .agg(평균MTTR_h=("수리소요시간_h", "mean"), 고장건수=("id", "count"))
    )
    result["평균MTTR_h"] = result["평균MTTR_h"].round(2)
    return result.sort_values(["체계명", "제대구분"])


def 제대별_고장현황(df: pd.DataFrame) -> pd.DataFrame:
    """
    제대구분 × 처리상태별 고장건수 집계.
    대시보드의 제대별 고장현황 비교 차트에 사용된다.

    Returns
    -------
    pd.DataFrame
        columns: ["제대구분", "처리상태", "건수"]
    """
    if df.empty:
        return pd.DataFrame(columns=["제대구분", "처리상태", "건수"])

    result = (
        df.groupby(["제대구분", "처리상태"], as_index=False)
          .size()
          .rename(columns={"size": "건수"})
    )
    return result.sort_values(["제대구분", "처리상태"])


def 불가동_추세(df: pd.DataFrame) -> pd.DataFrame:
    """
    월별 불가동 건수 추세 집계 (처리상태 = '수리중' 또는 '미해결').

    Returns
    -------
    pd.DataFrame
        columns: ["연월", "체계명", "불가동건수"]
    """
    불가동 = df[df["처리상태"] != "수리완료"].copy()
    if 불가동.empty:
        return pd.DataFrame(columns=["연월", "체계명", "불가동건수"])

    불가동["연월"] = pd.to_datetime(불가동["발생일시"], errors="coerce").dt.strftime("%Y-%m")
    result = (
        불가동.groupby(["연월", "체계명"], as_index=False)
              .size()
              .rename(columns={"size": "불가동건수"})
              .sort_values("연월")
    )
    return result


# ===========================================================================
# 내부 헬퍼
# ===========================================================================

def _fmt_dt(value) -> Optional[str]:
    """datetime → 'YYYY-MM-DD HH:MM:SS' 문자열 변환. None이면 None 반환."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value  # 이미 문자열인 경우


def _add_파생컬럼(df: pd.DataFrame) -> pd.DataFrame:
    """
    조회 결과 DataFrame에 분석용 파생 컬럼을 추가한다.

    추가 컬럼:
      - 수리소요시간_h : 수리완료시간 - 수리시작시간 (시간 단위, float)
      - 발생_연월      : 발생일시의 "YYYY-MM" 문자열 (월별 집계용)
    """
    if df.empty:
        df["수리소요시간_h"] = pd.Series(dtype=float)
        df["발생_연월"]     = pd.Series(dtype=str)
        return df

    시작 = pd.to_datetime(df["수리시작시간"], errors="coerce")
    완료 = pd.to_datetime(df["수리완료시간"], errors="coerce")
    df["수리소요시간_h"] = (완료 - 시작).dt.total_seconds() / 3600

    df["발생_연월"] = (
        pd.to_datetime(df["발생일시"], errors="coerce").dt.strftime("%Y-%m")
    )
    return df
