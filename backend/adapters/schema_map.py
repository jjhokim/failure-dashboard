"""
외부 CSV/Excel → 내부 고장이력 스키마 매핑 어댑터 (Phase 0-5).

외부 파일의 컬럼명을 내부 필드명으로 매핑하고, 필수 컬럼 누락·매핑 실패
컬럼을 오류/경고로 반환한다. 실제 저장은 crud.고장이력_DataFrame저장에서 수행.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# 내부 필드 정의 (고장이력 테이블 기준)
# ---------------------------------------------------------------------------

내부_필수: list[str] = [
    "제대구분", "발생일시", "체계명", "LRU명", "고장유형", "고장증상", "처리상태",
]

내부_선택: list[str] = [
    "수리시작시간", "수리완료시간", "정비조치내용",
    "op_hours_at_failure", "aldt_hours", "mdt_hours",
    "lcn", "effect_class", "narrative", "system_id",
]

# 외부 컬럼명 → 내부 필드명. 한국어는 항등 매핑 + 영어 별칭을 함께 지원한다.
기본_매핑: dict[str, str] = {
    # --- 한국어 (항등) ---
    "제대구분": "제대구분", "발생일시": "발생일시", "체계명": "체계명",
    "LRU명": "LRU명", "고장유형": "고장유형", "고장증상": "고장증상",
    "처리상태": "처리상태", "수리시작시간": "수리시작시간",
    "수리완료시간": "수리완료시간", "정비조치내용": "정비조치내용",
    "lcn": "lcn", "effect_class": "effect_class", "narrative": "narrative",
    "system_id": "system_id", "op_hours_at_failure": "op_hours_at_failure",
    "aldt_hours": "aldt_hours", "mdt_hours": "mdt_hours",
    # --- 영어 별칭 ---
    "echelon": "제대구분", "occurred_at": "발생일시", "date": "발생일시",
    "system": "체계명", "system_type": "체계명", "lru": "LRU명",
    "fault_type": "고장유형", "failure_type": "고장유형",
    "symptom": "고장증상", "status": "처리상태",
    "repair_start": "수리시작시간", "repair_end": "수리완료시간",
    "action": "정비조치내용", "op_hours": "op_hours_at_failure",
    "aldt": "aldt_hours", "mdt": "mdt_hours",
}


def 표준화(
    df: pd.DataFrame, 매핑: dict[str, str] | None = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    외부 DataFrame을 내부 스키마로 표준화한다.

    Returns
    -------
    (표준df, 누락_필수, 미매핑_컬럼)
        표준df    : 내부 필드명으로 rename되고 알려진 컬럼만 남긴 DataFrame
        누락_필수 : 매핑 결과에 없는 내부 필수 컬럼명 목록 (있으면 저장 차단)
        미매핑_컬럼: 매핑 사전에 없어 무시된 외부 컬럼명 목록 (경고)
    """
    매핑 = 매핑 or 기본_매핑

    rename_map: dict[str, str] = {}
    미매핑: list[str] = []
    for 외부컬럼 in df.columns:
        if 외부컬럼 in 매핑:
            rename_map[외부컬럼] = 매핑[외부컬럼]
        else:
            미매핑.append(외부컬럼)

    표준df = df.rename(columns=rename_map)
    알려진 = 내부_필수 + 내부_선택
    표준df = 표준df[[c for c in 표준df.columns if c in 알려진]].copy()
    # 중복 매핑 시 뒤 컬럼 우선 제거(첫 컬럼 유지)
    표준df = 표준df.loc[:, ~표준df.columns.duplicated()]

    누락_필수 = [c for c in 내부_필수 if c not in 표준df.columns]
    return 표준df, 누락_필수, 미매핑


def CSV_읽기(source) -> pd.DataFrame:
    """CSV 파일/버퍼를 읽는다 (UTF-8 BOM 허용)."""
    return pd.read_csv(source, encoding="utf-8-sig")


def Excel_읽기(source) -> pd.DataFrame:
    """Excel(.xlsx) 파일/버퍼를 읽는다."""
    return pd.read_excel(source, engine="openpyxl")


def 읽기(source, 파일명: str) -> pd.DataFrame:
    """파일 확장자에 따라 CSV/Excel 리더를 선택한다."""
    if 파일명.lower().endswith((".xlsx", ".xls")):
        return Excel_읽기(source)
    return CSV_읽기(source)
