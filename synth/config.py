"""
합성 정비데이터 생성 파라미터 (Phase 1).

주입 파라미터를 dataclass로 관리한다. 난수는 항상 seed 인자로 통제한다(원칙 #5).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SynthConfig:
    """합성 생성기 주입 파라미터."""

    # 규모
    n_records: int = 300
    n_clusters: int = 5

    # 군집 크기 분포: "uniform" | "geometric" | "dirichlet"
    cluster_size_dist: str = "uniform"

    # 공통원인 신호 강도 (0.0/0.25/0.5/0.75/1.0). 0.0이면 군집 구조 없음(대조군).
    delta: float = 0.5

    # 표현 편차 수준(0~1). 높을수록 동일 증상을 다양하게 서술(기록자 편차 모사).
    expression_variance: float = 0.5

    # 단발성(노이즈) 고장 비율.
    noise_ratio: float = 0.3

    # FMECA 치명도 파라미터 (군집별 lambda_에 배수 적용).
    beta: float = 1.0        # 고장영향확률
    alpha: float = 0.1       # 고장모드비
    lambda_: float = 1e-4    # 기준 고장률

    # 난수 시드 (전역 상태 비의존; 항상 명시)
    seed: int = 0

    # 도메인 크기
    n_lcn_leaves: int = 60
    time_span_days: int = 365

    # 장비 모집단 (고장률 분모 정의용 — Phase 0-2 systems 테이블 대응)
    n_systems: int = 12                     # 체계별 장비 대수 합
    op_hours_per_day: float = 8.0           # 장비 1대의 1일 평균 운용시간
