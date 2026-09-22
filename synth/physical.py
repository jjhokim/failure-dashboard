"""
[규칙 기반] 군집 라벨 → 중간 물리상태 변환 (Phase 1, R1 준수).

절대 규칙 R1: 군집 라벨→중간 물리상태 변환은 이 규칙 기반 코드로만 수행한다.
LLM(narrator)은 이 물리상태만 받으며 라벨을 보지 않는다.
delta=0.0이면 물리상태에 군집 구조가 전혀 없어야 한다(무주입 대조군).

이 모듈은 KoSBERT 등 탐지기 구성요소를 import하지 않는다(R2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synth.config import SynthConfig

BASE_DATE = pd.Timestamp("2025-01-01")

# 증상 코드 및 부품 풀 (관측 텍스트가 아니라 잠재 물리상태 코드)
SYMPTOMS = ["과열", "진동", "누유", "전압강하", "통신두절", "균열", "오작동", "소음"]

# 증상별 임무영향(EFF) 발생 확률.
# effect_class는 '증상'(물리상태)에서 확률적으로 파생되며, 군집 라벨의 함수가
# 아니다(R1). 탐지 채널(의미·구조·시간)은 effect_class를 사용하지 않으므로
# ARI 산출에 순환성을 만들지 않는다. 우선순위화(Phase 3)의 β·α 근사에만 쓰인다.
EFF_확률 = {
    "통신두절": 0.90, "전압강하": 0.70, "오작동": 0.70, "과열": 0.50,
    "균열": 0.50, "누유": 0.30, "진동": 0.30, "소음": 0.10,
}
ENVIRONMENTS = ["고온", "저온", "습윤", "분진", "진동환경", "정상"]
PARTS = [
    "구동모터", "전원공급장치", "제어보드", "유압펌프", "센서모듈",
    "통신모듈", "냉각팬", "기어박스", "배터리팩", "안테나",
]


def _stable_idx(s: str, n: int) -> int:
    """문자열 → 결정론적 인덱스(PYTHONHASHSEED 비의존)."""
    return sum(ord(ch) for ch in s) % n


def part_for_lcn(lcn: str) -> str:
    """LCN(구조 위치)에 결정론적으로 대응하는 부품명."""
    key = ".".join(lcn.split(".")[:3])
    return PARTS[_stable_idx(key, len(PARTS))]


def _build_lcn_pool(rng: np.random.RandomState, n_leaves: int) -> list[str]:
    """계층형 LCN 리프 풀 생성 (예: A.02.03.05)."""
    systems = ["A", "B", "C"]
    pool: set[str] = set()
    while len(pool) < n_leaves:
        s = systems[rng.randint(len(systems))]
        a, b, c = rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 10)
        pool.add(f"{s}.{a:02d}.{b:02d}.{c:02d}")
    return sorted(pool)


def _near_lcn(rep: str, rng: np.random.RandomState) -> str:
    """대표 LCN과 같은 하위계통(prefix 공유)에서 리프만 다른 LCN."""
    parts = rep.split(".")
    parts[-1] = f"{rng.randint(1, 10):02d}"
    return ".".join(parts)


def _time_window(config: SynthConfig, span: float) -> float:
    """
    군집 시각 밀집폭(일). delta가 클수록 좁아진다.

    모드 (사전등록 v2 Arm B):
      - "legacy": max(1.0, span×0.05×(1−δ) + 1.0)
                  δ=1에서 1.0일로 수렴 → 시간 채널이 거의 완벽한 판별자가 되어
                  채널 비교(C4)를 왜곡한다는 의심을 받은 기존 동작.
      - "floor" : span×0.05×(1−δ) + span×0.01
                  δ=1에서도 span의 1%(365일 기준 3.65일) 폭을 유지한다.
    """
    if getattr(config, "time_window_mode", "legacy") == "floor":
        return span * 0.05 * (1 - config.delta) + span * 0.01
    return max(1.0, span * 0.05 * (1 - config.delta) + 1.0)


def _cluster_sizes(
    n_signal: int, k: int, dist: str, rng: np.random.RandomState
) -> list[int]:
    if k <= 0 or n_signal <= 0:
        return []
    if dist == "geometric":
        w = np.array([0.5 ** i for i in range(k)], dtype=float)
        w /= w.sum()
        sizes = list(map(int, rng.multinomial(n_signal, w)))
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(k))
        sizes = list(map(int, rng.multinomial(n_signal, w)))
    else:  # uniform
        base = n_signal // k
        sizes = [base] * k
        for i in range(n_signal - base * k):
            sizes[i] += 1
    diff = n_signal - sum(sizes)
    sizes[0] += diff
    return sizes


def build_physical_states(
    config: SynthConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    군집 라벨 → 중간 물리상태(규칙 기반)를 생성한다.

    Returns
    -------
    (states_df, truth_df)
        states_df : record_id, lcn, symptom_code, environment, occurred_at
                    (※ cluster_id·β·α·λ 미포함 — narrator로 흐르는 관측 가능 물리상태)
        truth_df  : record_id, cluster_id, beta, alpha, lambda_, t_star, Cm
                    (정답 파일 — 라벨·치명도. narrator·탐지기로 절대 전달하지 않음)
    """
    rng = np.random.RandomState(config.seed)
    n = config.n_records
    n_noise = int(round(n * config.noise_ratio))
    n_signal = n - n_noise
    span = config.time_span_days

    lcn_pool = _build_lcn_pool(rng, config.n_lcn_leaves)
    sizes = _cluster_sizes(n_signal, config.n_clusters, config.cluster_size_dist, rng)

    # 군집 대표 물리상태
    clusters = []
    for _ in range(config.n_clusters):
        clusters.append(dict(
            rep_lcn=lcn_pool[rng.randint(len(lcn_pool))],
            rep_sym=SYMPTOMS[rng.randint(len(SYMPTOMS))],
            rep_env=ENVIRONMENTS[rng.randint(len(ENVIRONMENTS))],
            t_star=float(rng.uniform(0, span * 0.7)),
            lam=config.lambda_ * (0.5 + rng.rand() * 2.0),
        ))

    records, truth = [], []
    rid = 0

    for c, size in enumerate(sizes):
        cl = clusters[c]
        for _ in range(size):
            # delta에 비례한 각 축의 신호 주입 여부 (delta=0 → 전부 무신호)
            sig_lcn = rng.rand() < config.delta
            sig_sym = rng.rand() < config.delta
            sig_env = rng.rand() < config.delta
            sig_time = rng.rand() < config.delta

            lcn = _near_lcn(cl["rep_lcn"], rng) if sig_lcn else lcn_pool[rng.randint(len(lcn_pool))]
            sym = cl["rep_sym"] if sig_sym else SYMPTOMS[rng.randint(len(SYMPTOMS))]
            env = cl["rep_env"] if sig_env else ENVIRONMENTS[rng.randint(len(ENVIRONMENTS))]

            if sig_time:
                window = _time_window(config, span)
                day = cl["t_star"] + rng.uniform(-window, window)
            else:
                day = rng.uniform(0, span)
            day = float(min(max(day, 0), span))

            records.append(dict(
                record_id=rid, lcn=lcn, symptom_code=sym,
                environment=env, occurred_at=BASE_DATE + pd.Timedelta(days=day),
                effect_class="EFF" if rng.rand() < EFF_확률.get(sym, 0.5) else "NEFF",
            ))
            Cm = config.beta * config.alpha * cl["lam"] * (day + 1.0)
            truth.append(dict(
                record_id=rid, cluster_id=c, beta=config.beta, alpha=config.alpha,
                lambda_=cl["lam"], t_star=cl["t_star"], Cm=Cm,
            ))
            rid += 1

    # 노이즈(단발성) — LCN·시각·증상·환경 전부 균등
    for _ in range(n_noise):
        day = float(rng.uniform(0, span))
        sym = SYMPTOMS[rng.randint(len(SYMPTOMS))]
        records.append(dict(
            record_id=rid,
            lcn=lcn_pool[rng.randint(len(lcn_pool))],
            symptom_code=sym,
            environment=ENVIRONMENTS[rng.randint(len(ENVIRONMENTS))],
            occurred_at=BASE_DATE + pd.Timedelta(days=day),
            effect_class="EFF" if rng.rand() < EFF_확률.get(sym, 0.5) else "NEFF",
        ))
        truth.append(dict(
            record_id=rid, cluster_id=-1, beta=0.0, alpha=0.0,
            lambda_=0.0, t_star=np.nan, Cm=0.0,
        ))
        rid += 1

    states_df = pd.DataFrame(records)
    truth_df = pd.DataFrame(truth)

    # 순서가 라벨을 누출하지 않도록 셔플
    perm = rng.permutation(len(states_df))
    states_df = states_df.iloc[perm].reset_index(drop=True)
    truth_df = (
        truth_df.set_index("record_id")
        .loc[states_df["record_id"]]
        .reset_index()
    )
    return states_df, truth_df
