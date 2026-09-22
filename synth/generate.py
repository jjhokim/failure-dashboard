"""
합성 정비데이터 생성 실행기 (Phase 1).

build_physical_states(규칙 기반) → narrator(문장) → 관측 CSV/정답 CSV/생성 로그 저장.
전 과정 결정론적(시드 고정, 벽시계 미사용). 라벨 누출 검사를 자동 실행해 로그에 남긴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend import constants
from synth.config import SynthConfig
from synth.leakage import run_leakage_checks
from synth.narrator import get_narrator
from synth.physical import BASE_DATE, build_physical_states, part_for_lcn

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = _ROOT / "data" / "synth"

# 증상 코드 → 고장유형 (대시보드 스키마 호환용)
_증상_고장유형 = {
    "과열": "환경적", "진동": "기계적", "누유": "기계적", "전압강하": "전기적",
    "통신두절": "전기적", "균열": "기계적", "오작동": "소프트웨어", "소음": "기계적",
}


def build_systems(config: SynthConfig) -> pd.DataFrame:
    """
    장비 모집단(systems)을 생성한다 — 고장률 분모(Σ운용시간) 정의용.

    각 장비의 누적 운용시간 = 관측기간(일) × op_hours_per_day × 개체별 가동계수.
    가동계수는 시드 고정 난수로, 장비별 운용 편차를 모사한다.
    체계(A/B/C)에 라운드로빈 배정한다.
    """
    rng = np.random.RandomState(config.seed + 10_000)  # 물리상태 rng와 분리
    체계들 = ["A", "B", "C"]
    rows = []
    for i in range(config.n_systems):
        s = 체계들[i % len(체계들)]
        가동계수 = float(rng.uniform(0.6, 1.0))
        rows.append({
            "system_id": f"SYS-{s}-{i // len(체계들) + 1:03d}",
            "system_type": f"체계-{s}",
            "echelon": constants.제대_목록[i % len(constants.제대_목록)],
            "commissioned_at": BASE_DATE.strftime("%Y-%m-%d %H:%M:%S"),
            "cumulative_op_hours": round(
                config.time_span_days * config.op_hours_per_day * 가동계수, 1
            ),
        })
    return pd.DataFrame(rows)


def _assemble_observable(states_df: pd.DataFrame, narratives: list[str]) -> pd.DataFrame:
    """
    탐지기·대시보드가 읽는 '관측 가능' 데이터프레임.
    잠재 물리상태(symptom_code·environment)는 컬럼으로 노출하지 않는다
    (탐지기가 텍스트를 거치지 않고 지름길을 타는 것을 방지).
    """
    from synth.physical import _stable_idx  # 결정론적 제대 배정에 재사용

    rows = []
    for (_, r), narrative in zip(states_df.iterrows(), narratives):
        lcn = r["lcn"]
        마지막 = int(lcn.split(".")[-1])
        rows.append({
            "record_id": int(r["record_id"]),
            "발생일시": pd.Timestamp(r["occurred_at"]).strftime("%Y-%m-%d %H:%M:%S"),
            "체계명": f"체계-{lcn.split('.')[0]}",
            "_occurred_at": pd.Timestamp(r["occurred_at"]),
            "LRU명": f"LRU-{마지막:03d}",
            "고장유형": _증상_고장유형.get(r["symptom_code"], "기타"),
            "고장증상": narrative,          # narrative == 관측 텍스트
            "lcn": lcn,
            # effect_class는 증상(물리상태)에서 확률적으로 파생된 관측값이다.
            # 탐지 채널에는 쓰이지 않고 Phase 3 치명도 근사(β·α)에만 사용된다.
            "effect_class": r["effect_class"],
            "제대구분": constants.제대_목록[_stable_idx(lcn, len(constants.제대_목록))],
            "처리상태": "수리완료",
            "source": "synthetic",
        })
    return pd.DataFrame(rows)


def _attach_systems(
    obs: pd.DataFrame, systems: pd.DataFrame, config: SynthConfig
) -> pd.DataFrame:
    """
    각 고장 레코드를 동일 체계(system_type)의 장비에 결정론적으로 귀속시키고,
    고장 시점 누적 운용시간(op_hours_at_failure)을 부여한다.

    op_hours_at_failure = 경과일 × op_hours_per_day × 해당 장비 가동계수
      (가동계수 = 장비 누적운용시간 / (전체기간 × op_hours_per_day))

    귀속은 record_id 기반 라운드로빈이라 군집 라벨과 무관하다(R1).
    """
    obs = obs.copy()
    if systems.empty:
        obs["system_id"] = None
        obs["op_hours_at_failure"] = np.nan
        return obs.drop(columns=["_occurred_at"])

    전체시간 = max(1e-9, config.time_span_days * config.op_hours_per_day)
    by_type: dict[str, list] = {}
    for _, s in systems.iterrows():
        by_type.setdefault(s["system_type"], []).append(s)

    sid, ohf = [], []
    for _, r in obs.iterrows():
        후보 = by_type.get(r["체계명"]) or [systems.iloc[0]]
        s = 후보[int(r["record_id"]) % len(후보)]
        가동계수 = float(s["cumulative_op_hours"]) / 전체시간
        경과일 = max(0.0, (r["_occurred_at"] - BASE_DATE).total_seconds() / 86400.0)
        sid.append(s["system_id"])
        ohf.append(round(경과일 * config.op_hours_per_day * 가동계수, 2))

    obs["system_id"] = sid
    obs["op_hours_at_failure"] = ohf
    return obs.drop(columns=["_occurred_at"])


def generate(
    config: SynthConfig,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    save: bool = True,
) -> dict:
    """
    합성 데이터 1세트를 생성한다.

    Returns
    -------
    dict
        obs(관측 df), truth(정답 df), leakage(누출 검사), counts, paths(저장 시)
    """
    states_df, truth_df = build_physical_states(config)

    narrator = get_narrator(config)
    narratives = [
        narrator.render({
            "part": part_for_lcn(r["lcn"]),
            "symptom_code": r["symptom_code"],
            "environment": r["environment"],
        })
        for _, r in states_df.iterrows()
    ]

    systems = build_systems(config)
    obs = _attach_systems(_assemble_observable(states_df, narratives), systems, config)

    # 라벨 누출 검사 (record_id 기준 정렬 일치 상태에서 실행)
    cluster_ids = list(truth_df["cluster_id"].astype(int))
    leakage = run_leakage_checks(list(obs["고장증상"]), cluster_ids)

    유효군집 = truth_df.loc[truth_df["cluster_id"] >= 0, "cluster_id"].nunique()
    counts = {
        "n_records": int(len(obs)),
        "n_clusters": int(유효군집),
        "n_noise": int((truth_df["cluster_id"] == -1).sum()),
    }

    counts["total_op_hours"] = float(systems["cumulative_op_hours"].sum())
    result = {
        "obs": obs, "truth": truth_df, "systems": systems,
        "leakage": leakage, "counts": counts,
    }

    if save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{config.seed}_{config.delta}"
        obs_path = out / f"{stem}.csv"
        truth_path = out / f"{stem}_truth.csv"
        systems_path = out / f"{stem}_systems.csv"
        log_path = out / f"{stem}_log.json"

        obs.to_csv(obs_path, index=False, encoding="utf-8-sig")
        truth_df.to_csv(truth_path, index=False, encoding="utf-8-sig")
        systems.to_csv(systems_path, index=False, encoding="utf-8-sig")
        log = {
            "config": config.__dict__,
            "counts": counts,
            "leakage": leakage,
            "files": {
                "obs": obs_path.name,
                "truth": truth_path.name,
                "systems": systems_path.name,
            },
        }
        log_path.write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["paths"] = {
            "obs": str(obs_path), "truth": str(truth_path),
            "systems": str(systems_path), "log": str(log_path),
        }

    return result
