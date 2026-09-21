"""
합성 정비데이터 생성 실행기 (Phase 1).

build_physical_states(규칙 기반) → narrator(문장) → 관측 CSV/정답 CSV/생성 로그 저장.
전 과정 결정론적(시드 고정, 벽시계 미사용). 라벨 누출 검사를 자동 실행해 로그에 남긴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend import constants
from synth.config import SynthConfig
from synth.leakage import run_leakage_checks
from synth.narrator import get_narrator
from synth.physical import build_physical_states, part_for_lcn

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = _ROOT / "data" / "synth"

# 증상 코드 → 고장유형 (대시보드 스키마 호환용)
_증상_고장유형 = {
    "과열": "환경적", "진동": "기계적", "누유": "기계적", "전압강하": "전기적",
    "통신두절": "전기적", "균열": "기계적", "오작동": "소프트웨어", "소음": "기계적",
}


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

    obs = _assemble_observable(states_df, narratives)

    # 라벨 누출 검사 (record_id 기준 정렬 일치 상태에서 실행)
    cluster_ids = list(truth_df["cluster_id"].astype(int))
    leakage = run_leakage_checks(list(obs["고장증상"]), cluster_ids)

    유효군집 = truth_df.loc[truth_df["cluster_id"] >= 0, "cluster_id"].nunique()
    counts = {
        "n_records": int(len(obs)),
        "n_clusters": int(유효군집),
        "n_noise": int((truth_df["cluster_id"] == -1).sum()),
    }

    result = {"obs": obs, "truth": truth_df, "leakage": leakage, "counts": counts}

    if save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{config.seed}_{config.delta}"
        obs_path = out / f"{stem}.csv"
        truth_path = out / f"{stem}_truth.csv"
        log_path = out / f"{stem}_log.json"

        obs.to_csv(obs_path, index=False, encoding="utf-8-sig")
        truth_df.to_csv(truth_path, index=False, encoding="utf-8-sig")
        log = {
            "config": config.__dict__,
            "counts": counts,
            "leakage": leakage,
            "files": {"obs": obs_path.name, "truth": truth_path.name},
        }
        log_path.write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["paths"] = {
            "obs": str(obs_path), "truth": str(truth_path), "log": str(log_path)
        }

    return result
