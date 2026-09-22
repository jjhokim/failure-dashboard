"""
모의 BIT(Built-In Test) 로그 생성 (Phase 5).

**절대 규칙 R3**: BIT는 탐지 채널로 융합하지 않는다. 합성 BIT 코드가 군집 라벨의
결정론적 함수가 되면 순환성이 재도입되므로, 이 생성기는 다음을 지킨다.

  - 입력은 **관측 데이터(obs)의 lcn·발생일시뿐**이다.
  - `cluster_id`·truth 파일·탐지 결과를 **인자로 받지 않는다**.
  - bit_code는 LCN 문자열에서 결정론적으로 파생되며 군집 라벨과 무관하다.

즉 BIT는 "이미 관측된 고장 근방에서 장비가 자체진단 신호를 남겼다"는 상황만
모사하며, 군집 구조에 대한 추가 정보를 주지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_bit_events(
    obs: pd.DataFrame,
    seed: int = 0,
    coverage: float = 0.6,
    window_hours: float = 12.0,
    false_alarm_ratio: float = 0.2,
    lcn_col: str = "lcn",
    time_col: str = "발생일시",
    system_col: str = "system_id",
) -> pd.DataFrame:
    """
    관측 고장 레코드 근방에 모의 BIT 이벤트를 생성한다.

    Parameters
    ----------
    coverage : 고장 중 BIT가 포착한 비율 (나머지는 BIT 미탐)
    window_hours : 고장 시각 ±window 내에서 BIT 발생 시각을 뽑는다
    false_alarm_ratio : 고장과 무관한 오경보를 전체 대비 비율로 추가

    Returns
    -------
    pd.DataFrame  columns: [system_id, lcn, bit_code, occurred_at]

    ※ cluster_id 등 라벨 정보는 인자로도 결과로도 존재하지 않는다(R3).
    """
    if obs is None or obs.empty or lcn_col not in obs.columns:
        return pd.DataFrame(columns=["system_id", "lcn", "bit_code", "occurred_at"])

    rng = np.random.RandomState(seed)
    ts = pd.to_datetime(obs[time_col], errors="coerce")

    rows = []
    for i, (_, r) in enumerate(obs.iterrows()):
        if rng.rand() > coverage:
            continue  # BIT 미탐
        t = ts.iloc[i]
        if pd.isna(t):
            continue
        오프셋 = rng.uniform(-window_hours, window_hours)
        rows.append({
            "system_id": r.get(system_col),
            "lcn": r[lcn_col],
            "bit_code": _bit_code(str(r[lcn_col])),
            "occurred_at": (t + pd.Timedelta(hours=오프셋)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        })

    # 오경보: 기존 LCN 풀에서 무작위 LCN·무작위 시각
    n_false = int(len(rows) * false_alarm_ratio)
    lcn_pool = sorted(obs[lcn_col].dropna().astype(str).unique())
    if lcn_pool and n_false > 0 and ts.notna().any():
        t_min, t_max = ts.min(), ts.max()
        span_h = max(1.0, (t_max - t_min).total_seconds() / 3600)
        for _ in range(n_false):
            lcn = lcn_pool[rng.randint(len(lcn_pool))]
            rows.append({
                "system_id": None,
                "lcn": lcn,
                "bit_code": _bit_code(lcn),
                "occurred_at": (
                    t_min + pd.Timedelta(hours=float(rng.uniform(0, span_h)))
                ).strftime("%Y-%m-%d %H:%M:%S"),
            })

    return pd.DataFrame(
        rows, columns=["system_id", "lcn", "bit_code", "occurred_at"]
    )


def _bit_code(lcn: str) -> str:
    """
    LCN에서 결정론적으로 파생한 BIT 코드.
    군집 라벨이 아니라 **구조 위치**에만 의존한다(R3).
    """
    key = ".".join(lcn.split(".")[:2])
    번호 = sum(ord(c) for c in key) % 900 + 100
    return f"BIT-{번호}"
