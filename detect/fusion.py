"""
채널 융합 (Phase 2).

채널별 min-max 정규화 후 가중합:
    D = w_s·D_sem + w_l·D_lcn + w_t·D_time,   Σw = 1

채널 on/off는 가중치 0으로 표현하므로 ablation(채널 조합 실험)에 그대로 쓰인다.
"""

from __future__ import annotations

import numpy as np

# 가중치 기본값 (의미, 구조, 시간)
DEFAULT_WEIGHTS: dict[str, float] = {"sem": 0.5, "lcn": 0.25, "time": 0.25}


def minmax_normalize(D: np.ndarray) -> np.ndarray:
    """
    거리행렬을 [0, 1]로 min-max 정규화한다.
    비대각 원소 기준으로 min/max를 잡고, 상수 행렬이면 0 행렬을 반환한다.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n < 2:
        return np.zeros_like(D)

    off = ~np.eye(n, dtype=bool)
    vals = D[off]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(D)

    out = (D - lo) / (hi - lo)
    np.fill_diagonal(out, 0.0)
    return np.clip(out, 0.0, 1.0)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """가중치 합이 1이 되도록 정규화한다. 합이 0이면 ValueError."""
    total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if total <= 0:
        raise ValueError("가중치 합이 0입니다. 최소 한 채널은 활성화해야 합니다.")
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def fuse(
    channels: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    채널별 min-max 정규화 후 가중합으로 융합 거리행렬을 만든다.

    Parameters
    ----------
    channels : {"sem": D, "lcn": D, "time": D}
    weights  : 채널별 가중치. 미지정 시 DEFAULT_WEIGHTS.
               가중치 0인 채널은 계산에서 제외된다(ablation의 채널 off).

    Returns
    -------
    np.ndarray  융합 거리행렬 (대각 0, [0,1])
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    # channels에 없는 키는 무시
    weights = {k: v for k, v in weights.items() if k in channels}
    w = normalize_weights(weights)

    D_out = None
    for name, weight in w.items():
        if weight <= 0:
            continue  # 채널 off
        Dn = minmax_normalize(channels[name]) * weight
        D_out = Dn if D_out is None else D_out + Dn

    if D_out is None:
        raise ValueError("활성 채널이 없습니다.")

    np.fill_diagonal(D_out, 0.0)
    return np.clip(D_out, 0.0, 1.0)
