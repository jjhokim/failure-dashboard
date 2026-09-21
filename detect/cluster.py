"""
HDBSCAN 군집화 래퍼 (Phase 2).

precomputed 거리행렬 기반. min_cluster_size 탐색 범위는 [5, 8, 12]로 사전 고정한다
(실행 전 고정 — R5 사전등록 정신에 맞춤).
라벨 -1은 노이즈(단발성 고장)를 의미한다.

구현 메모: sklearn.cluster.HDBSCAN(metric='precomputed')을 사용한다.
별도 hdbscan 패키지와 동일 알고리즘이며 추가 빌드 의존성이 없다.
"""

from __future__ import annotations

import numpy as np

# 사전 고정된 min_cluster_size 탐색 범위
MIN_CLUSTER_SIZES: list[int] = [5, 8, 12]

NOISE_LABEL = -1


def cluster(D: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """
    융합 거리행렬을 HDBSCAN으로 군집화해 라벨 배열을 반환한다.

    Parameters
    ----------
    D : (n, n) precomputed 거리행렬
    min_cluster_size : 최소 군집 크기

    Returns
    -------
    np.ndarray  라벨 (노이즈는 -1)
    """
    from sklearn.cluster import HDBSCAN

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n < max(2, min_cluster_size):
        return np.full(n, NOISE_LABEL, dtype=int)

    # 대칭·비음수 보정 (수치 오차로 HDBSCAN이 거부하는 경우 방지)
    D = np.clip((D + D.T) / 2.0, 0.0, None)
    np.fill_diagonal(D, 0.0)

    model = HDBSCAN(metric="precomputed", min_cluster_size=int(min_cluster_size))
    return model.fit_predict(D).astype(int)


# 패키지 레벨 재노출용 별칭.
# detect/__init__.py가 cluster(함수)를 재노출하면 detect.cluster(모듈)를
# 가리므로, 패키지에서는 이 별칭으로 노출한다.
cluster_labels = cluster


def cluster_stats(labels: np.ndarray) -> dict:
    """군집 결과 요약: 군집 수, 노이즈 비율, 군집별 크기."""
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0:
        return {"n_clusters": 0, "noise_ratio": 0.0, "sizes": {}}

    유효 = labels[labels != NOISE_LABEL]
    uniq, counts = np.unique(유효, return_counts=True)
    return {
        "n_clusters": int(len(uniq)),
        "noise_ratio": float((labels == NOISE_LABEL).sum() / n),
        "sizes": {int(k): int(v) for k, v in zip(uniq, counts)},
    }


def cluster_sweep(
    D: np.ndarray, sizes: list[int] | None = None
) -> dict[int, np.ndarray]:
    """
    사전 고정된 min_cluster_size 범위를 순회하며 군집 라벨을 산출한다.

    Returns
    -------
    {min_cluster_size: labels}
    """
    sizes = MIN_CLUSTER_SIZES if sizes is None else sizes
    return {m: cluster(D, m) for m in sizes}
