"""공통원인 후보 탐지 파이프라인 (Phase 2).

R2: 이 패키지는 synth/(생성기)를 import하지 않는다.
R7: "AI"라는 표현은 이 KoSBERT/HDBSCAN 파이프라인에만 사용한다.
"""

from detect.baselines import baseline_rules, baseline_tfidf, tfidf_distance
from detect.channels import build_channels, lcn_distance, semantic_distance, time_distance
from detect.cluster import (
    MIN_CLUSTER_SIZES,
    NOISE_LABEL,
    cluster_labels,
    cluster_stats,
    cluster_sweep,
)
from detect.fusion import DEFAULT_WEIGHTS, fuse, minmax_normalize

# 주의: 군집화 함수는 cluster_labels로 재노출한다.
# 원래 이름 cluster를 여기서 재노출하면 detect.cluster '모듈'을 가려버려
# `import detect.cluster`가 함수로 해석되는 문제가 생긴다.
# 모듈이 필요하면 detect.cluster, 함수가 필요하면 detect.cluster.cluster를 쓴다.

__all__ = [
    "build_channels", "semantic_distance", "lcn_distance", "time_distance",
    "fuse", "minmax_normalize", "DEFAULT_WEIGHTS",
    "cluster_labels", "cluster_stats", "cluster_sweep",
    "MIN_CLUSTER_SIZES", "NOISE_LABEL",
    "baseline_tfidf", "baseline_rules", "tfidf_distance",
]
