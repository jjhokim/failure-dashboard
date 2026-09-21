"""
베이스라인 2종 (Phase 2).

  1. TF-IDF 코사인 거리 → 동일 HDBSCAN
  2. 키워드·동의어 사전 규칙 그룹핑

R2 준수 (중요): 아래 정비_키워드 사전은 이 모듈에서 독립적으로 작성한 것이며,
synth/narrator.py의 동의어 사전을 import하거나 복사하지 않는다. 생성기와 탐지기가
동일 사전을 공유하면 규칙 베이스라인이 생성 규칙을 그대로 되읽는 순환이 된다.
(도메인이 같으므로 일부 어휘가 겹칠 수는 있으나, 사전의 구성·분류 체계는 독립이다.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from detect.cluster import NOISE_LABEL, cluster

# ---------------------------------------------------------------------------
# 규칙 베이스라인용 정비 도메인 키워드 사전 (탐지기 측 독립 작성 — R2)
# 분류 체계: 고장 메커니즘 중심 (생성기의 '증상코드→표현' 체계와 독립)
# ---------------------------------------------------------------------------

정비_키워드: dict[str, list[str]] = {
    "열관련":   ["과열", "발열", "온도", "고온", "열"],
    "기계진동": ["진동", "떨림", "흔들", "소음", "잡음", "음"],
    "유체누설": ["누유", "누설", "샘", "오일", "유체", "기름"],
    "전원계통": ["전압", "전원", "출력", "강하", "저하", "배터리"],
    "통신계통": ["통신", "신호", "링크", "교신", "두절", "끊김", "데이터"],
    "구조손상": ["균열", "크랙", "파손", "마모", "손상"],
    "제어이상": ["오작동", "제어", "기능", "비정상", "동작", "이상"],
}


# ===========================================================================
# 베이스라인 1: TF-IDF 코사인 거리 + HDBSCAN
# ===========================================================================

def tfidf_distance(texts: list[str]) -> np.ndarray:
    """TF-IDF(문자 n-gram) 코사인 거리행렬."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vec.fit_transform([str(t) for t in texts])
    X = np.asarray(X.todense(), dtype=float)

    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = X / norm
    D = 1.0 - (unit @ unit.T)
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 1.0)


def baseline_tfidf(texts: list[str], min_cluster_size: int = 5) -> np.ndarray:
    """베이스라인 1: TF-IDF 거리 → HDBSCAN 라벨."""
    return cluster(tfidf_distance(texts), min_cluster_size)


# ===========================================================================
# 베이스라인 2: 키워드·동의어 규칙 그룹핑
# ===========================================================================

def _rule_category(text: str) -> str | None:
    """문장에서 가장 많이 매칭된 키워드 범주를 반환. 매칭 없으면 None."""
    s = str(text)
    best, best_hits = None, 0
    for 범주, 키워드들 in 정비_키워드.items():
        hits = sum(1 for kw in 키워드들 if kw in s)
        if hits > best_hits:
            best, best_hits = 범주, hits
    return best


def baseline_rules(
    texts: list[str], min_group_size: int = 5
) -> np.ndarray:
    """
    베이스라인 2: 키워드·동의어 사전 규칙 그룹핑.

    범주 매칭이 없거나 그룹 크기가 min_group_size 미만이면 노이즈(-1).
    """
    범주들 = [_rule_category(t) for t in texts]
    counts = pd.Series([c for c in 범주들 if c]).value_counts()
    유효 = {c for c, n in counts.items() if n >= min_group_size}

    코드: dict[str, int] = {c: i for i, c in enumerate(sorted(유효))}
    return np.array(
        [코드.get(c, NOISE_LABEL) if c else NOISE_LABEL for c in 범주들],
        dtype=int,
    )
