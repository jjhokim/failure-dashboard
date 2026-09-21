"""
라벨 누출 검사 3종 (Phase 1). 생성 후 자동 실행되며 결과를 dict로 반환한다.

절대 규칙 R2: 검사 2(문장→cluster_id 예측)에 KoSBERT를 쓰지 않는다.
여기서는 TF-IDF(문자 n-gram) + 로지스틱 회귀만 사용한다.

검사 목적: 정답 라벨이 관측 텍스트로 결정론적으로 흘러들어가 순환검증이
발생하지 않았는지 점검한다. 무주입(delta=0) 대조군 대비 상승 여부로 해석한다.
"""

from __future__ import annotations

import itertools

import numpy as np


def token_leakage_rate(narratives: list[str]) -> float:
    """검사 1: 군집명·라벨 토큰의 문장 내 출현률 (0이어야 정상)."""
    금지토큰 = ["군집", "cluster", "클러스터", "라벨", "label"]
    if not narratives:
        return 0.0
    hits = sum(
        1 for s in narratives if any(t in s.lower() for t in 금지토큰)
    )
    return hits / len(narratives)


def tfidf_logreg_accuracy(
    narratives: list[str], cluster_ids: list[int]
) -> dict | None:
    """
    검사 2: TF-IDF 로지스틱 회귀로 문장→cluster_id 예측 정확도(5-fold).

    무주입(delta=0) 대비 유의 상승 여부 해석용. chance(=1/군집수)와 함께 반환.
    노이즈(-1)는 제외. KoSBERT 미사용(R2).
    """
    from collections import Counter

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    X = [n for n, c in zip(narratives, cluster_ids) if c != -1]
    y = [c for c in cluster_ids if c != -1]
    if len(set(y)) < 2:
        return None

    min_class = min(Counter(y).values())
    if min_class < 2:
        return None
    cv = max(2, min(5, min_class))

    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
        LogisticRegression(max_iter=1000),
    )
    scores = cross_val_score(pipe, X, y, cv=cv)
    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "chance": 1.0 / len(set(y)),
        "cv": cv,
    }


def cluster_vocab_jaccard(
    narratives: list[str], cluster_ids: list[int]
) -> dict | None:
    """검사 3: 군집 간 어휘 Jaccard 중복도 (평균/최대)."""
    from collections import defaultdict

    groups: dict[int, set] = defaultdict(set)
    for s, c in zip(narratives, cluster_ids):
        if c == -1:
            continue
        for tok in s.replace(".", " ").split():
            groups[c].add(tok)

    keys = list(groups)
    if len(keys) < 2:
        return None

    js = []
    for a, b in itertools.combinations(keys, 2):
        inter = len(groups[a] & groups[b])
        uni = len(groups[a] | groups[b])
        js.append(inter / uni if uni else 0.0)
    return {"mean_jaccard": float(np.mean(js)), "max_jaccard": float(np.max(js))}


def run_leakage_checks(
    narratives: list[str], cluster_ids: list[int]
) -> dict:
    """3종 검사를 모두 실행해 요약 dict를 반환한다."""
    return {
        "token_leakage_rate": token_leakage_rate(narratives),
        "tfidf_logreg": tfidf_logreg_accuracy(narratives, cluster_ids),
        "vocab_jaccard": cluster_vocab_jaccard(narratives, cluster_ids),
    }
