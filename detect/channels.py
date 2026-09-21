"""
3채널 거리행렬 (Phase 2).

  - 의미(semantic): KoSBERT 임베딩 코사인 거리
  - 구조(lcn)     : lcn_tree 상 두 LCN의 트리 경로거리 / (최대깊이×2)
  - 시간(time)    : 1 - exp(-|Δt| / τ)

R2 준수: 이 모듈은 synth/(생성기)를 import하지 않는다. 생성기와 동일한
문자열·임베딩 공간·타임스탬프 값을 공유하지 않으며, 탐지기는 관측 CSV의
텍스트/LCN/시각만 입력으로 받는다.

R7 준수: "AI"라는 표현은 이 KoSBERT/HDBSCAN 파이프라인에만 사용한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

# 기본 KoSBERT 모델명 (설정값으로 교체 가능)
DEFAULT_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "cache" / "embeddings"


# ===========================================================================
# 임베더 인터페이스
# ===========================================================================

class Embedder(Protocol):
    """문장 리스트 → (n, d) 임베딩 행렬."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class KoSBERTEmbedder:
    """
    KoSBERT(sentence-transformers) 임베더. 모델은 최초 사용 시 lazy load.

    모델명은 설정값이며 기본은 snunlp/KR-SBERT-V40K-klueNLI-augSTS.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True),
            dtype=float,
        )


class HashingEmbedder:
    """
    [오프라인 테스트 전용] 결정론적 해시 기반 문자 n-gram 임베더.

    ⚠ 실험 결과 산출에 사용하지 않는다. KoSBERT를 설치할 수 없는 환경에서
    파이프라인 배선(거리행렬·융합·군집화)을 검증하기 위한 더미 임베더다.
    실험 실행 시에는 KoSBERTEmbedder를 사용해야 한다.
    """

    def __init__(self, dim: int = 64, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            s = str(t)
            for j in range(max(1, len(s) - self.ngram + 1)):
                gram = s[j:j + self.ngram]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return out


# ===========================================================================
# 의미 채널
# ===========================================================================

def _cache_key(texts: list[str], model_name: str) -> str:
    h = hashlib.md5()
    h.update(model_name.encode("utf-8"))
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def embed_texts(
    texts: list[str],
    embedder: Embedder | None = None,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> np.ndarray:
    """
    문장 임베딩을 계산한다. 결과는 cache/embeddings/{data_hash}.npy에 캐시된다.
    embedder 미지정 시 KoSBERTEmbedder(model_name)를 사용한다.
    """
    embedder = embedder or KoSBERTEmbedder(model_name)
    name = getattr(embedder, "model_name", embedder.__class__.__name__)

    path = None
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{_cache_key(list(texts), name)}.npy"
        if path.exists():
            return np.load(path)

    emb = embedder.encode(list(texts))
    if path is not None:
        np.save(path, emb)
    return emb


def semantic_distance(
    texts: list[str],
    embedder: Embedder | None = None,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> np.ndarray:
    """
    의미 채널 거리행렬 = 1 - 코사인 유사도, [0, 1] 범위로 clip.
    """
    emb = embed_texts(texts, embedder, model_name, use_cache)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = emb / norm
    sim = unit @ unit.T
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 1.0)


# ===========================================================================
# 구조(LCN) 채널
# ===========================================================================

def _lcn_path(lcn: str) -> list[str]:
    """LCN 문자열을 조상 경로 리스트로 변환 (예: A.01.02 → [A, A.01, A.01.02])."""
    parts = str(lcn).split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def lcn_distance(lcns: list[str], max_depth: int | None = None) -> np.ndarray:
    """
    구조 채널 거리행렬.

    두 LCN의 트리 경로거리(= 각자 최소공통조상까지의 간선 수 합)를
    (최대깊이 × 2)로 나눠 [0, 1] 정규화한다. 동일 LCN은 0.
    max_depth 미지정 시 입력에서 관측된 최대 깊이를 사용한다.
    """
    paths = [_lcn_path(l) for l in lcns]
    depths = [len(p) - 1 for p in paths]
    md = max_depth if max_depth is not None else max(depths + [1])
    denom = max(1, md * 2)

    n = len(lcns)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = paths[i], paths[j]
            공통 = 0
            for a, b in zip(pi, pj):
                if a == b:
                    공통 += 1
                else:
                    break
            거리 = (len(pi) - 공통) + (len(pj) - 공통)
            D[i, j] = D[j, i] = 거리 / denom
    return np.clip(D, 0.0, 1.0)


# ===========================================================================
# 시간 채널
# ===========================================================================

def time_distance(times, tau_days: float = 30.0) -> np.ndarray:
    """
    시간 채널 거리행렬 = 1 - exp(-|Δt| / τ).  τ 기본 30일.
    """
    ts = pd.to_datetime(pd.Series(list(times)), errors="coerce")
    days = ts.astype("int64").to_numpy() / (1e9 * 86400.0)
    dt = np.abs(days[:, None] - days[None, :])
    tau = max(float(tau_days), 1e-9)
    D = 1.0 - np.exp(-dt / tau)
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 1.0)


# ===========================================================================
# 통합 진입점
# ===========================================================================

def build_channels(
    df: pd.DataFrame,
    embedder: Embedder | None = None,
    model_name: str = DEFAULT_MODEL,
    tau_days: float = 30.0,
    text_col: str = "고장증상",
    lcn_col: str = "lcn",
    time_col: str = "발생일시",
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    """
    관측 DataFrame에서 3채널 거리행렬을 만든다.

    Returns
    -------
    {"sem": D_sem, "lcn": D_lcn, "time": D_time}
    """
    return {
        "sem": semantic_distance(
            list(df[text_col]), embedder, model_name, use_cache
        ),
        "lcn": lcn_distance(list(df[lcn_col])),
        "time": time_distance(list(df[time_col]), tau_days),
    }
