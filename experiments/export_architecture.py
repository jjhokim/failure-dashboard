"""
파이프라인 아키텍처 그림 생성 (논문 그림 0).

    python -m experiments.export_architecture

핵심: 생성기 → 관측데이터 → 탐지기 → 우선순위화의 흐름과 함께,
**순환검증을 막는 차단선 R1·R2·R3**를 명시적으로 표시한다.
정답(truth)이 관측 경로로 새지 않는다는 것이 이 그림의 메시지다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "paper_assets"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

C_GEN = "#dce8f5"     # 생성기
C_OBS = "#e8f0dc"     # 관측 데이터
C_DET = "#f5e6d0"     # 탐지기
C_TRUTH = "#f5dcdc"   # 정답(truth)
C_LINE = "#333333"
C_BLOCK = "#c0392b"   # 차단선


def _box(ax, x, y, w, h, text, color, fontsize=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        facecolor=color, edgecolor=C_LINE, linewidth=1.0,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            linespacing=1.4)


def _arrow(ax, x1, y1, x2, y2, style="-|>", color=C_LINE, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
        color=color, linewidth=lw, linestyle=ls, shrinkA=2, shrinkB=2,
    ))


def _block_badge(ax, x, y, label, direction="right"):
    """
    차단 배지: 짧은 점선 + X + 라벨.

    긴 화살표로 그리면 다른 박스·경로와 교차해 가독성이 무너지므로,
    차단 지점에 짧게 표시하고 라벨을 바깥쪽에 둔다.
    """
    d = 0.045
    if direction == "right":
        x2, y2, lx, ly, ha, va = x + d, y, x + d + 0.012, y, "left", "center"
    elif direction == "down":
        x2, y2, lx, ly, ha, va = x, y - d, x, y - d - 0.022, "center", "top"
    else:  # left
        x2, y2, lx, ly, ha, va = x - d, y, x - d - 0.012, y, "right", "center"

    ax.plot([x, x2], [y, y2], color=C_BLOCK, linewidth=1.3,
            linestyle=(0, (3, 2.5)))
    mx, my = (x + x2) / 2, (y + y2) / 2
    s = 0.016
    ax.plot([mx - s, mx + s], [my - s * 1.6, my + s * 1.6],
            color=C_BLOCK, linewidth=2.4)
    ax.plot([mx - s, mx + s], [my + s * 1.6, my - s * 1.6],
            color=C_BLOCK, linewidth=2.4)
    ax.text(lx, ly, label, fontsize=8, color=C_BLOCK,
            fontweight="bold", ha=ha, va=va)


def build() -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── 생성기 (좌측) ─────────────────────────────────────────
    ax.text(0.145, 0.965, "합성 생성기 (synth/)", fontsize=10,
            fontweight="bold", ha="center")
    _box(ax, 0.03, 0.845, 0.23, 0.085,
         "군집 라벨 c\n(정답 — 생성에만 사용)", C_TRUTH, bold=True)
    _box(ax, 0.03, 0.705, 0.23, 0.095,
         "physical.py  [규칙 기반]\n라벨 → 물리상태\n(LCN·증상·환경·시각)", C_GEN)
    _box(ax, 0.03, 0.565, 0.23, 0.095,
         "narrator.py  [템플릿/LLM]\n물리상태 → 자유서술\ncluster_id 인자 없음", C_GEN)

    _arrow(ax, 0.145, 0.845, 0.145, 0.802)
    _arrow(ax, 0.145, 0.705, 0.145, 0.662)

    # ── 관측 데이터 (중앙) ────────────────────────────────────
    _box(ax, 0.375, 0.60, 0.25, 0.135,
         "관측 데이터\n발생일시 · 체계 · LRU\nLCN · 자유서술 · effect_class",
         C_OBS, fontsize=9, bold=True)
    _arrow(ax, 0.26, 0.612, 0.375, 0.660)

    # 정답 파일 (별도 보관)
    _box(ax, 0.375, 0.845, 0.25, 0.085,
         "정답 파일 truth.csv\ncluster_id · β · α · λ · t*", C_TRUTH, bold=True)
    _arrow(ax, 0.145, 0.888, 0.375, 0.888, ls=(0, (2, 2)), lw=1.0)

    # BIT
    _box(ax, 0.375, 0.435, 0.25, 0.085,
         "모의 BIT 로그\n(관측 LCN·시각에서만 생성)", C_OBS, fontsize=8.5)
    _arrow(ax, 0.50, 0.60, 0.50, 0.522)

    # ── 탐지기 (우측) ─────────────────────────────────────────
    ax.text(0.855, 0.965, "탐지 · 우선순위화", fontsize=10,
            fontweight="bold", ha="center")
    _box(ax, 0.74, 0.755, 0.23, 0.12,
         "channels.py  3채널 거리\n의미(KoSBERT) · 구조(LCN)\n· 시간(exp 감쇠)", C_DET)
    _box(ax, 0.74, 0.625, 0.23, 0.085,
         "fusion.py\nmin-max 정규화 + 가중합", C_DET)
    _box(ax, 0.74, 0.495, 0.23, 0.085,
         "cluster.py\nHDBSCAN (precomputed)", C_DET)
    _box(ax, 0.74, 0.345, 0.23, 0.105,
         "prioritize/scoring.py\n규모 · 추세 · 치명도\n→ 순위표", C_DET)

    _arrow(ax, 0.625, 0.680, 0.74, 0.800)
    _arrow(ax, 0.855, 0.755, 0.855, 0.712)
    _arrow(ax, 0.855, 0.625, 0.855, 0.582)
    _arrow(ax, 0.855, 0.495, 0.855, 0.452)

    # ── 차단 배지 R1 · R2 · R3 · 정답 ─────────────────────────
    # R1: 라벨 → narrator 직결 차단 (라벨은 physical(규칙)만 거친다)
    _block_badge(ax, 0.145, 0.560, "R1  라벨→문장 직결", direction="down")
    # R2: 생성기 → 탐지기로 문자열·임베딩 공간·사전이 흐르는 경로 차단
    # (⇎ 등 일부 기호는 Malgun Gothic에 글리프가 없어 두부(□)로 렌더링된다)
    _block_badge(ax, 0.275, 0.800, "R2  생성기·탐지기 분리", direction="right")
    # R3: BIT → 탐지 채널 융합 차단 (조인 전용)
    _block_badge(ax, 0.638, 0.478, "R3", direction="right")
    # 정답이 탐지기로 가는 경로 차단
    _block_badge(ax, 0.638, 0.888, "정답 차단", direction="right")

    # ── 평가 경로 (정답은 평가에서만 사용) ─────────────────────
    _box(ax, 0.375, 0.16, 0.25, 0.10,
         "평가 (experiments/)\nARI · Spearman ρ · P@5\nDetection Delay", "#eeeeee",
         fontsize=8.5, bold=True)
    _arrow(ax, 0.50, 0.845, 0.50, 0.262, ls=(0, (2, 2)), lw=1.1)
    ax.text(0.515, 0.33, "정답은 평가에만", fontsize=7.8, color="#666666")
    _arrow(ax, 0.855, 0.345, 0.855, 0.21)
    _arrow(ax, 0.74, 0.21, 0.625, 0.21)

    # ── 범례 ──────────────────────────────────────────────────
    ax.plot([0.03, 0.075], [0.075, 0.075], color=C_BLOCK,
            linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(0.085, 0.075, "차단된 경로 (순환검증 방지)", fontsize=8,
            va="center", color=C_BLOCK)
    ax.text(0.40, 0.075,
            "R1 라벨→문장 직결 차단   R2 생성기·탐지기 공간 분리   "
            "R3 BIT는 조인 전용",
            fontsize=8, va="center", color=C_BLOCK)

    ax.set_title("공통원인 스크리닝 파이프라인과 순환검증 차단 설계",
                 fontsize=12, fontweight="bold", pad=14)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig0_architecture.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"저장: {p.relative_to(_ROOT)}")
    return p


if __name__ == "__main__":
    build()
