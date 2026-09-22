"""
논문·보고서용 그림·표 추출 (후보작업 1).

    python -m experiments.export_figures

입력: results/{run_id}/summary.csv, raw.csv, results/v2/*
출력: paper_assets/
    fig1_delta_sweep.png        δ별 ARI (주 결과 = legacy)
    fig2_channel_ablation.png   채널 조합별 ARI
    fig3_legacy_vs_floor.png    주 결과 vs 후속 민감도 분석
    fig4_c6_variants.png        치명도 3변형 순위 정확도
    tableN_*.md / .tex          동일 내용의 마크다운·LaTeX 표

서술 방침(사용자 합의): **legacy가 주 결과**, floor는 사전등록된 후속 민감도
분석으로 병기한다. 그림에도 이 구분을 명시한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
RESULTS = _ROOT / "results"
OUT = _ROOT / "paper_assets"

# 한글 렌더링 (Windows 기본 폰트)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

C_MAIN, C_SUB, C_ACC = "#2c5f8d", "#c0762a", "#6b8e23"


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  저장: {p.relative_to(_ROOT)}")
    return p


def _write_table(df: pd.DataFrame, stem: str, caption: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.md").write_text(
        f"**{caption}**\n\n{df.to_markdown(index=False)}\n", encoding="utf-8"
    )
    try:
        (OUT / f"{stem}.tex").write_text(
            df.to_latex(index=False, caption=caption, escape=False),
            encoding="utf-8",
        )
    except Exception:  # to_latex는 jinja2 의존이 있을 수 있음
        pass
    print(f"  저장: paper_assets/{stem}.md")


# ===========================================================================
# 그림 1 — δ 스윕
# ===========================================================================

def fig_delta_sweep(summary: pd.DataFrame) -> None:
    d = summary[summary["조건"] == "delta_sweep"].sort_values("delta")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.errorbar(d["delta"], d["ARI_mean"], yerr=d.get("ARI_std"),
                marker="o", color=C_MAIN, capsize=4, linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("주입 강도 δ")
    ax.set_ylabel("ARI")
    ax.set_title("주입 강도별 공통원인 회수 성능 (8시드 mean±std)")
    ax.grid(alpha=0.3)
    _save(fig, "fig1_delta_sweep.png")

    t = d[["delta", "ARI_mean", "ARI_std", "noise_ratio_mean"]].copy()
    t.columns = ["δ", "ARI 평균", "ARI 표준편차", "노이즈 비율"]
    _write_table(t.round(3), "table1_delta_sweep",
                 "표 1. 주입 강도별 ARI (주 결과, legacy 생성기, 8시드)")


# ===========================================================================
# 그림 2 — 채널 ablation
# ===========================================================================

def fig_channel_ablation(summary: pd.DataFrame) -> None:
    a = summary[summary["조건"] == "channel_ablation"].sort_values("ARI_mean")
    if a.empty:
        return
    색 = [C_ACC if ch == "sem+lcn+time" else C_MAIN for ch in a["channels"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(a["channels"], a["ARI_mean"], xerr=a.get("ARI_std"),
            color=색, capsize=3)
    ax.set_xlabel("ARI")
    ax.set_title("채널 조합별 성능 (δ=1.0, 8시드)\n녹색 = 3채널 융합")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "fig2_channel_ablation.png")

    t = a.sort_values("ARI_mean", ascending=False)[
        ["channels", "ARI_mean", "ARI_std"]].copy()
    t.columns = ["채널 조합", "ARI 평균", "ARI 표준편차"]
    _write_table(t.round(3), "table2_channel_ablation",
                 "표 2. 채널 조합별 ARI (주 결과, δ=1.0, 8시드). "
                 "융합이 시간 단독에 미달 — C4 불충족")


# ===========================================================================
# 그림 3 — legacy(주) vs floor(후속)
# ===========================================================================

def fig_legacy_vs_floor() -> None:
    vpath = RESULTS / "v2" / "verdicts.json"
    if not vpath.exists():
        return
    v = json.loads(vpath.read_text(encoding="utf-8"))
    b = v.get("armB")
    if not b:
        return

    # 주 결과(legacy)는 1차 summary에서
    legacy_fused = legacy_time = None
    for run in RESULTS.glob("*"):
        p = run / "summary.csv"
        if p.exists():
            s = pd.read_csv(p, encoding="utf-8-sig")
            a = s[s["조건"] == "channel_ablation"]
            if not a.empty:
                legacy_fused = float(
                    a[a["channels"] == "sem+lcn+time"]["ARI_mean"].iloc[0])
                legacy_time = float(a[a["channels"] == "time"]["ARI_mean"].iloc[0])
                break
    if legacy_fused is None:
        return

    라벨 = ["3채널 융합", "시간 채널 단독"]
    legacy = [legacy_fused, legacy_time]
    floor = [b["fused_mean"], b["time_only_mean"]]

    x = range(len(라벨))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar([i - 0.2 for i in x], legacy, width=0.4,
           label="주 결과 (legacy, 밀집폭 1일)", color=C_MAIN)
    ax.bar([i + 0.2 for i in x], floor, width=0.4,
           label="후속 민감도 (floor, 밀집폭 3.65일)", color=C_SUB)
    ax.set_xticks(list(x))
    ax.set_xticklabels(라벨)
    ax.set_ylabel("ARI")
    ax.set_title("시간 밀집폭 가정에 따른 융합 이점의 역전\n(δ=1.0)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig3_legacy_vs_floor.png")

    t = pd.DataFrame({
        "생성기 가정": ["legacy (밀집폭 1일, 주 결과)",
                       "floor (밀집폭 3.65일, 후속)"],
        "3채널 융합": [round(legacy_fused, 3), round(b["fused_mean"], 3)],
        "시간 단독": [round(legacy_time, 3), round(b["time_only_mean"], 3)],
        "융합 우위": ["아니오 (C4 불충족)", "예 (CI [%.3f, %.3f])" % (
            b["diff_ci"]["lo"], b["diff_ci"]["hi"])],
    })
    _write_table(t, "table3_legacy_vs_floor",
                 "표 3. 시간 밀집폭 가정에 따른 결론 변화. "
                 "융합 이점은 신호의 시간 분산에 조건부다.")


# ===========================================================================
# 그림 4 — C6 치명도 변형
# ===========================================================================

def fig_c6_variants() -> None:
    p = RESULTS / "v2" / "c6_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p, encoding="utf-8-sig")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.bar(s["variant"], s["spearman_mean"],
            yerr=s.get("spearman_std"), color=C_MAIN, capsize=4)
    ax1.set_ylabel("Spearman ρ")
    ax1.set_title("치명도 산식별 순위 정확도")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(s["variant"], s["corr_size_crit_mean"], color=C_SUB)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_ylabel("corr(규모, 치명도)")
    ax2.set_title("규모-치명도 상관")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("상관을 제거(S1)하면 순위가 오히려 나빠진다 → 상관은 신호", fontsize=9)
    _save(fig, "fig4_c6_variants.png")

    t = s.copy()
    t.columns = ["변형", "ρ 평균", "ρ 표준편차", "P@5", "corr(규모,치명도)"]
    _write_table(t.round(3), "table4_c6_variants",
                 "표 4. 치명도 산식 3변형 비교 (평가시드 100~107). "
                 "S0 = 현행 유지")


def main() -> None:
    print("[그림·표 추출]")
    summary = None
    for run in sorted(RESULTS.glob("*")):
        p = run / "summary.csv"
        if p.exists():
            summary = pd.read_csv(p, encoding="utf-8-sig")
            print(f"  주 결과: {run.name}")
            break

    if summary is not None:
        fig_delta_sweep(summary)
        fig_channel_ablation(summary)
    else:
        print("  [경고] 1차 실험 summary.csv 없음 — 그림 1·2 생략")

    fig_legacy_vs_floor()
    fig_c6_variants()
    print(f"[완료] {OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
