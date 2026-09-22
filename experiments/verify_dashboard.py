"""
대시보드 동작 검증 (수정 사항 반영 확인).

    python -m experiments.verify_dashboard

캡처만 하는 capture_dashboard.py와 달리, 이 스크립트는 **검증**이 목적이다.
  - 각 메뉴를 실제 클릭하고 화면에 나타난 텍스트를 확인한다
  - 버튼이 있는 화면(공통원인 후보)은 버튼까지 눌러 결과를 확인한다
  - Streamlit 예외 박스·Traceback 문구가 있으면 실패로 처리한다
  - 기대 문구가 보이지 않으면 실패로 처리한다

종료 코드 0이면 전부 통과, 1이면 실패 항목이 있다.
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

# 메뉴 라벨에 이모지가 있어 cp949 콘솔에서 출력이 깨진다. UTF-8로 고정한다.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
SHOTS = _ROOT / "paper_assets" / "verify"
PORT = 8631
URL = f"http://localhost:{PORT}"

# 화면 오류 징후로 간주할 문구
ERROR_MARKERS = ["Traceback", "KeyError", "AttributeError", "TypeError",
                 "ValueError", "ModuleNotFound", "탐지 실패", "파일 읽기 실패"]

# (메뉴 라벨, 이 화면에 반드시 보여야 할 문구들, 캡처 파일명)
CHECKS = [
    ("📊 요약 대시보드",
     ["전체 현황 요약", "고유가용도 Ai", "MTTR", "검증용"], "v1_summary.png"),
    ("🔍 분석",
     ["파레토 분석", "LCN 상위 노드별 파레토"], "v2_analysis.png"),
    ("🧩 공통원인 후보",
     ["공통원인 후보 탐지", "의미 가중치", "탐지 실행"], "v3_common.png"),
    ("🧪 실험 결과",
     ["실험 결과"], "v4_experiments.png"),
    ("🔗 BIT 연결",
     ["BIT", "시각 윈도우", "연결된 쌍"], "v5_bit.png"),
    ("✏️ 고장 입력",
     ["고장이력 신규 등록", "제대구분"], "v6_input.png"),
    ("📋 고장 현황",
     ["고장이력 조회", "조회 필터"], "v7_status.png"),
]


def _start():
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=str(_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def verify() -> bool:
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    proc = _start()
    실패: list[str] = []

    try:
        time.sleep(18)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(URL, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(6000)

            for 라벨, 기대문구들, 파일 in CHECKS:
                print(f"\n▶ {라벨}")
                try:
                    page.get_by_text(라벨, exact=True).first.click(timeout=15_000)
                    page.wait_for_timeout(4500)
                except Exception as e:  # noqa: BLE001
                    실패.append(f"{라벨}: 메뉴 클릭 실패 ({type(e).__name__})")
                    print("  ✗ 메뉴 클릭 실패")
                    continue

                본문 = page.inner_text("body")

                # 오류 징후
                오류 = [m for m in ERROR_MARKERS if m in 본문]
                if 오류:
                    실패.append(f"{라벨}: 오류 문구 {오류}")
                    print(f"  ✗ 오류 문구 발견: {오류}")
                else:
                    print("  ✓ 오류 문구 없음")

                # 기대 문구
                누락 = [t for t in 기대문구들 if t not in 본문]
                if 누락:
                    실패.append(f"{라벨}: 기대 문구 누락 {누락}")
                    print(f"  ✗ 기대 문구 누락: {누락}")
                else:
                    print(f"  ✓ 기대 문구 {len(기대문구들)}종 확인")

                page.screenshot(path=str(SHOTS / 파일), full_page=True)

                # 공통원인 후보는 버튼까지 눌러 결과 확인
                if "공통원인" in 라벨:
                    print("  · [탐지 실행] 클릭")
                    try:
                        page.get_by_role("button", name="탐지 실행").click(timeout=15_000)
                        page.wait_for_timeout(45_000)  # 임베딩·군집화 대기
                        본문2 = page.inner_text("body")
                        오류2 = [m for m in ERROR_MARKERS if m in 본문2]
                        if 오류2:
                            실패.append(f"{라벨}(탐지실행): 오류 {오류2}")
                            print(f"  ✗ 탐지 실행 오류: {오류2}")
                        elif "우선순위 순위표" in 본문2 and "발견 군집" in 본문2:
                            print("  ✓ 탐지 결과 렌더링 확인 (순위표·군집 수)")
                        else:
                            실패.append(f"{라벨}(탐지실행): 결과 미표시")
                            print("  ✗ 탐지 결과가 표시되지 않음")
                        page.screenshot(path=str(SHOTS / "v3b_common_result.png"),
                                        full_page=True)
                    except Exception as e:  # noqa: BLE001
                        실패.append(f"{라벨}(탐지실행): {type(e).__name__}")
                        print(f"  ✗ 탐지 실행 실패: {type(e).__name__}")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    print("\n" + "=" * 60)
    if 실패:
        print(f"실패 {len(실패)}건")
        for f in 실패:
            print(f"  - {f}")
        return False
    print(f"전체 통과 — {len(CHECKS)}개 화면 + 탐지 실행")
    return True


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
