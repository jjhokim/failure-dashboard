"""
대시보드 스크린샷 캡처 (논문 그림 6).

    python -m experiments.capture_dashboard

Streamlit 앱을 헤드리스로 띄우고 playwright로 각 메뉴를 클릭해 캡처한다.
사이드바 라디오 항목을 실제로 클릭하므로, 앱이 정상 동작하지 않으면 실패한다
(= 캡처 성공 자체가 동작 검증을 겸한다).

전제: 로컬 DB에 lcn이 있는 데이터가 있어야 신규 메뉴가 내용을 보여준다.
      없으면 안내문만 캡처된다.

주의: 개발 전용 의존성(playwright)이며 requirements.txt에는 넣지 않는다.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "paper_assets" / "screenshots"
PORT = 8613
URL = f"http://localhost:{PORT}"

# (파일명, 사이드바 메뉴 라벨) — 라벨은 app.py의 st.radio 항목과 일치해야 한다
TARGETS = [
    ("dash1_summary.png", "📊 요약 대시보드"),
    ("dash2_analysis.png", "🔍 분석"),
    ("dash3_common_cause.png", "🧩 공통원인 후보"),
    ("dash4_experiments.png", "🧪 실험 결과"),
    ("dash5_bit.png", "🔗 BIT 연결"),
]


def _start_app() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=str(_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def capture() -> list[Path]:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    proc = _start_app()
    saved: list[Path] = []
    try:
        time.sleep(18)  # 앱 기동 대기
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 950})
            page.goto(URL, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(6000)

            for 파일명, 라벨 in TARGETS:
                try:
                    # 사이드바 라디오 항목 클릭
                    item = page.get_by_text(라벨, exact=True).first
                    item.click(timeout=15_000)
                    page.wait_for_timeout(5000)
                except Exception as e:  # noqa: BLE001
                    print(f"  [경고] '{라벨}' 클릭 실패: {type(e).__name__}")
                path = OUT / 파일명
                page.screenshot(path=str(path), full_page=True)
                saved.append(path)
                print(f"  저장: {path.relative_to(_ROOT)}")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
    return saved


if __name__ == "__main__":
    print("[대시보드 캡처]")
    files = capture()
    print(f"[완료] {len(files)}장 · {OUT.relative_to(_ROOT)}")
