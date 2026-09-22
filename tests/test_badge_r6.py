"""
R6 배지 동적 전환 검증.

Phase 0-1에서 "sources 도입 후 데이터 출처에 따라 자동 전환"하기로 했던 것을
실제로 구현했는지 확인한다. 배지가 실제 데이터 성격과 어긋나면 R6 위반이다.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("app", _ROOT / "frontend" / "app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _df(sources: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"source": sources})


def test_단일출처_샘플(app):
    assert app._배지_문구(_df(["sample"] * 10)) == "샘플데이터 기준 · 검증용"


def test_단일출처_합성(app):
    assert app._배지_문구(_df(["synthetic"] * 10)) == "합성데이터 기준 · 검증용"


def test_단일출처_외부(app):
    assert app._배지_문구(_df(["import"] * 5)) == "외부데이터 기준 · 검증용"


def test_혼합이면_비중_표기(app):
    s = app._배지_문구(_df(["synthetic"] * 75 + ["sample"] * 25))
    assert "혼합" in s
    assert "합성데이터 75%" in s
    assert "검증용" in s


def test_빈데이터(app):
    assert app._배지_문구(pd.DataFrame()) == "데이터 없음 · 검증용"


def test_source컬럼_없으면_안전하게_처리(app):
    assert "검증용" in app._배지_문구(pd.DataFrame({"x": [1, 2]}))


def test_모든_배지에_검증용_표기(app):
    """R6: 어떤 데이터 구성이든 '검증용'이 빠지면 안 된다."""
    for sources in (["sample"], ["synthetic"], ["import"],
                    ["sample", "synthetic"], ["synthetic", "import", "sample"]):
        assert "검증용" in app._배지_문구(_df(sources))


def test_배지_갱신이_전역값에_반영(app):
    app._배지_갱신(_df(["synthetic"] * 4))
    assert app.배지_현재 == "합성데이터 기준 · 검증용"
    app._배지_갱신(_df(["sample"] * 4))
    assert app.배지_현재 == "샘플데이터 기준 · 검증용"
