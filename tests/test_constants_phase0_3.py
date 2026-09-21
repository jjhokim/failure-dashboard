"""
Phase 0-3 제대 명칭 상수화 검증.

제대구분 Enum이 backend/constants.py 단일 소스에서 파생됨을 보증한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import constants  # noqa: E402
from backend.models import 제대구분  # noqa: E402


def test_enum이_상수와_일치():
    assert [e.value for e in 제대구분] == constants.제대_목록


def test_제대_등급_설명_매핑():
    assert constants.제대_등급[constants.부대정비] == "1제대"
    assert constants.제대_등급[constants.야전정비] == "2제대"
    assert constants.제대_등급[constants.창정비] == "3제대"
    assert set(constants.제대_설명) == set(constants.제대_목록)
