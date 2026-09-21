"""합성 정비데이터 생성기 패키지 (Phase 1).

R1: 라벨→물리상태는 physical.py(규칙 기반)에서만. narrator는 라벨 미입력.
R2: 이 패키지는 KoSBERT 등 탐지기 구성요소를 import하지 않는다.
"""

from synth.config import SynthConfig
from synth.generate import generate

__all__ = ["SynthConfig", "generate"]
