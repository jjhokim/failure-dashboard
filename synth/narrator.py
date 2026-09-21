"""
[LLM/템플릿] 중간 물리상태 → 자유서술 문장 (Phase 1, R1 준수).

절대 규칙 R1: narrator는 물리상태(부품·증상·환경)만 받는다. cluster_id는
render() 시그니처에서 배제된다 — "정답 라벨은 신경망을 거치지 않는다".

실제 LLM API 키가 없을 때를 위해 템플릿+동의어 치환 기반 TemplateNarrator를
기본 구현으로 둔다(오프라인 재현성). LLM 어댑터는 동일한 render(state)->str
인터페이스를 따른다. 사용한 프롬프트 전문은 synth/prompts/에 보관한다.

이 모듈은 KoSBERT 등 탐지기 구성요소를 import하지 않는다(R2).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# 증상 코드 → 서술 동의어 (표현 편차 모사)
증상_동의어 = {
    "과열": ["과열", "온도 급상승", "발열 이상", "이상 고온"],
    "진동": ["비정상 진동", "진동 증가", "떨림 발생", "이상 진동"],
    "누유": ["오일 누유", "유체 누설", "누유 흔적", "기름 샘"],
    "전압강하": ["전압 강하", "출력 전압 저하", "전압 불안정", "전원 저하"],
    "통신두절": ["통신 두절", "신호 끊김", "데이터 링크 상실", "교신 불가"],
    "균열": ["균열 발생", "크랙 확인", "표면 균열", "미세 균열"],
    "오작동": ["오작동", "비정상 동작", "제어 이상", "기능 이상"],
    "소음": ["이상 소음", "잡음 발생", "소음 증가", "비정상 음"],
}

# 환경 코드 → 서술 구
환경_구 = {
    "고온": ["고온 환경에서", "높은 기온 조건에서"],
    "저온": ["저온 환경에서", "한랭 조건에서"],
    "습윤": ["습윤한 환경에서", "고습 조건에서"],
    "분진": ["분진이 많은 환경에서", "먼지 조건에서"],
    "진동환경": ["진동이 심한 환경에서", "고진동 조건에서"],
    "정상": ["정상 운용 중", "통상 운용 조건에서"],
}

# 문장 템플릿
템플릿 = [
    "{env} {part}에서 {sym}이(가) 발생함.",
    "{part} {sym} 확인. {env} 포착됨.",
    "{env} 운용 중 {part} {sym} 증상으로 정비 요청.",
    "{part}의 {sym}. {env} 재현됨.",
]


class Narrator(Protocol):
    """물리상태 dict → 서술 문자열. cluster_id는 인자에 없음."""

    def render(self, state: dict) -> str: ...


class TemplateNarrator:
    """
    템플릿+동의어 치환 기반 오프라인 서술 생성기.

    state = {"part": str, "symptom_code": str, "environment": str}
    (cluster_id·lcn·라벨은 포함되지 않는다 — R1)
    """

    def __init__(self, expression_variance: float = 0.5, seed: int = 0):
        self.expression_variance = float(expression_variance)
        self.rng = np.random.RandomState(seed)

    def _pick(self, options: list[str]) -> str:
        # 표현 편차가 클수록 동의어를 무작위 선택, 작을수록 표준형(첫 항목) 유지
        if len(options) > 1 and self.rng.rand() < self.expression_variance:
            return options[self.rng.randint(len(options))]
        return options[0]

    def render(self, state: dict) -> str:
        if "cluster_id" in state:
            raise ValueError("R1 위반: narrator는 cluster_id를 받을 수 없습니다.")
        part = state["part"]
        sym = self._pick(증상_동의어.get(state["symptom_code"], [state["symptom_code"]]))
        env = self._pick(환경_구.get(state["environment"], [state["environment"]]))
        tmpl = self._pick(템플릿)
        return tmpl.format(env=env, part=part, sym=sym)


def get_narrator(config) -> Narrator:
    """설정에 맞는 narrator를 반환한다. 기본은 오프라인 TemplateNarrator."""
    return TemplateNarrator(
        expression_variance=config.expression_variance, seed=config.seed
    )
