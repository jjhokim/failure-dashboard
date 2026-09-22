"""
KIMST 학술대회 논문 초안 생성 (2페이지·2칼럼 템플릿 채우기).

    python -m experiments.build_kimst_paper

업로드된 템플릿(e6d10184-_____.docx)의 단락 스타일을 보존한 채 본문만 교체한다.
스타일을 직접 만들지 않고 템플릿 단락의 run 서식을 재사용하므로 학회 양식이
깨지지 않는다.

주의: 템플릿 제약이 **2페이지 이하**이므로 분량을 의도적으로 압축했다.
      그림·표는 자리만 표시하고 실제 삽입은 사용자가 수행한다.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document

_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(
    "C:/Users/jjho9/.claude/uploads/0d289df9-73b8-4972-b91f-d455cc4c851e/"
    "e6d10184-_____.docx"
)
OUT = _ROOT / "paper_assets" / "KIMST_초안.docx"


def _set(par, text: str) -> None:
    """단락의 첫 run 서식을 유지한 채 텍스트를 교체한다."""
    if par.runs:
        par.runs[0].text = text
        for r in par.runs[1:]:
            r.text = ""
    else:
        par.add_run(text)


# ── 원고 내용 ────────────────────────────────────────────────────────────
TITLE_KO = "정비기록 자유서술 기반 공통원인 후보 탐지 및 우선순위화"
TITLE_EN = ("Common-Cause Candidate Screening and Prioritization "
            "from Free-Text Maintenance Records")

AUTHORS_KO = "김민준*"
AUTHORS_EN = "Min Jun Kim*"
AFFIL = "* LIG넥스원 미사일시스템 IPS연구소"
EMAIL = "(minjun.kim@lignex1.com)"

ABSTRACT = (
    "Maintenance records of guided weapon systems contain free-text narratives "
    "that hold key observations on failures, but failure codes alone are too "
    "coarse to reveal common causes. We propose a screening pipeline that fuses "
    "three channels-semantic (KoSBERT), structural (LCN tree distance) and "
    "temporal-and clusters them with HDBSCAN, then ranks candidate clusters by "
    "size, trend and criticality. To avoid circular validation, the label-to-"
    "observation path is blocked by design. Under preregistered criteria with "
    "eight seeds, the fusion advantage proved conditional on the temporal "
    "dispersion of the injected signal."
)

KEYWORDS = ("Key Words : Guided Weapon, Common-Cause Failure, Maintenance Record, "
            "Text Clustering, Circular Validation")

SEC1_TITLE = "1. 서론"
SEC1 = (
    "무기체계의 가용도 향상과 총수명주기비용 관리를 위해 MRO 관점의 데이터 수집과 "
    "활용의 중요성이 커지고 있다. 특히 실제 운용 필드에서 발생하는 정비·고장 데이터의 "
    "수집 방향성과 품질 관리가 선결 과제로 제시된 바 있다[1]. 그러나 수집된 정비기록 중 "
    "고장분류 코드는 경향을 식별할 만큼의 상세도를 갖지 못하는 경우가 많고, 실제 정보는 "
    "정비사가 작성한 자유서술에 담겨 있다[2]. "
    "본 연구는 수집 이후 단계, 즉 축적된 자유서술 정비기록에서 동일 원인에 기인할 "
    "가능성이 있는 고장군, 곧 공통원인 후보를 자동으로 선별하고 조치 우선순위를 "
    "매기는 스크리닝 파이프라인을 제안한다. "
    "다만 합성데이터로 이러한 군집화를 평가할 때는 순환검증 위험이 존재한다. 정답 "
    "라벨이 관측 특징으로 결정론적으로 흘러들어가면 측정된 성능은 모델의 능력이 아니라 "
    "신호 주입 강도를 재게 된다[3]. 본 연구는 이를 설계 단계에서 차단한 점을 주요 "
    "기여로 한다."
)

SEC2_TITLE = "2. 제안 방법 및 검증 설계"
SEC21_TITLE = "2.1 3채널 융합 탐지 및 우선순위화"
SEC21 = (
    "정비기록 한 건을 노드로 두고 세 채널의 거리행렬을 만든다. 의미 채널은 KoSBERT "
    "문장 임베딩의 코사인 거리, 구조 채널은 군수관리번호(LCN) 계층트리의 경로거리를 "
    "최대깊이로 정규화한 값, 시간 채널은 1-exp(-|Δt|/τ)로 정의한다(τ=30일). "
    "각 채널을 min-max 정규화한 뒤 가중합하여 융합 거리행렬을 만들고, 사전계산 거리 "
    "기반 HDBSCAN으로 군집화한다. 가중치를 0으로 두면 해당 채널이 꺼지므로 동일 "
    "구현으로 채널 절제(ablation) 실험이 가능하다. "
    "군집 단위 점수는 규모, 추세, 치명도를 각각 표준화해 가중합한다. 이때 탐지기는 "
    "주입된 치명도 파라미터를 알 수 없으므로 관측치 기반 대체값만 사용한다."
)

SEC22_TITLE = "2.2 순환검증 차단 설계"
SEC22 = (
    "합성데이터 생성 시 군집 라벨이 관측 텍스트로 새지 않도록 세 가지 규칙을 강제했다. "
    "첫째, 라벨에서 중간 물리상태(LCN·증상·환경·시각)로의 변환은 규칙 기반 코드만 "
    "수행하며, 문장 생성기는 군집 식별자를 인자로 받지 않는다. 둘째, 생성기와 탐지기는 "
    "문자열·임베딩 공간·동의어 사전을 공유하지 않는다. 셋째, 모의 BIT 로그는 LCN과 "
    "시각 윈도우 조인에만 사용하고 탐지 채널로 융합하지 않는다. "
    "생성 직후에는 금지 토큰 출현률, TF-IDF 로지스틱 회귀의 문장→라벨 예측 정확도, "
    "군집 간 어휘 Jaccard 중복도의 세 가지 누출 검사를 자동 실행한다. "
    "또한 신호를 전혀 주입하지 않은 대조군과 정답 라벨을 무작위 순열한 대조군을 두어, "
    "효과가 없는 데이터에서 허위 군집이 형성되지 않는지 확인했다[3]."
)

TABLE1_CAP = "Table 1. Preregistered criteria and results (8 seeds)"

SEC23_TITLE = "2.3 실험 및 결과"
SEC23 = (
    "합격기준과 절제 조합을 실행 전에 사전등록하고, 모든 수치는 8개 시드의 평균과 "
    "표준편차로 보고했다. 합성데이터는 600건, 10개 군집 조건으로 생성했다. "
    "무주입 대조군에서 조정 랜드지수(ARI)는 -0.002±0.010, 노이즈 비율은 0.832였고, "
    "라벨 순열 대조군에서도 ARI가 -0.003±0.006으로 붕괴하여 순환성이 없음을 확인했다. "
    "주입 강도를 높이면 ARI가 단조 증가했으나(δ=1.0에서 0.736±0.248), 3채널 융합이 "
    "시간 채널 단독(0.853±0.037)에 미치지 못해 융합 우위 기준은 충족하지 못했다. "
    "원인을 확인하기 위해 가중치를 탐색한 결과 최적해가 시간 채널 단독으로 수렴했고, "
    "이는 생성기가 δ=1.0에서 군집 발생 시각을 하루 이내로 과도하게 밀집시켜 시간 "
    "채널이 거의 완전한 판별자가 되었기 때문이었다. 밀집폭을 관측기간의 1%로 완화하자 "
    "융합이 0.938, 시간 단독이 0.786으로 역전되었으며 차이의 95% 신뢰구간은 "
    "[0.090, 0.236]으로 0을 포함하지 않았다. 이 조건에서 가중치를 다시 탐색하면 세 "
    "채널이 모두 유지되었다."
)

FIG1_CAP = "Fig. 1. Pipeline and the blocked paths preventing circular validation"
FIG1_IMAGE = _ROOT / "paper_assets" / "fig0_architecture.png"

# Table 1 — 사전등록 기준과 결과 (7행 2열 템플릿에 맞춤)
TABLE1_ROWS = [
    ("Criterion (preregistered)", "Result (8 seeds)"),
    ("No-injection control, ARI", "-0.002 ± 0.010  (pass)"),
    ("No-injection control, noise ratio", "0.832  (pass)"),
    ("Label permutation, ARI", "-0.003 ± 0.006  (pass)"),
    ("Monotonicity over injection level", "pass"),
    ("Fusion >= best single channel", "0.736 < 0.853  (fail)"),
    ("Ranking, Spearman rho", "0.526, CI [0.267, 0.760]  (pass)"),
]

SEC3_TITLE = "3. 결론"
SEC3 = (
    "본 연구는 자유서술 정비기록에서 공통원인 후보를 선별·우선순위화하는 파이프라인을 "
    "구현하고, 순환검증을 차단한 합성 평가 설계로 내적 타당도를 검증했다. "
    "핵심 결과는 채널 융합의 이점이 무조건적이지 않고 공통원인 신호의 시간 분산 정도에 "
    "조건부라는 점이다. 군집이 하루 내에 몰리는 극단적 가정에서는 시간 채널만으로 "
    "충분하지만, 밀집폭이 완화되면 융합이 유의하게 우세했다. "
    "한계로는 첫째, 평가가 합성데이터에 기반하므로 실운용 성능을 주장하지 않는다. "
    "둘째, 시간 밀집폭은 실측 근거가 없는 설계값이며 공통원인 고장의 시간 창은 문헌상 "
    "임무와 점검주기에 따라 선택되는 값이다[4]. 셋째, 주입 강도가 낮은 구간에서는 "
    "신호가 회수되지 않아 본 도구는 조기경보가 아니라 사후 스크리닝으로 위치시켜야 "
    "한다. 향후 실제 야전 정비기록을 확보하여 시간 밀집도를 추정하는 연구가 필요하다."
)

ACK = ("본 연구는 저자 소속 기관의 자율연구과제로 수행되었다. "
       "(발표 전 보안성 검토 필요)")

REFS = [
    "[1]\tM. J. Kim, J. Y. Park and D. M. Jeong, “Study on the directing of data "
    "collection for MRO based on guided missile failure data,” Journal of the "
    "Korea Academia-Industrial cooperation Society, Vol. 26, No. 10, "
    "pp. 104-110, 2025.",
    "[2]\tM. Stewart, et. al., “Text-Mining Maintenance Records to Automate the "
    "Identification and Grouping of Failure Modes,” Offshore Technology "
    "Conference, OTC-30697-MS, 2020.",
    "[3]\tN. Kriegeskorte, W. K. Simmons, P. S. F. Bellgowan and C. I. Baker, "
    "“Circular analysis in systems neuroscience: the dangers of double dipping,” "
    "Nature Neuroscience, Vol. 12, No. 5, pp. 535-540, 2009.",
    "[4]\tU.S. Nuclear Regulatory Commission, “Common-Cause Failure Database and "
    "Analysis System,” NUREG/CR-6268 Rev.1, 2007.",
]

# 템플릿 단락 인덱스 → 새 내용
# 제목은 반드시 템플릿의 '제목 슬롯'에 넣는다. 본문 단락에 제목을 줄바꿈으로
# 붙이면 제목 서식(돋움 진하게)이 적용되지 않는다.
REPLACE = {
    0: TITLE_KO, 1: TITLE_EN,
    3: AUTHORS_KO, 4: AUTHORS_EN,
    6: AFFIL, 7: "", 9: EMAIL,
    12: ABSTRACT, 14: KEYWORDS,
    17: SEC1_TITLE, 18: SEC1,
    20: SEC2_TITLE,
    21: SEC21_TITLE, 22: SEC21,
    24: SEC22_TITLE,                 # 항 제목 슬롯을 절 제목으로 사용
    25: "", 26: "", 27: "", 28: "",
    29: SEC22,
    31: FIG1_CAP, 32: "", 33: "",    # 그림 캡션(아키텍처)은 방법 절에 배치
    35: "", 37: "",
    39: "", 40: "", 41: "",
    43: SEC23_TITLE, 44: SEC23,      # 2.2 절 제목 슬롯 → 2.3 실험 및 결과
    46: SEC3_TITLE, 47: SEC3,
    49: "후       기", 50: ACK,
    52: "References",
    53: REFS[0], 54: REFS[1], 55: REFS[2], 56: REFS[3],
}

# 2.3 본문 뒤에 표 캡션 단락을 새로 삽입한다(템플릿에 남는 슬롯이 없으므로).
INSERT_BEFORE = 46          # '3. 결론' 앞
CAPTION_STYLE_FROM = 31     # 캡션 서식을 가져올 단락

# 템플릿 안내문(58번 이후)은 초안에서 제거한다
DROP_FROM = 58


def build() -> Path:
    doc = Document(str(TEMPLATE))
    paras = doc.paragraphs

    for idx, text in REPLACE.items():
        if idx < len(paras):
            _set(paras[idx], text)

    # 표 캡션 단락 삽입 (2.3 본문 뒤 · '3. 결론' 앞)
    캡션 = paras[INSERT_BEFORE].insert_paragraph_before(TABLE1_CAP)
    캡션.style = paras[CAPTION_STYLE_FROM].style
    캡션.paragraph_format.alignment = paras[CAPTION_STYLE_FROM].paragraph_format.alignment
    if paras[CAPTION_STYLE_FROM].runs and 캡션.runs:
        src, dst = paras[CAPTION_STYLE_FROM].runs[0].font, 캡션.runs[0].font
        dst.name, dst.size, dst.bold = src.name, src.size, src.bold

    # 작성 안내문 제거
    for p in paras[DROP_FROM:]:
        _set(p, "")

    # Table 1 채우기 (템플릿 표 서식 유지)
    if doc.tables:
        tb = doc.tables[0]
        for r, row in enumerate(tb.rows):
            값 = TABLE1_ROWS[r] if r < len(TABLE1_ROWS) else ("", "")
            for c, cell in enumerate(row.cells[:2]):
                for j, p in enumerate(cell.paragraphs):
                    _set(p, 값[c] if j == 0 else "")

    # Fig. 1 이미지 삽입 (캡션 단락 바로 앞)
    if FIG1_IMAGE.exists():
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm

        캡션단락 = None
        for p in doc.paragraphs:
            if p.text.strip().startswith("Fig. 1."):
                캡션단락 = p
                break
        if 캡션단락 is not None:
            그림 = 캡션단락.insert_paragraph_before("")
            그림.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # 2칼럼 폭(약 8.5cm)에 맞춘다
            그림.add_run().add_picture(str(FIG1_IMAGE), width=Cm(8.2))
    else:
        print(f"[경고] 그림 파일 없음: {FIG1_IMAGE}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"저장: {OUT.relative_to(_ROOT)}")
    return OUT


if __name__ == "__main__":
    build()
