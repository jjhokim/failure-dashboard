# 관련 연구 (2장 초안)

> 논문 2장에 쓸 문헌 정리. 각 절 끝에 **본 연구와의 차별점**을 명시했다.
> 미확인 문헌은 [확인 필요]로 표시했다 — 원문 대조 없이 인용하지 말 것.

---

## 2.1 정비기록 텍스트 마이닝과 고장모드 군집화

산업 정비기록(Maintenance Work Order, MWO)의 자유서술은 설비 고장에 대한 핵심
관측을 담고 있으나, 고장분류 코드만으로는 경향을 식별할 만큼의 상세도·정확도가
부족하다는 문제가 반복적으로 지적되어 왔다 [1, 2].

이에 따라 자유서술을 직접 다루는 접근이 제안되었다.

- **어휘 기반**: TF-IDF, 연관규칙, 용어행렬, 도메인 사전 구축 등의 전처리 기반
  텍스트 마이닝으로 고장모드를 식별·그룹화 [1]
- **임베딩 기반**: word2vec, Word2Vec Bi-LSTM, Sentence-BERT 등으로 정비기록을
  벡터화한 뒤 k-means·계층군집으로 고장모드를 발견 [2, 3]
- 기술·산업 도메인의 자유 텍스트에는 **계층군집이 유망**하다는 보고 [2]

> **본 연구와의 차별점**
> 1. 선행 연구는 대체로 **텍스트 단일 채널**에 의존한다. 본 연구는 텍스트(의미)에
>    더해 **LCN 구조거리**와 **시간 근접성**을 동등한 채널로 융합하고, 그 융합이
>    실제로 이득인지를 ablation으로 검증한다.
> 2. 선행 연구의 평가는 대체로 **사후 해석·전문가 검토**에 의존한다. 본 연구는
>    정답을 알고 있는 합성데이터로 **회수 정확도를 정량 측정**하되, 그 과정에서
>    발생하는 순환성을 설계로 차단한다(2.3).
> 3. 군집화 결과를 **우선순위화(규모·추세·치명도)** 까지 연결한다.

---

## 2.2 공통원인 고장(CCF) 분석과 시간 창

원자력 분야의 CCF 데이터베이스 체계(NUREG/CR-6268)와 PRA 모델링 지침
(NUREG/CR-5485)은 CCF 사건을 다음 조건으로 정의한다 [4, 5, 6]:

1. 둘 이상의 부품이 고장·열화되고,
2. **PRA 임무 성공이 불확실해지는 "선택된(selected) 기간" 내에** 발생하며,
3. 단일 공유 원인·결합 메커니즘에 기인하고,
4. 정해진 부품 경계 안에서 발생한다.

여기서 결정적인 점은 **시간 창이 표준화된 상수가 아니라 임무·점검주기에 따라
분석자가 선택하는 값**이라는 것이다 [4, 6].

> **본 연구에의 함의 (5장 한계와 직결)**
> 본 연구의 합성 생성기는 공통원인 군집의 시간 밀집폭을 파라미터로 둔다.
> 문헌상 "옳은" 값이 존재하지 않으므로 단일 값을 고르는 대신, 밀집폭을 1~30일로
> 스윕해 **결론이 이 가정에 어떻게 의존하는지를 보고**한다. 실제 운용이 어느
> 영역에 해당하는지는 해당 체계의 점검주기와 CCF 시간 창 정의에 달려 있으며,
> 본 연구는 이를 결정하지 않는다.

---

## 2.3 순환분석(circular analysis)과 평가 설계 ★ 본 연구의 방법론적 근거

Kriegeskorte 등은 시스템 신경과학에서 **동일 데이터를 선택과 선택적 분석에 함께
쓰는 "double dipping"** 이 검정통계량이 선택기준과 독립이 아닐 때 왜곡된 기술통계와
무효한 추론을 낳는다는 점을 보였다 [7]. 이들은 **실험 효과가 없다고 알려진
데이터**(과제 없는 휴지기 fMRI)에 통용되는 분석 절차를 적용해 **허위 효과가
생성됨**을 실증했고, 2008년 주요 저널 fMRI 논문 134편 중 **42%가 최소 한 번의
비독립 선택적 분석**을 수행했다고 보고했다 [7].

> **본 연구에의 적용**
> 합성데이터로 군집화를 평가할 때 동일한 위험이 존재한다. 정답 군집 라벨이
> 관측 특징(텍스트·LCN·시각)으로 결정론적으로 흘러들어가면, 측정된 ARI는
> **모델 성능이 아니라 주입 강도**를 재게 된다. 본 연구는 이를 설계로 차단한다.
>
> | 규칙 | 내용 | Kriegeskorte 대응 |
> |---|---|---|
> | R1 | 라벨→물리상태 변환은 규칙 기반 코드만. 문장 생성기는 `cluster_id`를 인자로 받지 않음 | 선택과 분석의 분리 |
> | R2 | 생성기와 탐지기가 문자열·임베딩 공간·사전을 공유하지 않음 | 비독립 경로 차단 |
> | R3 | BIT 로그는 조인 전용, 탐지 채널로 융합하지 않음 | 라벨의 결정론적 함수 배제 |
> | 무주입 대조군 | δ=0에서 ARI≈0, 노이즈≥0.8 확인 | **효과 없는 데이터로 허위 효과 점검** [7]과 동일한 논리 |
> | 라벨 순열 대조군 | 라벨을 섞으면 ARI가 0으로 붕괴 | 귀무 하 통계량 확인 |
>
> 추가로 라벨 누출 검사 3종(금지 토큰 출현률·TF-IDF 로지스틱 예측·어휘 Jaccard)을
> 생성 직후 자동 실행한다.

---

## 2.4 군집화 알고리즘과 한국어 문장 임베딩

- **HDBSCAN**: 밀도 기반 계층군집. 군집 수를 사전 지정하지 않고 노이즈(-1)를
  명시적으로 분리하므로, **단발성 고장이 섞인 정비기록**에 적합하다.
  본 연구는 사전계산 거리행렬(`metric='precomputed'`)로 3채널 융합 거리를 투입한다.
  [확인 필요: McInnes, Healy & Astels, *hdbscan*, JOSS 2017 원문 서지]
- **KoSBERT**: 한국어 문장 임베딩. 본 연구는
  `snunlp/KR-SBERT-V40K-klueNLI-augSTS`를 기본 모델로 사용한다.
  [확인 필요: SNU NLP 모델 카드·인용 형식]
- **Sentence-BERT**: 문장 단위 의미 유사도 임베딩의 기반 [확인 필요: Reimers &
  Gurevych, EMNLP 2019]

---

## 2.5 RAM 지표와 정비 제대

- 방위사업청 「무기체계 RAM 업무지침」(2018) — Ai/Ao·MTBF·MTTR 정의 근거
  [확인 필요: 사용자 원문 대조]
- 「종합군수지원 개발 실무지침서」(2015.7) <표 1-3> — 유도탄 정비 제대 명칭
  (부대정비/야전정비/창정비) [확인 필요: 사용자 원문 대조]

> **본 연구의 처리**
> - 기존 구현이 `MTBF/(MTBF+MTTR)`을 "Ao"로 표기하던 것을 **Ai(고유가용도)로 정정**.
>   Ao = MTBM/(MTBM+MDT)는 MDT/ALDT 결측 시 **산출하지 않는다**(억지 계산 금지).
> - MTTR은 진행 중 수리건을 **우측절단으로 편입해 Kaplan-Meier**로 보정한다.
>   완료건만 평균하면 MTTR이 과소, Ai가 과대 추정된다.

---

## 참고문헌 (초안)

[1] Text-Mining Maintenance Records to Automate the Identification and Grouping of
    Failure Modes. *OTC Offshore Technology Conference*, 2020.
    https://onepetro.org/OTCONF/proceedings-abstract/20OTC/20OTC/D041S055R002/107412

[2] Semantic and Engineering-Based Embedding for Classification List Development.
    *Machine Learning and Knowledge Extraction*, 8(3), 61.
    https://doi.org/10.3390/make8030061

[3] Leveraging Failure Modes and Effect Analysis for Technical Language Processing.
    *Machine Learning and Knowledge Extraction*, 7(2), 42.
    https://doi.org/10.3390/make7020042

[4] NUREG/CR-6268 Rev.1, *Common-Cause Failure Database and Analysis System*,
    U.S. NRC / INL. https://www.nrc.gov/docs/ML0729/ML072970404.pdf

[5] NUREG/CR-5485, *Guidelines on Modeling Common-Cause Failures in PRA*, U.S. NRC.
    https://nrcoe.inl.gov/publicdocs/CCF/NUREGCR-5485_Guidelines%20on%20Modeling%20Common-Cause%20Failures%20in%20PRA.pdf

[6] *Common-Cause Failure Analysis in Event Assessment*, U.S. NRC.
    https://www.nrc.gov/docs/ML0817/ML081720219.pdf

[7] Kriegeskorte, N., Simmons, W.K., Bellgowan, P.S.F., & Baker, C.I. (2009).
    Circular analysis in systems neuroscience: the dangers of double dipping.
    *Nature Neuroscience*, 12(5), 535–540. https://doi.org/10.1038/nn.2303

[8] Extracting failure time data from industrial maintenance records using text
    mining. *Advanced Engineering Informatics*.
    https://www.sciencedirect.com/science/article/abs/pii/S1474034616301380

---

## 작성 시 주의

1. [확인 필요] 표시 문헌은 **원문을 직접 확인한 뒤** 서지를 확정할 것.
   특히 국내 규정 2종은 사용자가 원문 대조하기로 되어 있다.
2. [1][2][3][8]은 **검색 결과 스니펫 수준으로만 확인**했다. 인용 전 초록/본문을
   읽고 주장이 실제로 그 문헌에 있는지 대조할 것.
3. [7]만이 본 연구 설계의 **직접적 이론 근거**이므로, 2.3은 원문을 반드시 읽고
   쓸 것. (요약 수준의 재인용은 피할 것)
