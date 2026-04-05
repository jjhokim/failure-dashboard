# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

방위산업 IPS(종합군수지원) 보조 도구. 정비 3제대(1제대: 운용부대, 2제대: 정비부대, 3제대: 창정비)의 고장이력을 입력·조회·분석하는 대시보드.

- 사용자: 각 제대 정비 담당자, IPS 담당자
- 체계: 익명화된 가상 체계 (체계-A, 체계-B 등)
- 실행 환경: 로컬 단일 PC (다중 접속/보안 구조는 미구현)

## 기술 스택

- **Python 3.10**, Streamlit (프론트엔드), SQLite (데이터 저장)
- **pandas, numpy** (데이터 처리), **Plotly** (시각화)
- Jupyter Notebook (EDA / 프로토타이핑)

## 개발 명령어

```bash
# 의존성 설치
pip install -r requirements.txt

# 대시보드 실행
streamlit run frontend/app.py

# Jupyter 노트북 실행
jupyter notebook notebooks/eda.ipynb
```

## 아키텍처

```
failure-dashboard/
├── backend/
│   ├── database.py   # SQLite 연결, 테이블 초기화
│   ├── models.py     # 데이터 스키마 (dataclass / pydantic)
│   └── crud.py       # DB CRUD 함수 (입력, 조회, 필터링)
├── frontend/
│   └── app.py        # Streamlit 멀티페이지 앱 진입점
├── data/
│   └── sample_data.csv  # 현실적인 가상 샘플 데이터
└── notebooks/
    └── eda.ipynb     # 탐색적 데이터 분석, 지표 검증
```

### 데이터 흐름

```
frontend/app.py
    └── backend/crud.py   (입력 저장 / 조회 필터)
            └── backend/database.py  (SQLite 연결)
                    └── backend/models.py  (스키마 정의)
```

- `frontend`는 `backend`만 import한다. DB를 직접 호출하지 않는다.
- `crud.py`는 pandas DataFrame을 반환하여 Streamlit에서 바로 사용한다.
- 샘플 데이터(`data/sample_data.csv`)는 앱 최초 실행 시 DB에 seed된다.

## 핵심 도메인 개념

| 용어 | 설명 |
|------|------|
| **Ao** (Operational Availability) | 운용 가용도 = MTBF / (MTBF + MTTR) |
| **MTBF** | 평균고장간격 (Mean Time Between Failures) |
| **MTTR** | 평균수리시간 (Mean Time To Repair) |
| **LRU** | 현장교체품목 (Line Replaceable Unit) |
| **불가동** | 장비가 임무 수행 불가한 상태 |
| **1제대** | 운용부대 (사용자 부대) |
| **2제대** | 정비부대 (야전 정비) |
| **3제대** | 창정비 (공장급 정비) |

## 주요 데이터 필드

고장이력 레코드의 핵심 필드:

| 필드 | 설명 |
|------|------|
| `제대구분` | 운용부대 / 정비부대 / 창정비 |
| `발생일시` | datetime |
| `체계명` | 체계-A, 체계-B, … |
| `LRU명` | 고장 발생 부품명 |
| `고장유형` | 전기적 / 기계적 / 소프트웨어 / 기타 |
| `고장증상` | 자유 텍스트 |
| `수리시작시간` / `수리완료시간` | datetime (MTTR 계산에 사용) |
| `정비조치내용` | 자유 텍스트 |
| `처리상태` | 수리완료 / 수리중 / 미해결 |

## 코드 작성 규칙

- **모든 주석과 UI 텍스트는 한국어**로 작성한다.
- 분석 지표(Ao, MTBF, MTTR) 계산 로직은 `crud.py` 또는 별도 `analytics.py`에 집중한다. Streamlit 페이지에 계산 로직을 직접 작성하지 않는다.
- Streamlit 페이지가 많아질 경우 `frontend/pages/` 디렉터리로 분리한다.

## 향후 확장 방향 (현재 미구현)

- AI 기반 고장 예측 (데이터 충분히 수집 후 적용)
- 다중 사용자 접속 / 인증 구조
- 보안 강화 (네트워크 분리, 암호화)
