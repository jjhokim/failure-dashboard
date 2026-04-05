"""
고장·결함 분석 대시보드 — Streamlit 앱 진입점

실행:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, date, time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.database import init_db
from backend.models import (
    LRU_목록, 고장이력, 고장유형, 처리상태, 제대구분, 체계명,
)
from backend.crud import (
    KPI_요약,
    고장이력_단건조회,
    고장이력_삭제,
    고장이력_수정,
    고장이력_저장,
    고장이력_전체조회,
    미해결_고장_조회,
    불가동_추세,
    월별_고장건수,
    제대별_고장현황,
    체계별_MTTR,
    파레토_분석,
)

# ---------------------------------------------------------------------------
# 페이지 설정 (가장 먼저 호출)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="고장·결함 분석 대시보드",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ---------------------------------------------------------------------------
# 색상 상수
# ---------------------------------------------------------------------------

C = {
    "bg":       "#0e1117",
    "card":     "#1e2130",
    "border":   "#2d3348",
    "text":     "#e8eaf0",
    "muted":    "#8892a4",
    "blue":     "#4c9be8",
    "green":    "#3ecf8e",
    "orange":   "#f5a623",
    "red":      "#e85454",
    "purple":   "#a78bfa",
}

색상_제대 = {
    "운용부대": C["blue"],
    "정비부대": C["orange"],
    "창정비":   C["purple"],
}
색상_처리상태 = {
    "수리완료": C["green"],
    "수리중":   C["orange"],
    "미해결":   C["red"],
}

# Plotly 공통 레이아웃 (다크 테마)
_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
    margin=dict(t=32, b=48, l=16, r=16),
    xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
)

# ---------------------------------------------------------------------------
# 전역 CSS 주입
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
    /* ── 전체 배경 ── */
    .stApp {{ background-color: {C['bg']}; }}

    /* ── 사이드바 ── */
    [data-testid="stSidebar"] {{
        background-color: {C['card']};
        border-right: 1px solid {C['border']};
    }}

    /* ── 탭 ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: {C['card']};
        border-radius: 8px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px;
        color: {C['muted']};
        padding: 6px 18px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {C['blue']}22;
        color: {C['blue']};
    }}

    /* ── KPI 카드 ── */
    .kpi-card {{
        background: {C['card']};
        border-radius: 10px;
        padding: 18px 20px 14px;
        border-left: 4px solid #444;
        margin-bottom: 4px;
    }}
    .kpi-label {{
        color: {C['muted']};
        font-size: 12px;
        font-weight: 600;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin: 0 0 6px;
    }}
    .kpi-value {{
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }}
    .kpi-sub {{
        color: {C['muted']};
        font-size: 12px;
        margin: 4px 0 0;
    }}

    /* ── 섹션 헤더 ── */
    .section-header {{
        font-size: 14px;
        font-weight: 600;
        color: {C['muted']};
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: 20px 0 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid {C['border']};
    }}

    /* ── 상태 뱃지 ── */
    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }}

    /* ── expander ── */
    [data-testid="stExpander"] {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 8px;
    }}

    /* ── 입력 위젯 ── */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {{
        background: {C['bg']};
        border-color: {C['border']};
        color: {C['text']};
    }}

    /* ── divider ── */
    hr {{ border-color: {C['border']} !important; }}

    /* ── 데이터프레임 헤더 ── */
    [data-testid="stDataFrame"] th {{
        background: {C['card']} !important;
        color: {C['muted']} !important;
        font-size: 12px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def _h(text: str) -> str:
    """
    HTML 블록 내 non-ASCII 문자(한글 등)를 HTML NCR(&#N;)로 변환한다.
    Windows 환경에서 unsafe_allow_html 렌더링 시 발생하는 인코딩 깨짐을 방지한다.
    """
    return "".join(f"&#{ord(c)};" if ord(c) > 127 else c for c in text)


def _kpi_card(label: str, value: str, sub: str = "", border_color: str = C["blue"]) -> None:
    """HTML 기반 KPI 카드 렌더링."""
    sub_html = "" if not sub else f'<p class="kpi-sub">{_h(sub)}</p>'
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{border_color}">
            <p class="kpi-label">{_h(label)}</p>
            <p class="kpi-value" style="color:{border_color}">{_h(value)}</p>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section(title: str) -> None:
    st.markdown(
        f'<p class="section-header">{_h(title)}</p>',
        unsafe_allow_html=True,
    )


def _빈_차트(메시지: str = "표시할 데이터가 없습니다.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=메시지, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=13, color=C["muted"]),
    )
    fig.update_layout(
        **_PLOT_LAYOUT,
        xaxis_visible=False, yaxis_visible=False, height=280,
    )
    return fig


def _parse_dt_text(text: str, 오류: list) -> datetime | None:
    """'YYYY-MM-DD HH:MM:SS' 텍스트를 datetime으로 변환. 실패 시 오류 목록에 추가."""
    text = text.strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        오류.append(f"날짜 형식 오류: '{text}' — YYYY-MM-DD HH:MM:SS 형식으로 입력하세요.")
        return None


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------

def _사이드바() -> str:
    with st.sidebar:
        # 타이틀 — 네이티브 컴포넌트 사용 (HTML 블록 인코딩 문제 방지)
        st.markdown("### 🛠️ 고장·결함 분석")
        st.caption("Fault & Defect Analysis Dashboard")

        st.divider()

        메뉴 = st.radio(
            "메뉴",
            ["📊 요약 대시보드", "✏️ 고장 입력", "📋 고장 현황", "🔍 분석"],
            label_visibility="collapsed",
        )

        st.divider()

        # DB 현황 요약 — 네이티브 컴포넌트 사용
        df_all = 고장이력_전체조회()
        kpi    = KPI_요약(df_all)
        ao_pct = kpi["가용도_Ao"] * 100
        ao_delta_color = "normal" if ao_pct >= 80 else "inverse"

        st.metric(
            label="현재 가용도 (Ao)",
            value=f"{ao_pct:.1f}%",
            delta=f"목표 80% {'달성' if ao_pct >= 80 else '미달'}",
            delta_color=ao_delta_color,
        )
        st.caption(f"미완료 {kpi['미완료_건수']}건 / 전체 {kpi['총_고장건수']}건")

        st.divider()
        st.caption("IPS 보조 도구 v0.1 · 로컬 전용")

    return 메뉴


# ---------------------------------------------------------------------------
# 탭 1: 요약 대시보드
# ---------------------------------------------------------------------------

def 페이지_요약대시보드() -> None:
    _section("전체 현황 요약")

    df  = 고장이력_전체조회()
    kpi = KPI_요약(df)
    ao  = kpi["가용도_Ao"]

    # KPI 카드 5개
    c1, c2, c3, c4, c5 = st.columns(5)
    ao_color = C["green"] if ao >= 0.80 else C["red"]
    with c1:
        _kpi_card("가동률 Ao", f"{ao*100:.1f}%",
                  sub="목표 ≥ 80%", border_color=ao_color)
    with c2:
        _kpi_card("MTBF", f"{kpi['MTBF_h']:.1f}h",
                  sub="평균 고장 간격", border_color=C["blue"])
    with c3:
        _kpi_card("MTTR", f"{kpi['MTTR_h']:.1f}h",
                  sub="평균 수리 시간", border_color=C["orange"])
    with c4:
        _kpi_card("총 고장건수", f"{kpi['총_고장건수']}건",
                  border_color=C["purple"])
    with c5:
        _kpi_card("미완료", f"{kpi['미완료_건수']}건",
                  sub="수리중 + 미해결", border_color=C["red"])

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    # 월별 고장 트렌드 — 꺾은선
    with col_l:
        _section("월별 고장 트렌드 (제대별)")
        월별 = 월별_고장건수(df)
        if 월별.empty:
            st.plotly_chart(_빈_차트(), use_container_width=True)
        else:
            fig = px.line(
                월별, x="연월", y="고장건수", color="제대구분",
                markers=True,
                color_discrete_map=색상_제대,
                labels={"연월": "", "고장건수": "고장건수 (건)", "제대구분": "제대구분"},
                height=320,
            )
            fig.update_traces(line=dict(width=2), marker=dict(size=6))
            fig.update_layout(**_PLOT_LAYOUT, xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)

    # 미해결 목록
    with col_r:
        _section("미해결 / 수리중 목록")
        미해결 = 미해결_고장_조회(limit=10)
        if 미해결.empty:
            st.success("현재 미해결 고장이 없습니다.")
        else:
            표시 = 미해결[["발생일시", "제대구분", "체계명", "LRU명", "처리상태"]].copy()
            st.dataframe(
                표시,
                use_container_width=True,
                hide_index=True,
                height=320,
                column_config={
                    "처리상태": st.column_config.TextColumn("처리상태", width="small"),
                },
            )

    st.divider()

    # 제대별 MTTR 수평 막대
    _section("제대별 평균 수리시간 (MTTR) 비교")
    mttr_df = 체계별_MTTR(df)
    if mttr_df.empty:
        st.plotly_chart(_빈_차트(), use_container_width=True)
    else:
        fig2 = px.bar(
            mttr_df, x="평균MTTR_h", y="체계명", color="제대구분",
            barmode="group", orientation="h",
            color_discrete_map=색상_제대,
            labels={"평균MTTR_h": "평균 MTTR (h)", "체계명": ""},
            height=260,
        )
        fig2.update_layout(**_PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# 탭 2: 고장 입력
# ---------------------------------------------------------------------------

def 페이지_고장입력() -> None:
    _section("고장이력 신규 등록")

    with st.form("고장입력_폼", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            입력_제대 = st.selectbox("제대구분 *", [e.value for e in 제대구분])
            입력_체계명 = st.selectbox("체계명 *", [e.value for e in 체계명])
            입력_LRU   = st.selectbox("LRU명 *", LRU_목록)
            입력_고장유형 = st.selectbox("고장유형 *", [e.value for e in 고장유형])
            입력_처리상태 = st.selectbox("처리상태 *", [e.value for e in 처리상태])

        with col2:
            # 발생일시: date + time 분리 입력
            st.markdown(
                f'<p style="font-size:13px;color:{C["muted"]};margin-bottom:4px">'
                f'{_h("발생일시 *")}</p>',
                unsafe_allow_html=True,
            )
            d1, d2 = st.columns(2)
            with d1:
                입력_발생일 = st.date_input(
                    "발생 날짜", value=date.today(), label_visibility="collapsed"
                )
            with d2:
                입력_발생시간 = st.time_input(
                    "발생 시간", value=time(0, 0), label_visibility="collapsed",
                    step=60,
                )

            입력_수리시작 = st.text_input(
                "수리시작시간", placeholder="YYYY-MM-DD HH:MM:SS (선택)"
            )
            입력_수리완료 = st.text_input(
                "수리완료시간", placeholder="YYYY-MM-DD HH:MM:SS (선택)"
            )
            입력_증상 = st.text_area(
                "고장증상 *", height=80, placeholder="고장 증상을 상세히 입력하세요."
            )
            입력_조치 = st.text_area(
                "정비조치내용", height=80, placeholder="수행한 조치 내용 (선택)"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        제출 = st.form_submit_button(
            "고장이력 등록", type="primary", use_container_width=True
        )

    if 제출:
        오류: list[str] = []

        if not 입력_증상.strip():
            오류.append("고장증상을 입력해주세요.")

        # 발생일시 조합
        발생dt = datetime(
            입력_발생일.year, 입력_발생일.month, 입력_발생일.day,
            입력_발생시간.hour, 입력_발생시간.minute,
        )

        수리시작dt = _parse_dt_text(입력_수리시작, 오류)
        수리완료dt = _parse_dt_text(입력_수리완료, 오류)

        if 오류:
            for msg in 오류:
                st.error(msg)
        else:
            record = 고장이력(
                제대구분=입력_제대,
                발생일시=발생dt,
                체계명=입력_체계명,
                LRU명=입력_LRU,
                고장유형=입력_고장유형,
                고장증상=입력_증상.strip(),
                수리시작시간=수리시작dt,
                수리완료시간=수리완료dt,
                정비조치내용=입력_조치.strip() or None,
                처리상태=입력_처리상태,
                등록일시=datetime.now(),
            )
            new_id = 고장이력_저장(record)
            st.success(
                f"고장이력이 등록되었습니다.  |  ID: {new_id}  |  "
                f"{입력_제대} · {입력_체계명} · {입력_LRU}  |  "
                f"발생일시: {발생dt.strftime('%Y-%m-%d %H:%M')}"
            )
            st.balloons()


# ---------------------------------------------------------------------------
# 탭 3: 고장 현황
# ---------------------------------------------------------------------------

def 페이지_고장현황() -> None:
    _section("고장이력 조회")

    with st.expander("조회 필터", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_제대 = st.selectbox("제대구분", ["전체"] + [e.value for e in 제대구분])
        with fc2:
            f_체계 = st.selectbox("체계명", ["전체"] + [e.value for e in 체계명])
        with fc3:
            f_유형 = st.selectbox("고장유형", ["전체"] + [e.value for e in 고장유형])
        with fc4:
            f_상태 = st.selectbox("처리상태", ["전체"] + [e.value for e in 처리상태])

        dc1, dc2 = st.columns(2)
        with dc1:
            f_시작 = st.date_input("발생일 (시작)", value=date(2025, 1, 1))
        with dc2:
            f_종료 = st.date_input("발생일 (종료)", value=date.today())

    df = 고장이력_전체조회(
        제대구분=None if f_제대 == "전체" else f_제대,
        체계명=None   if f_체계 == "전체" else f_체계,
        고장유형=None if f_유형 == "전체" else f_유형,
        처리상태=None if f_상태 == "전체" else f_상태,
        날짜_시작=str(f_시작),
        날짜_종료=str(f_종료),
    )

    st.caption(f"조회 결과: {len(df)}건")

    if df.empty:
        st.info("조건에 맞는 고장이력이 없습니다.")
        return

    표시컬럼 = [
        "id", "제대구분", "발생일시", "체계명", "LRU명",
        "고장유형", "고장증상", "처리상태", "수리소요시간_h",
    ]
    st.dataframe(
        df[표시컬럼].rename(columns={"수리소요시간_h": "수리소요(h)"}),
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "id":          st.column_config.NumberColumn("ID", width="small"),
            "수리소요(h)": st.column_config.NumberColumn("수리소요(h)", format="%.1f"),
        },
    )

    st.divider()
    _section("레코드 수정 / 삭제")

    sel_id = st.number_input("레코드 ID 선택", min_value=1, step=1, value=int(df["id"].iloc[0]))
    row = 고장이력_단건조회(int(sel_id))

    if row is None:
        st.warning("해당 ID의 레코드가 없습니다.")
        return

    with st.expander(f"ID {sel_id} 상세 내용", expanded=True):
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown(f"**제대구분** {row['제대구분']}")
            st.markdown(f"**발생일시** {row['발생일시']}")
            st.markdown(f"**체계명** {row['체계명']}　**LRU명** {row['LRU명']}")
            st.markdown(f"**고장유형** {row['고장유형']}")
            st.markdown(f"**고장증상** {row['고장증상']}")
        with dc2:
            st.markdown(f"**처리상태** {row['처리상태']}")
            st.markdown(f"**수리시작** {row['수리시작시간'] or '—'}")
            st.markdown(f"**수리완료** {row['수리완료시간'] or '—'}")
            소요_h = row.get("수리소요시간_h")
            소요_표시 = f"{소요_h:.1f}h" if pd.notna(소요_h) else "—"
            st.markdown(f"**수리소요** {소요_표시}")
            st.markdown(f"**정비조치** {row['정비조치내용'] or '—'}")

    상태값목록 = [e.value for e in 처리상태]
    현재상태idx = 상태값목록.index(row["처리상태"]) if row["처리상태"] in 상태값목록 else 0

    with st.form(f"수정_폼_{sel_id}"):
        m1, m2 = st.columns(2)
        with m1:
            new_상태 = st.selectbox("처리상태 변경", 상태값목록, index=현재상태idx)
            new_수리완료 = st.text_input(
                "수리완료시간 수정",
                value=row["수리완료시간"] or "",
                placeholder="YYYY-MM-DD HH:MM:SS",
            )
        with m2:
            new_조치 = st.text_area(
                "정비조치내용 수정",
                value=row["정비조치내용"] or "",
                height=120,
            )

        bc1, bc2 = st.columns(2)
        with bc1:
            수정_제출 = st.form_submit_button("수정 저장", type="primary", use_container_width=True)
        with bc2:
            삭제_제출 = st.form_submit_button("레코드 삭제", type="secondary", use_container_width=True)

    if 수정_제출:
        수정내용: dict = {"처리상태": new_상태}
        if new_수리완료.strip():
            수정내용["수리완료시간"] = new_수리완료.strip()
        if new_조치.strip():
            수정내용["정비조치내용"] = new_조치.strip()
        고장이력_수정(int(sel_id), 수정내용)
        st.success(f"ID {sel_id} 수정 완료.")
        st.rerun()

    if 삭제_제출:
        고장이력_삭제(int(sel_id))
        st.warning(f"ID {sel_id} 삭제 완료.")
        st.rerun()


# ---------------------------------------------------------------------------
# 탭 4: 분석
# ---------------------------------------------------------------------------

def 페이지_분석() -> None:
    _section("고장 분석")

    df_all = 고장이력_전체조회()

    af1, af2 = st.columns(2)
    with af1:
        a_제대 = st.selectbox(
            "제대구분 (분석 대상)", ["전체"] + [e.value for e in 제대구분], key="a_제대"
        )
    with af2:
        a_체계 = st.selectbox(
            "체계명 (분석 대상)", ["전체"] + [e.value for e in 체계명], key="a_체계"
        )

    df = 고장이력_전체조회(
        제대구분=None if a_제대 == "전체" else a_제대,
        체계명=None   if a_체계 == "전체" else a_체계,
    )
    st.caption(f"분석 대상: {len(df)}건")
    st.divider()

    # 파레토 분석
    _section("파레토 분석")
    p1, p2 = st.columns(2)

    with p1:
        st.markdown(
            f'<p style="color:{C["muted"]};font-size:12px;margin-bottom:6px">'
            '고장유형별</p>',
            unsafe_allow_html=True,
        )
        pareto_유형 = 파레토_분석(df, "고장유형")
        fig_p1 = (
            _파레토_차트(pareto_유형, "고장유형")
            if not pareto_유형.empty else _빈_차트()
        )
        st.plotly_chart(fig_p1, use_container_width=True)

    with p2:
        st.markdown(
            f'<p style="color:{C["muted"]};font-size:12px;margin-bottom:6px">'
            'LRU별</p>',
            unsafe_allow_html=True,
        )
        pareto_LRU = 파레토_분석(df, "LRU명")
        fig_p2 = (
            _파레토_차트(pareto_LRU, "LRU명")
            if not pareto_LRU.empty else _빈_차트()
        )
        st.plotly_chart(fig_p2, use_container_width=True)

    st.divider()

    # 제대별 고장현황 비교
    _section("제대별 고장현황 비교 (처리상태)")
    현황_df = 제대별_고장현황(df_all)
    if 현황_df.empty:
        st.plotly_chart(_빈_차트(), use_container_width=True)
    else:
        fig_현황 = px.bar(
            현황_df, x="제대구분", y="건수", color="처리상태",
            barmode="stack",
            color_discrete_map=색상_처리상태,
            labels={"건수": "고장건수 (건)", "제대구분": "", "처리상태": "처리상태"},
            height=320,
        )
        fig_현황.update_layout(**_PLOT_LAYOUT)
        st.plotly_chart(fig_현황, use_container_width=True)

    st.divider()

    # 불가동 추세
    _section("불가동 추세 (월별 · 체계별)")
    불가동_df = 불가동_추세(df)
    if 불가동_df.empty:
        st.info("불가동 데이터가 없습니다.")
    else:
        fig_bd = px.line(
            불가동_df, x="연월", y="불가동건수", color="체계명",
            markers=True,
            labels={"연월": "", "불가동건수": "불가동 건수", "체계명": "체계명"},
            height=300,
        )
        fig_bd.update_traces(line=dict(width=2), marker=dict(size=6))
        fig_bd.update_layout(**_PLOT_LAYOUT, xaxis_tickangle=-40)
        st.plotly_chart(fig_bd, use_container_width=True)


def _파레토_차트(df: pd.DataFrame, 기준컬럼: str) -> go.Figure:
    """파레토 막대 + 누적 비율선 콤보 차트."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df[기준컬럼], y=df["건수"],
        name="건수",
        marker_color=C["blue"],
        marker_line_color="rgba(0,0,0,0)",
        yaxis="y1",
    ))

    fig.add_trace(go.Scatter(
        x=df[기준컬럼], y=df["누적비율_%"],
        name="누적비율",
        mode="lines+markers",
        marker=dict(color=C["orange"], size=7),
        line=dict(color=C["orange"], width=2),
        yaxis="y2",
    ))

    # 80% 기준선
    fig.add_shape(
        type="line", xref="paper", yref="y2",
        x0=0, x1=1, y0=80, y1=80,
        line=dict(color=C["muted"], width=1, dash="dash"),
    )
    fig.add_annotation(
        xref="paper", yref="y2", x=1.0, y=80,
        text="80%", showarrow=False,
        font=dict(size=11, color=C["muted"]),
        xanchor="left",
    )

    layout = dict(**_PLOT_LAYOUT)
    layout["yaxis"]  = dict(title="건수",           gridcolor=C["border"], zerolinecolor=C["border"])
    layout["yaxis2"] = dict(title="누적비율 (%)",   overlaying="y", side="right",
                            range=[0, 108], ticksuffix="%",
                            gridcolor="rgba(0,0,0,0)")
    layout["legend"] = dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)")
    layout["height"] = 340
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 메인 헤더 + 라우팅
# ---------------------------------------------------------------------------

def main() -> None:
    # 상단 헤더 — 네이티브 컴포넌트 사용 (HTML 블록 인코딩 문제 방지)
    col_title, col_info = st.columns([5, 1])
    with col_title:
        st.title("🛠️ 고장·결함 분석 대시보드")
        st.caption("Fault & Defect Analysis Dashboard  |  방위산업 IPS 보조 도구  |  정비 3제대")
    # 헤더 하단 구분선
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {C["border"]};margin:0 0 16px">',
        unsafe_allow_html=True,
    )

    메뉴 = _사이드바()

    if   메뉴 == "📊 요약 대시보드": 페이지_요약대시보드()
    elif 메뉴 == "✏️ 고장 입력":    페이지_고장입력()
    elif 메뉴 == "📋 고장 현황":    페이지_고장현황()
    elif 메뉴 == "🔍 분석":         페이지_분석()


if __name__ == "__main__":
    main()
