# -*- coding: utf-8 -*-
"""
한국장 주식 매매 신호 분석기 (지표 기반, AI 미사용)
- 이동평균 5/20/60/120/240일, RSI(14), MACD(12,26,9), 볼린저밴드(20,2σ), ATR(14), 거래량
- 지표 점수화 → 매수/관망/매도 신호
- 피벗 포인트 + 지지/저항 레벨 → 다음날 매수·매도 추정가 산출
실행: streamlit run stock_signal_kr.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def krx_tick_round(price: float) -> float:
    """KRX 호가 단위로 반올림"""
    if price < 2000:
        tick = 1
    elif price < 5000:
        tick = 5
    elif price < 20000:
        tick = 10
    elif price < 50000:
        tick = 50
    elif price < 200000:
        tick = 100
    elif price < 500000:
        tick = 500
    else:
        tick = 1000
    return round(price / tick) * tick


def fmt(price: float) -> str:
    return f"{price:,.0f}원"


# ─────────────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 이동평균 (240일선 포함)
    for w in (5, 20, 60, 120, 240):
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    # RSI(14) - Wilder 방식
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD(12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]

    # 볼린저밴드(20, 2σ)
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_up"] = mid + 2 * std
    df["BB_low"] = mid - 2 * std

    # ATR(14)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # 거래량 20일 평균
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    return df


# ─────────────────────────────────────────────
# 신호 점수화
# ─────────────────────────────────────────────

def score_signals(df: pd.DataFrame):
    """지표별 점수와 근거를 반환. 양수=매수 우위, 음수=매도 우위"""
    last = df.iloc[-1]
    rows = []

    # 1) 이동평균 배열 (정배열/역배열, 240일선 위치)
    mas = [last[f"MA{w}"] for w in (5, 20, 60, 120, 240)]
    s = 0
    if all(pd.notna(mas)):
        if last["Close"] > last["MA240"]:
            s += 1
            note = "종가가 240일선 위 (장기 상승 추세)"
        else:
            s -= 1
            note = "종가가 240일선 아래 (장기 하락 추세)"
        if mas[0] > mas[1] > mas[2] > mas[3] > mas[4]:
            s += 2
            note += " / 완전 정배열"
        elif mas[0] < mas[1] < mas[2] < mas[3] < mas[4]:
            s -= 2
            note += " / 완전 역배열"
    else:
        note = "240일선 계산에 데이터 부족"
    rows.append(("이동평균 배열", s, note))

    # 2) 골든/데드 크로스 (MA20 vs MA60, 최근 5거래일)
    s = 0
    note = "최근 교차 없음"
    diff = df["MA20"] - df["MA60"]
    recent, prev = diff.iloc[-5:], diff.iloc[-6:-1]
    if pd.notna(diff.iloc[-1]):
        if (prev <= 0).any() and diff.iloc[-1] > 0:
            s, note = 2, "MA20이 MA60 상향 돌파 (골든크로스)"
        elif (prev >= 0).any() and diff.iloc[-1] < 0:
            s, note = -2, "MA20이 MA60 하향 돌파 (데드크로스)"
        elif diff.iloc[-1] > 0:
            s, note = 1, "MA20 > MA60 유지 (중기 상승)"
        else:
            s, note = -1, "MA20 < MA60 유지 (중기 하락)"
    rows.append(("골든/데드크로스", s, note))

    # 3) RSI
    r = last["RSI"]
    if r < 30:
        s, note = 2, f"RSI {r:.1f} 과매도 (반등 가능)"
    elif r < 40:
        s, note = 1, f"RSI {r:.1f} 매도 우위 완화 구간"
    elif r > 70:
        s, note = -2, f"RSI {r:.1f} 과매수 (조정 가능)"
    elif r > 60:
        s, note = -1, f"RSI {r:.1f} 매수 과열 접근"
    else:
        s, note = 0, f"RSI {r:.1f} 중립"
    rows.append(("RSI(14)", s, note))

    # 4) MACD
    s = 0
    note = "중립"
    h_now, h_prev = df["MACD_hist"].iloc[-1], df["MACD_hist"].iloc[-2]
    if pd.notna(h_now) and pd.notna(h_prev):
        if h_prev <= 0 < h_now:
            s, note = 2, "MACD 시그널 상향 돌파 (매수 전환)"
        elif h_prev >= 0 > h_now:
            s, note = -2, "MACD 시그널 하향 돌파 (매도 전환)"
        elif h_now > 0 and h_now > h_prev:
            s, note = 1, "MACD 히스토그램 양(+) 확대 (상승 모멘텀)"
        elif h_now < 0 and h_now < h_prev:
            s, note = -1, "MACD 히스토그램 음(-) 확대 (하락 모멘텀)"
    rows.append(("MACD(12,26,9)", s, note))

    # 5) 볼린저밴드
    s = 0
    note = "밴드 내 중립"
    if pd.notna(last["BB_low"]):
        if last["Close"] <= last["BB_low"]:
            s, note = 1, "하단 밴드 이탈 (과매도, 기술적 반등 여지)"
        elif last["Close"] >= last["BB_up"]:
            s, note = -1, "상단 밴드 이탈 (과열, 조정 여지)"
    rows.append(("볼린저밴드(20,2σ)", s, note))

    # 6) 거래량
    s = 0
    note = "평균 수준"
    if pd.notna(last["VOL_MA20"]) and last["VOL_MA20"] > 0:
        ratio = last["Volume"] / last["VOL_MA20"]
        bullish = last["Close"] >= last["Open"]
        if ratio >= 1.5 and bullish:
            s, note = 1, f"거래량 평균의 {ratio:.1f}배 + 양봉 (매수세 유입)"
        elif ratio >= 1.5 and not bullish:
            s, note = -1, f"거래량 평균의 {ratio:.1f}배 + 음봉 (매도세 출회)"
        else:
            note = f"거래량 평균의 {ratio:.1f}배"
    rows.append(("거래량", s, note))

    total = sum(r[1] for r in rows)
    if total >= 5:
        verdict, color = "강력 매수", "🟢"
    elif total >= 2:
        verdict, color = "매수", "🟢"
    elif total <= -5:
        verdict, color = "강력 매도", "🔴"
    elif total <= -2:
        verdict, color = "매도", "🔴"
    else:
        verdict, color = "관망", "🟡"
    return rows, total, verdict, color


# ─────────────────────────────────────────────
# 다음날 매수/매도 추정가 (지지/저항 레벨)
# ─────────────────────────────────────────────

def estimate_prices(df: pd.DataFrame, tick_round):
    last = df.iloc[-1]
    C, H, L = last["Close"], last["High"], last["Low"]
    atr = last["ATR"]

    # 피벗 포인트
    P = (H + L + C) / 3
    R1, S1 = 2 * P - L, 2 * P - H
    R2, S2 = P + (H - L), P - (H - L)

    # 지지 후보: 현재가 아래에 있는 레벨들
    sup_cand = {"피벗 S1": S1, "피벗 S2": S2, "볼린저 하단": last["BB_low"],
                "종가-0.5ATR": C - 0.5 * atr}
    for w in (5, 20, 60, 120, 240):
        v = last[f"MA{w}"]
        if pd.notna(v) and v < C:
            sup_cand[f"MA{w}"] = v
    supports = sorted(
        [(k, v) for k, v in sup_cand.items() if pd.notna(v) and v < C],
        key=lambda x: -x[1],
    )

    # 저항 후보: 현재가 위에 있는 레벨들
    res_cand = {"피벗 R1": R1, "피벗 R2": R2, "볼린저 상단": last["BB_up"],
                "종가+0.5ATR": C + 0.5 * atr}
    for w in (5, 20, 60, 120, 240):
        v = last[f"MA{w}"]
        if pd.notna(v) and v > C:
            res_cand[f"MA{w}"] = v
    resists = sorted(
        [(k, v) for k, v in res_cand.items() if pd.notna(v) and v > C],
        key=lambda x: x[1],
    )

    buy1 = supports[0] if supports else ("종가-0.5ATR", C - 0.5 * atr)
    buy2 = supports[1] if len(supports) > 1 else ("종가-1.0ATR", C - atr)
    sell1 = resists[0] if resists else ("종가+0.5ATR", C + 0.5 * atr)
    sell2 = resists[1] if len(resists) > 1 else ("종가+1.0ATR", C + atr)

    out = {
        "buy1": (buy1[0], tick_round(buy1[1])),
        "buy2": (buy2[0], tick_round(buy2[1])),
        "sell1": (sell1[0], tick_round(sell1[1])),
        "sell2": (sell2[0], tick_round(sell2[1])),
        "range_low": tick_round(C - atr),
        "range_high": tick_round(C + atr),
        "pivot": tick_round(P),
        "atr": atr,
        "supports": supports,
        "resists": resists,
    }
    return out



# ─────────────────────────────────────────────
# 종목명 검색 (국내장)
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def load_krx_listing():
    """KRX 전 종목 리스트 (이름→코드 매칭용, 1일 캐시)"""
    lst = fdr.StockListing("KRX")
    return lst[["Code", "Name"]].dropna()


# ─────────────────────────────────────────────
# 백테스트 (최근 1년 정확도 검증)
# ─────────────────────────────────────────────

def run_backtest(raw: pd.DataFrame, tick_round, eval_days=250, horizon=5):
    """매일 그날까지 데이터로 다음날 매수·매도 추정가를 내고, 실제 다음날 시세와 대조"""
    df = compute_indicators(raw)
    n = len(df)
    start = max(260, n - eval_days - 1)
    recs = []
    for i in range(start, n - 1):
        row = df.iloc[i]
        if pd.isna(row["MA240"]) or pd.isna(row["ATR"]):
            continue
        est = estimate_prices(df.iloc[: i + 1], tick_round)
        nxt = df.iloc[i + 1]
        b, s = est["buy1"][1], est["sell1"][1]
        hit_b = nxt["Low"] <= b          # 다음날 저가가 매수 추정가에 도달
        hit_s = nxt["High"] >= s         # 다음날 고가가 매도 추정가에 도달
        in_r = est["range_low"] <= nxt["Close"] <= est["range_high"]
        ret = np.nan
        j = i + 1 + horizon
        if hit_b and j < n:              # 매수가 체결됐다면 5거래일 뒤 종가 수익률
            ret = df.iloc[j]["Close"] / b - 1
        recs.append((hit_b, hit_s, hit_b and hit_s, in_r, ret))
    a = pd.DataFrame(recs, columns=["hitB", "hitS", "both", "inR", "ret"])
    filled = a["ret"].dropna()
    return {
        "평가일수": len(a),
        "매수가 도달률(%)": round(a["hitB"].mean() * 100, 1),
        "매도가 도달률(%)": round(a["hitS"].mean() * 100, 1),
        "당일 왕복 도달률(%)": round(a["both"].mean() * 100, 1),
        "±ATR 범위 적중률(%)": round(a["inR"].mean() * 100, 1),
        "매수체결→5일 평균수익(%)": round(filled.mean() * 100, 2) if len(filled) else np.nan,
        "매수체결→5일 승률(%)": round((filled > 0).mean() * 100, 1) if len(filled) else np.nan,
    }


KR_SET = [("삼성전자", "005930"), ("SK하이닉스", "000660"),
          ("효성중공업", "298040"), ("가온전선", "000500")]
US_SET = [("NVIDIA", "NVDA"), ("Amazon", "AMZN"),
          ("Alphabet", "GOOGL"), ("JPMorgan", "JPM")]


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="매매 신호 분석기", page_icon="📈", layout="wide")
st.title("📈 주식 매매 신호 분석기")
st.caption("이동평균(5/20/60/120/240) · RSI · MACD · 볼린저밴드 · ATR · 거래량 기반 규칙형 분석 (AI 미사용)")

tab_a, tab_b = st.tabs(["📊 종목 분석", "✅ 정확도 검증 (최근 1년)"])

# ══════════ 탭1: 종목 분석 ══════════
with tab_a:
    with st.sidebar:
        st.header("설정")
        market = st.radio("시장 선택", ["🇰🇷 한국장", "🇺🇸 미국장"], horizontal=True)
        is_kr = market.startswith("🇰🇷")
        code, disp_name = None, ""
        if is_kr:
            q = st.text_input("종목명 검색", value="삼성전자",
                              help="이름 일부만 입력해도 됨 (예: 효성, 하이닉스)")
            if q.strip():
                try:
                    lst = load_krx_listing()
                    m = lst[lst["Name"].str.contains(q.strip(), case=False,
                                                     na=False, regex=False)]
                    exact = m[m["Name"] == q.strip()]
                    m = pd.concat([exact, m]).drop_duplicates().head(20)
                    if m.empty:
                        st.warning("검색 결과 없음. 다른 이름으로 시도할 것.")
                    else:
                        opts = [f"{r.Name} ({r.Code})" for r in m.itertuples()]
                        sel = st.selectbox("종목 선택", opts)
                        disp_name = sel.rsplit(" (", 1)[0]
                        code = sel.rsplit("(", 1)[1].rstrip(")")
                except Exception as e:
                    st.error(f"종목 리스트 로드 실패: {e}")
        else:
            code = st.text_input("티커", value="AAPL", help="예: AAPL, NVDA, JPM")
            disp_name = code
        years = st.slider("데이터 기간(년)", 2, 5, 3)
        run = st.button("분석 실행", type="primary", use_container_width=True)

    if run and code:
        tick = krx_tick_round if is_kr else us_tick_round
        money = fmt if is_kr else fmt_us
        up_c, dn_c = ("red", "blue") if is_kr else ("green", "red")
        try:
            with st.spinner("일봉 데이터 수집 중..."):
                start = pd.Timestamp.today() - pd.DateOffset(years=years)
                raw = fdr.DataReader(code.strip(), start)
            if raw is None or len(raw) < 260:
                st.error("데이터 부족 (240일선 계산에 최소 1년 이상 필요). 입력 확인 요망.")
                st.stop()

            df = compute_indicators(raw)
            rows, total, verdict, icon = score_signals(df)
            est = estimate_prices(df, tick)
            last = df.iloc[-1]
            last_date = df.index[-1].strftime("%Y-%m-%d")

            st.subheader(f"{icon} {disp_name} — 종합 신호: **{verdict}** (점수 {total:+d})")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"기준 종가 ({last_date})", money(last["Close"]))
            c2.metric("다음날 매수 추정가 (1차 지지)", money(est["buy1"][1]),
                      delta=f"{(est['buy1'][1]/last['Close']-1)*100:+.2f}%")
            c3.metric("다음날 매도 추정가 (1차 저항)", money(est["sell1"][1]),
                      delta=f"{(est['sell1'][1]/last['Close']-1)*100:+.2f}%")
            c4, c5, c6 = st.columns(3)
            c4.metric("예상 등락 범위 (±1 ATR)",
                      f"{money(est['range_low'])} ~ {money(est['range_high'])}")
            c5.metric("2차 매수 추정가", f"{money(est['buy2'][1])} ({est['buy2'][0]})")
            c6.metric("2차 매도 추정가", f"{money(est['sell2'][1])} ({est['sell2'][0]})")
            st.info(f"매수 근거: **{est['buy1'][0]}** 지지 / 매도 근거: **{est['sell1'][0]}** 저항 / "
                    f"피벗 {money(est['pivot'])}, ATR {money(est['atr'])}")

            st.subheader("지표별 판단 근거")
            st.dataframe(pd.DataFrame(rows, columns=["지표", "점수", "판단 근거"]),
                         use_container_width=True, hide_index=True)

            with st.expander("지지/저항 레벨 상세"):
                lc, rc = st.columns(2)
                lc.markdown("**지지선 (현재가 아래)**")
                lc.dataframe(pd.DataFrame([(k, money(tick(v))) for k, v in est["supports"]],
                                          columns=["레벨", "가격"]),
                             hide_index=True, use_container_width=True)
                rc.markdown("**저항선 (현재가 위)**")
                rc.dataframe(pd.DataFrame([(k, money(tick(v))) for k, v in est["resists"]],
                                          columns=["레벨", "가격"]),
                             hide_index=True, use_container_width=True)

            d = df.iloc[-260:]
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=(f"{disp_name} 일봉 + 이동평균", "RSI(14)", "MACD"))
            fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                         low=d["Low"], close=d["Close"],
                                         increasing_line_color=up_c,
                                         decreasing_line_color=dn_c, name="일봉"), row=1, col=1)
            for ma, c in {"MA5": "#999", "MA20": "#f0a", "MA60": "#0a0",
                          "MA120": "#fa0", "MA240": "#00f"}.items():
                fig.add_trace(go.Scatter(x=d.index, y=d[ma],
                                         line=dict(color=c, width=1.2), name=ma), row=1, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=d["RSI"],
                                     line=dict(color="#a0f", width=1), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            fig.add_trace(go.Bar(x=d.index, y=d["MACD_hist"], name="MACD Hist",
                                 marker_color=np.where(d["MACD_hist"] >= 0, up_c, dn_c)),
                          row=3, col=1)
            fig.update_layout(height=760, xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h", y=1.06),
                              margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.warning("⚠️ 신호·추정가는 과거 데이터 기반 기술적 참고 지표일 뿐 투자 권유가 아님. "
                       "투자 판단과 손실 책임은 이용자 본인에게 있음.")
        except Exception as e:
            st.error(f"오류 발생: {e}")
    elif not run:
        st.markdown("사이드바(모바일: 좌측 상단 `>` )에서 **시장·종목**을 고르고 `분석 실행`을 누르면 "
                    "신호와 다음날 매수·매도 추정가가 표시됨.")

# ══════════ 탭2: 정확도 검증 ══════════
with tab_b:
    st.markdown(
        "지정 종목에 대해 **최근 1년(약 250거래일)** 동안 매일 \"그날까지의 데이터\"로 "
        "다음날 매수·매도 추정가를 산출하고, 실제 다음날 시세와 대조함.\n"
        "- **매수/매도가 도달률**: 다음날 저가/고가가 추정가에 실제로 닿은 비율 (주문 체결 가능성)\n"
        "- **±ATR 범위 적중률**: 다음날 종가가 예상 등락 범위 안에 들어온 비율\n"
        "- **매수체결→5일 수익**: 매수 추정가에 체결됐다고 가정, 5거래일 뒤 종가 기준 수익률과 승률"
    )
    if st.button("검증 실행 (한국 4종목 + 미국 4종목)", type="primary"):
        results_kr, results_us = [], []
        prog = st.progress(0.0, text="백테스트 진행 중...")
        universe = [("KR", n, c) for n, c in KR_SET] + [("US", n, c) for n, c in US_SET]
        for k, (mk, name, tkr) in enumerate(universe):
            try:
                start = pd.Timestamp.today() - pd.DateOffset(years=3)
                raw = fdr.DataReader(tkr, start)
                tick = krx_tick_round if mk == "KR" else us_tick_round
                r = run_backtest(raw, tick)
                r = {"종목": name, **r}
                (results_kr if mk == "KR" else results_us).append(r)
            except Exception as e:
                st.warning(f"{name}({tkr}) 실패: {e}")
            prog.progress((k + 1) / len(universe),
                          text=f"{name} 완료 ({k+1}/{len(universe)})")
        prog.empty()
        if results_kr:
            st.subheader("🇰🇷 한국장 검증 결과")
            st.dataframe(pd.DataFrame(results_kr), use_container_width=True, hide_index=True)
        if results_us:
            st.subheader("🇺🇸 미국장 검증 결과")
            st.dataframe(pd.DataFrame(results_us), use_container_width=True, hide_index=True)
        st.caption("해석 가이드: 도달률이 높다고 무조건 좋은 게 아님 — 매수가 도달은 \"주문이 체결될 기회\"를 뜻하고, "
                   "전략의 유효성은 **매수체결→5일 평균수익과 승률**로 판단할 것. "
                   "±ATR 적중률은 변동폭 예측의 신뢰도를 나타냄.")
