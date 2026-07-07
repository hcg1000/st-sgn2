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


def us_tick_round(price: float) -> float:
    """미국장 최소 호가 $0.01 반올림"""
    return round(price, 2)


def fmt_us(price: float) -> str:
    return f"${price:,.2f}"



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

    # ── 단기(5일 이내) 지표 ──
    df["MA10"] = df["Close"].rolling(10).mean()
    gain7 = delta.clip(lower=0).ewm(alpha=1 / 7, adjust=False).mean()
    loss7 = (-delta.clip(upper=0)).ewm(alpha=1 / 7, adjust=False).mean()
    df["RSI7"] = 100 - 100 / (1 + gain7 / loss7.replace(0, np.nan))
    ll5 = df["Low"].rolling(5).min()
    hh5 = df["High"].rolling(5).max()
    raw_k = 100 * (df["Close"] - ll5) / (hh5 - ll5).replace(0, np.nan)
    df["STO_K"] = raw_k.rolling(3).mean()
    df["STO_D"] = df["STO_K"].rolling(3).mean()
    df["VOL_MA5"] = df["Volume"].rolling(5).mean()

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
    prev = diff.iloc[-6:-1]
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
# 단기(5일 이내) 신호 점수화
# ─────────────────────────────────────────────

def score_short(df: pd.DataFrame):
    """단기(5일 이내) 매매 점수. 양수=매수 우위, 음수=매도 우위"""
    last = df.iloc[-1]
    rows = []

    # 1) 초단기 크로스 (MA5 vs MA10, 최근 3거래일)
    s, note = 0, "데이터 부족"
    diff = df["MA5"] - df["MA10"]
    if pd.notna(diff.iloc[-1]):
        prev = diff.iloc[-4:-1]
        if (prev <= 0).any() and diff.iloc[-1] > 0:
            s, note = 2, "MA5가 MA10 상향 돌파 (단기 반등 개시)"
        elif (prev >= 0).any() and diff.iloc[-1] < 0:
            s, note = -2, "MA5가 MA10 하향 돌파 (단기 이탈)"
        elif diff.iloc[-1] > 0:
            s, note = 1, "MA5 > MA10 유지 (단기 상승 지속)"
        else:
            s, note = -1, "MA5 < MA10 유지 (단기 하락 지속)"
    rows.append(("MA5/10 크로스", s, note))

    # 2) 단기 RSI(7)
    r = last["RSI7"]
    if pd.isna(r):
        s, note = 0, "데이터 부족"
    elif r < 20:
        s, note = 2, f"RSI7 {r:.1f} 극단 과매도 (단기 반등 기대)"
    elif r < 30:
        s, note = 1, f"RSI7 {r:.1f} 과매도권"
    elif r > 80:
        s, note = -2, f"RSI7 {r:.1f} 극단 과매수 (단기 조정 우려)"
    elif r > 70:
        s, note = -1, f"RSI7 {r:.1f} 과매수권"
    else:
        s, note = 0, f"RSI7 {r:.1f} 중립"
    rows.append(("RSI(7)", s, note))

    # 3) 스토캐스틱(5,3,3)
    s, note = 0, "중립"
    if len(df) >= 2:
        k0, k1 = df["STO_K"].iloc[-1], df["STO_K"].iloc[-2]
        d0, d1 = df["STO_D"].iloc[-1], df["STO_D"].iloc[-2]
        if pd.notna(k0) and pd.notna(d0) and pd.notna(k1) and pd.notna(d1):
            if k1 <= d1 and k0 > d0 and k0 < 30:
                s, note = 2, f"%K가 %D 상향 돌파 + 침체권({k0:.0f})"
            elif k1 >= d1 and k0 < d0 and k0 > 70:
                s, note = -2, f"%K가 %D 하향 돌파 + 과열권({k0:.0f})"
            elif k0 > d0:
                s, note = 1, f"%K > %D 유지 ({k0:.0f})"
            else:
                s, note = -1, f"%K < %D 유지 ({k0:.0f})"
    rows.append(("스토캐스틱(5,3,3)", s, note))

    # 4) 볼린저 %B
    s, note = 0, "밴드 중앙권"
    bw = last["BB_up"] - last["BB_low"]
    if pd.notna(bw) and bw > 0:
        pb = (last["Close"] - last["BB_low"]) / bw
        if pb <= 0.05:
            s, note = 1, f"%B {pb:.2f} 하단 밀착 (단기 반등 여지)"
        elif pb >= 0.95:
            s, note = -1, f"%B {pb:.2f} 상단 밀착 (단기 과열)"
        else:
            note = f"%B {pb:.2f}"
    rows.append(("볼린저 %B", s, note))

    # 5) 갭 + 단기 거래량
    s, note = 0, "특이 없음"
    if len(df) >= 2 and pd.notna(last["VOL_MA5"]) and last["VOL_MA5"] > 0:
        gap = last["Open"] / df["Close"].iloc[-2] - 1
        vr = last["Volume"] / last["VOL_MA5"]
        bullish = last["Close"] >= last["Open"]
        if gap > 0.01 and vr >= 1.5 and bullish:
            s, note = 1, f"상승 갭 {gap*100:+.1f}% + 거래량 {vr:.1f}배 (매수세)"
        elif gap < -0.01 and vr >= 1.5 and not bullish:
            s, note = -1, f"하락 갭 {gap*100:+.1f}% + 거래량 {vr:.1f}배 (투매)"
        else:
            note = f"갭 {gap*100:+.1f}%, 거래량 {vr:.1f}배"
    rows.append(("갭+거래량(5일)", s, note))

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
    """다음날 매수·매도 추정가.
    개선점: ① 도달권(±1.5 ATR) 밖 레벨 배제 → 비현실적 추정 방지
            ② 0.25 ATR 이내 다중 레벨은 평균해 '합류' 레벨로 채택 → 신뢰도 강화"""
    last = df.iloc[-1]
    C, H, L = last["Close"], last["High"], last["Low"]
    atr = last["ATR"]

    P = (H + L + C) / 3
    R1, S1 = 2 * P - L, 2 * P - H
    R2, S2 = P + (H - L), P - (H - L)

    cands = {"피벗 S1": S1, "피벗 S2": S2, "피벗 R1": R1, "피벗 R2": R2,
             "볼린저 하단": last["BB_low"], "볼린저 상단": last["BB_up"],
             "종가-0.5ATR": C - 0.5 * atr, "종가+0.5ATR": C + 0.5 * atr}
    for w in (5, 10, 20, 60, 120, 240):
        v = last.get(f"MA{w}", np.nan)
        if pd.notna(v):
            cands[f"MA{w}"] = v

    sup_all = sorted([(k, v) for k, v in cands.items() if pd.notna(v) and v < C],
                     key=lambda x: -x[1])
    res_all = sorted([(k, v) for k, v in cands.items() if pd.notna(v) and v > C],
                     key=lambda x: x[1])

    def _select(levels, side):
        if side == "sup":
            inr = [(k, v) for k, v in levels if v >= C - 1.5 * atr]
            fb1 = ("ATR 보정 지지", C - 0.6 * atr)
            fb2 = ("ATR 확장 지지", C - 1.0 * atr)
        else:
            inr = [(k, v) for k, v in levels if v <= C + 1.5 * atr]
            fb1 = ("ATR 보정 저항", C + 0.6 * atr)
            fb2 = ("ATR 확장 저항", C + 1.0 * atr)
        if not inr:
            return fb1, fb2
        base = inr[0]
        cluster = [(k, v) for k, v in inr if abs(v - base[1]) <= 0.25 * atr]
        if len(cluster) >= 2:
            lvl1 = (f"{base[0]} 합류×{len(cluster)}",
                    float(np.mean([v for _, v in cluster])))
            rest = [x for x in inr if x not in cluster]
        else:
            lvl1, rest = base, inr[1:]
        lvl2 = rest[0] if rest else fb2
        return lvl1, lvl2

    buy1, buy2 = _select(sup_all, "sup")
    sell1, sell2 = _select(res_all, "res")

    return {
        "buy1": (buy1[0], tick_round(buy1[1])),
        "buy2": (buy2[0], tick_round(buy2[1])),
        "sell1": (sell1[0], tick_round(sell1[1])),
        "sell2": (sell2[0], tick_round(sell2[1])),
        "range_low": tick_round(C - atr),
        "range_high": tick_round(C + atr),
        "pivot": tick_round(P),
        "atr": atr,
        "supports": sup_all,
        "resists": res_all,
    }


# ─────────────────────────────────────────────
# 종목명 검색 (국내장)
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def load_krx_listing():
    """KRX 전 종목 리스트 (이름→코드 매칭용, 1일 캐시)"""
    lst = fdr.StockListing("KRX")
    return lst[["Code", "Name"]].dropna()




def _flatten(d: pd.DataFrame) -> pd.DataFrame:
    if isinstance(d.columns, pd.MultiIndex):
        d = d.copy()
        d.columns = d.columns.get_level_values(0)
    return d


def fetch_ohlcv(ticker: str, years: int, is_kr: bool) -> pd.DataFrame:
    """1차 FinanceDataReader → 실패/부족 시 yfinance 폴백"""
    start = pd.Timestamp.today() - pd.DateOffset(years=years)
    df = None
    try:
        df = fdr.DataReader(ticker, start)
    except Exception:
        df = None
    if df is not None and len(df) >= 260:
        return df
    # 폴백: yfinance
    import yfinance as yf
    cands = [f"{ticker}.KS", f"{ticker}.KQ"] if is_kr else [ticker]
    for t in cands:
        try:
            d = yf.download(t, start=start.strftime("%Y-%m-%d"),
                            progress=False, auto_adjust=False)
            d = _flatten(d)
            if d is not None and len(d) >= 260:
                return d[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    return df  # 마지막 시도 결과(부족하더라도) 반환


# ─────────────────────────────────────────────
# 백테스트 (최근 1년 정확도 검증)
# ─────────────────────────────────────────────

def run_backtest(raw: pd.DataFrame, tick_round, eval_days=250, horizon=5):
    """매일 그날까지 데이터로 다음날 매수·매도 추정가를 내고, 실제 다음날 시세와 대조.
    아울러 중장기/단기 매수신호가 뜬 날의 5거래일 뒤 수익률로 신호 유효성도 검증."""
    df = compute_indicators(raw)
    n = len(df)
    start = max(260, n - eval_days - 1)
    recs, lt_rets, st_rets = [], [], []
    for i in range(start, n - 1):
        row = df.iloc[i]
        if pd.isna(row["MA240"]) or pd.isna(row["ATR"]):
            continue
        sub = df.iloc[: i + 1]
        est = estimate_prices(sub, tick_round)
        nxt = df.iloc[i + 1]
        b, s = est["buy1"][1], est["sell1"][1]
        hit_b = nxt["Low"] <= b
        hit_s = nxt["High"] >= s
        in_r = est["range_low"] <= nxt["Close"] <= est["range_high"]
        ret = np.nan
        j = i + 1 + horizon
        if hit_b and j < n:
            ret = df.iloc[j]["Close"] / b - 1
        recs.append((hit_b, hit_s, hit_b and hit_s, in_r, ret))
        # 신호 조건부 수익 (신호일 종가 → 5거래일 뒤 종가)
        j2 = i + horizon
        if j2 < n:
            base = df.iloc[j2]["Close"] / row["Close"] - 1
            if score_signals(sub)[1] >= 2:
                lt_rets.append(base)
            if score_short(sub)[1] >= 2:
                st_rets.append(base)
    a = pd.DataFrame(recs, columns=["hitB", "hitS", "both", "inR", "ret"])
    filled = a["ret"].dropna()

    def _sig(x):
        return f"{np.mean(x)*100:+.2f} ({len(x)}회)" if x else "신호 없음"

    return {
        "평가일수": len(a),
        "매수가 도달률(%)": round(a["hitB"].mean() * 100, 1),
        "매도가 도달률(%)": round(a["hitS"].mean() * 100, 1),
        "±ATR 범위 적중률(%)": round(a["inR"].mean() * 100, 1),
        "매수체결→5일 평균수익(%)": round(filled.mean() * 100, 2) if len(filled) else np.nan,
        "매수체결→5일 승률(%)": round((filled > 0).mean() * 100, 1) if len(filled) else np.nan,
        "중장기 매수신호→5일수익(%)": _sig(lt_rets),
        "단기 매수신호→5일수익(%)": _sig(st_rets),
    }


# ─────────────────────────────────────────────
# 지수 구성종목 일괄 스캔 (TOP10)
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def load_universe(market: str):
    """지수 구성종목 [(이름, 코드)] 목록"""
    if market == "KOSPI200":
        lst = None
        try:
            lst = fdr.StockListing("KOSPI200")
        except Exception:
            lst = None
        if lst is None or len(lst) < 50:
            # 폴백: KOSPI 시가총액 상위 200
            k = fdr.StockListing("KOSPI")
            if "Marcap" in k.columns:
                k = k.sort_values("Marcap", ascending=False)
            lst = k.head(200)
        ccol = "Code" if "Code" in lst.columns else (
            "Symbol" if "Symbol" in lst.columns else lst.columns[0])
        ncol = "Name" if "Name" in lst.columns else ccol
        return [(str(r[ncol]), str(r[ccol])) for _, r in lst.head(200).iterrows()]
    sp = fdr.StockListing("S&P500")
    ccol = "Symbol" if "Symbol" in sp.columns else "Code"
    return [(str(r["Name"]), str(r[ccol])) for _, r in sp.iterrows()]


def _score_one(name, tkr, raw, is_kr):
    """단일 종목 점수·추정가 산출 (데이터 부족 시 None)"""
    if raw is None or len(raw) < 260:
        return None
    d = compute_indicators(raw)
    _, lt, lt_v, _ = score_signals(d)
    _, sh, sh_v, _ = score_short(d)
    est = estimate_prices(d, krx_tick_round if is_kr else us_tick_round)
    return {"종목": name, "코드": tkr, "종가": float(d.iloc[-1]["Close"]),
            "중장기점수": lt, "중장기신호": lt_v,
            "단기점수": sh, "단기신호": sh_v,
            "매수추정가": est["buy1"][1]}


def scan_kr(universe, progress_cb=None, workers=8):
    """한국: 스레드 병렬 수집"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out, done = [], 0

    def work(item):
        name, tkr = item
        try:
            return _score_one(name, tkr, fetch_ohlcv(tkr, 2, True), True)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, it) for it in universe]
        for f in as_completed(futs):
            r = f.result()
            if r:
                out.append(r)
            done += 1
            if progress_cb:
                progress_cb(done)
    return out


def scan_us(universe, progress_cb=None, chunk=80):
    """미국: yfinance 배치 다운로드 (80종목 단위)"""
    import yfinance as yf
    out, done = [], 0
    for i in range(0, len(universe), chunk):
        part = universe[i:i + chunk]
        try:
            data = yf.download([t.replace(".", "-") for _, t in part],
                               period="2y", group_by="ticker",
                               auto_adjust=False, progress=False, threads=True)
        except Exception:
            data = None
        for name, tkr in part:
            raw = None
            if data is not None and len(data) > 0:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        raw = data[tkr.replace(".", "-")]
                    else:
                        raw = data
                    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
                except Exception:
                    raw = None
            try:
                r = _score_one(name, tkr, raw, False)
                if r:
                    out.append(r)
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done)
    return out


KR_SET = [("삼성전자", "005930"), ("SK하이닉스", "000660"),
          ("효성중공업", "298040"), ("가온전선", "000500")]
US_SET = [("NVIDIA", "NVDA"), ("Amazon", "AMZN"),
          ("Alphabet", "GOOGL"), ("JPMorgan", "JPM")]


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="매매 신호 분석기", page_icon="📈", layout="wide")
st.title("📈 주식 매매 신호 분석기")
st.caption("중장기 6지표(MA240 포함) + 단기 5지표(RSI7·스토캐스틱 등) 규칙형 분석 · 추정가는 도달권 클리핑+합류 레벨 적용 (AI 미사용)")

tab_a, tab_b, tab_c = st.tabs(["📊 종목 분석", "✅ 정확도 검증", "🏆 TOP10 스캔"])

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
                raw = fetch_ohlcv(code.strip(), years, is_kr)
            if raw is None or len(raw) < 260:
                st.error("데이터 부족 (240일선 계산에 최소 1년 이상 필요). 입력 확인 요망.")
                st.stop()

            df = compute_indicators(raw)
            rows_l, tot_l, ver_l, ic_l = score_signals(df)
            rows_s, tot_s, ver_s, ic_s = score_short(df)
            est = estimate_prices(df, tick)
            last = df.iloc[-1]
            last_date = df.index[-1].strftime("%Y-%m-%d")

            st.subheader(disp_name)
            sc1, sc2 = st.columns(2)
            sc1.metric("📅 중장기 신호", f"{ic_l} {ver_l}",
                       delta=f"점수 {tot_l:+d}", delta_color="off")
            sc2.metric("⚡ 단기(5일) 신호", f"{ic_s} {ver_s}",
                       delta=f"점수 {tot_s:+d}", delta_color="off")
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
            all_rows = ([("📅 중장기", *r) for r in rows_l]
                        + [("⚡ 단기", *r) for r in rows_s])
            st.dataframe(pd.DataFrame(all_rows,
                                      columns=["구분", "지표", "점수", "판단 근거"]),
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
        "입력한 종목에 대해 **최근 1년(약 250거래일)** 동안 매일 “그날까지의 데이터”로 "
        "다음날 매수·매도 추정가를 산출하고, 실제 다음날 시세와 대조함.\n"
        "- **매수/매도가 도달률**: 다음날 저가/고가가 추정가에 실제로 닿은 비율 (주문 체결 가능성)\n"
        "- **±ATR 범위 적중률**: 다음날 종가가 예상 등락 범위 안에 들어온 비율\n"
        "- **매수체결→5일 수익**: 매수 추정가 체결 가정, 5거래일 뒤 종가 기준 수익률·승률"
    )
    kr_in = st.text_input("🇰🇷 한국 종목명 (쉼표 구분, 비우면 생략)",
                          value="삼성전자, SK하이닉스, 효성중공업, 가온전선")
    us_in = st.text_input("🇺🇸 미국 티커 (쉼표 구분, 비우면 생략)",
                          value="NVDA, AMZN, GOOGL, JPM")
    c_a, c_b = st.columns(2)
    end_date = c_a.date_input("검증 종료일", value=pd.Timestamp.today().date(),
                              help="이 날짜까지의 최근 N거래일을 검증함")
    eval_days = int(c_b.number_input("검증 기간 (거래일 수)", min_value=20,
                                     max_value=500, value=250, step=10,
                                     help="예: 60=약 3개월, 125=약 6개월, 250=약 1년"))

    if st.button("검증 실행", type="primary"):
        universe = []
        if kr_in.strip():
            try:
                lst = load_krx_listing()
                for nm in [x.strip() for x in kr_in.split(",") if x.strip()]:
                    m = lst[lst["Name"] == nm]
                    if m.empty:
                        m = lst[lst["Name"].str.contains(nm, case=False,
                                                         na=False, regex=False)]
                    if m.empty:
                        st.warning(f"’{nm}’ 종목을 찾지 못함 — 건너뜀")
                    else:
                        universe.append(("KR", m.iloc[0]["Name"], m.iloc[0]["Code"]))
            except Exception as e:
                st.error(f"종목 리스트 로드 실패: {e}")
        for tk in [x.strip().upper() for x in us_in.split(",") if x.strip()]:
            universe.append(("US", tk, tk))

        if not universe:
            st.error("검증할 종목이 없음. 종목명 또는 티커를 입력할 것.")
            st.stop()

        results_kr, results_us = [], []
        prog = st.progress(0.0, text="백테스트 진행 중...")
        for k, (mk, name, tkr) in enumerate(universe):
            try:
                is_kr_bt = (mk == "KR")
                end_ts = pd.Timestamp(end_date)
                gap_y = max(0.0, (pd.Timestamp.today() - end_ts).days / 365)
                years_need = min(10, int(np.ceil((eval_days + 320) / 250 + gap_y)))
                raw = fetch_ohlcv(tkr, years_need, is_kr_bt)
                if raw is not None:
                    raw = raw[raw.index <= end_ts]
                need = eval_days + 262
                if raw is None or len(raw) < need:
                    st.warning(f"{name}({tkr}): 데이터 부족으로 건너뜀 "
                               f"(확보 {0 if raw is None else len(raw)}일 / 필요 {need}일 — "
                               f"240일선+검증기간 계산에 필요)")
                else:
                    tick = krx_tick_round if is_kr_bt else us_tick_round
                    r = {"종목": name, **run_backtest(raw, tick, eval_days=eval_days)}
                    (results_kr if is_kr_bt else results_us).append(r)
            except Exception as e:
                st.warning(f"{name}({tkr}) 실패: {e}")
            prog.progress((k + 1) / len(universe), text=f"{name} 완료 ({k+1}/{len(universe)})")
        prog.empty()

        if results_kr:
            st.subheader(f"🇰🇷 한국장 검증 결과 (~{end_date}, {eval_days}거래일)")
            st.dataframe(pd.DataFrame(results_kr), use_container_width=True, hide_index=True)
        if results_us:
            st.subheader(f"🇺🇸 미국장 검증 결과 (~{end_date}, {eval_days}거래일)")
            st.dataframe(pd.DataFrame(results_us), use_container_width=True, hide_index=True)
        if results_kr or results_us:
            st.caption("해석 가이드: 도달률은 “주문 체결 기회”의 빈도이고, 전략 유효성은 "
                       "**매수체결→5일 평균수익(양수 여부)과 승률(50% 초과 여부)**로 판단할 것. "
                       "±ATR 적중률은 변동폭 예측의 신뢰도임.")


# ══════════ 탭3: TOP10 스캔 ══════════
with tab_c:
    st.markdown("코스피200 / S&P500 구성종목을 일괄 스캔해 "
                "**단기·중장기 매수 점수 상위 10종목**을 뽑음.")
    m2 = st.radio("스캔 대상", ["🇰🇷 코스피200", "🇺🇸 S&P500"],
                  horizontal=True, key="scan_mkt")
    is_kr_scan = m2.startswith("🇰🇷")
    n_max = 200 if is_kr_scan else 500
    n_scan = st.slider("스캔 종목 수", 50, n_max, n_max, step=50, key="scan_n",
                       help="줄이면 빨라짐 (목록 앞쪽부터 스캔)")
    st.caption("예상 소요: 코스피200 전체 약 1분, S&P500 전체 약 1~3분")

    if st.button("스캔 실행", type="primary", key="scan_btn"):
        uni = load_universe("KOSPI200" if is_kr_scan else "S&P500")[:n_scan]
        prog = st.progress(0.0, text="스캔 중...")

        def _cb(done):
            prog.progress(min(done / len(uni), 1.0),
                          text=f"스캔 중... {done}/{len(uni)}")

        res = scan_kr(uni, _cb) if is_kr_scan else scan_us(uni, _cb)
        prog.empty()
        st.session_state["scan_res"] = (m2, len(uni), res)

    if "scan_res" in st.session_state:
        m_lbl, n_uni, res = st.session_state["scan_res"]
        if not res:
            st.error("스캔 결과 없음 (데이터 수집 실패). 잠시 후 재시도 요망.")
        else:
            money = fmt if m_lbl.startswith("🇰🇷") else fmt_us
            rdf = pd.DataFrame(res)
            rdf["종가"] = rdf["종가"].map(money)
            rdf["매수추정가"] = rdf["매수추정가"].map(money)
            st.caption(f"{m_lbl} {n_uni}종목 중 {len(res)}종목 분석 완료 "
                       f"(제외분은 상장 1년 미만 등 데이터 부족)")

            top_s = rdf.sort_values(["단기점수", "중장기점수"],
                                    ascending=False).head(10).reset_index(drop=True)
            top_l = rdf.sort_values(["중장기점수", "단기점수"],
                                    ascending=False).head(10).reset_index(drop=True)
            top_s.insert(0, "순위", range(1, len(top_s) + 1))
            top_l.insert(0, "순위", range(1, len(top_l) + 1))

            st.subheader("⚡ 단기(5일) 매수 TOP 10")
            st.dataframe(top_s[["순위", "종목", "코드", "단기점수", "단기신호",
                                "중장기점수", "종가", "매수추정가"]],
                         use_container_width=True, hide_index=True)
            st.subheader("📅 중장기 매수 TOP 10")
            st.dataframe(top_l[["순위", "종목", "코드", "중장기점수", "중장기신호",
                                "단기점수", "종가", "매수추정가"]],
                         use_container_width=True, hide_index=True)
            st.warning("⚠️ 점수 상위는 규칙 기반 기술적 순위일 뿐 매수 권유가 아님. "
                       "종목별 상세는 ‘종목 분석’ 탭에서 확인하고, "
                       "투자 판단과 손실 책임은 이용자 본인에게 있음.")
