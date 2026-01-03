import streamlit as st
import ccxt
import pandas as pd
import requests
import plotly.graph_objects as go

# =====================================================
# CONFIG — OPSI A PRO v3 (STEP 6A)
# =====================================================
ENTRY_TF = "4h"
SR_TF = "1d"

LIMIT_4H = 200
LIMIT_1D = 200
BACKTEST_LIMIT = 600
MAX_FORWARD = 80

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28

SR_LOOKBACK = 5
ZONE_BUFFER = 0.008

MIN_USDT_VOLUME = 2_000_000

VALID_CANDLES = {
    "Bullish Engulfing",
    "Hammer",
    "Strong Bullish",
    "Normal"
}

# =====================================================
# 🔥 STEP 6A — TP OPTIMIZATION (ONLY CHANGE)
# =====================================================
TP1_R = 0.8
TP2_R = 2.0
TP1_PORTION = 0.3
TP2_PORTION = 0.7
SOFT_BE_R = 0.0

# =====================================================
# MARKET SYMBOL FETCHER
# =====================================================
@st.cache_data(ttl=300)
def get_liquid_symbols(min_vol):
    url = "https://www.okx.com/api/v5/market/tickers"
    r = requests.get(url, params={"instType": "SPOT"}, timeout=15)
    r.raise_for_status()
    return [
        d["instId"]
        for d in r.json()["data"]
        if d["instId"].endswith("-USDT")
        and float(d["volCcy24h"]) >= min_vol
    ]

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h, l, c = df.high, df.low, df.close
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (h + l) / 2

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    stl = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)

    stl.iloc[0] = upper.iloc[0]
    trend.iloc[0] = -1

    for i in range(1, len(df)):
        if c.iloc[i] > stl.iloc[i - 1]:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i - 1])
            trend.iloc[i] = 1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i - 1])
            trend.iloc[i] = -1

    return stl, trend

def volume_oscillator(v, f, s):
    ef = v.ewm(span=f, adjust=False).mean()
    es = v.ewm(span=s, adjust=False).mean()
    return (ef - es) / es * 100

# =====================================================
# PRICE ACTION
# =====================================================
def detect_candle(df):
    o, h, l, c = df.open, df.high, df.low, df.close
    po, pc = o.shift(1), c.shift(1)
    body = abs(c - o)
    rng = h - l

    if c.iloc[-1] > o.iloc[-1] and pc.iloc[-1] < po.iloc[-1] and c.iloc[-1] > po.iloc[-1]:
        return "Bullish Engulfing"

    if c.iloc[-1] > o.iloc[-1] and (o.iloc[-1] - l.iloc[-1]) > 2 * body.iloc[-1]:
        return "Hammer"

    if rng.iloc[-1] > 0 and body.iloc[-1] / rng.iloc[-1] > 0.65 and c.iloc[-1] > o.iloc[-1]:
        return "Strong Bullish"

    return "Normal"

# =====================================================
# SUPPORT
# =====================================================
def find_support(df, lb):
    supports = []
    for i in range(lb, len(df) - lb):
        if df.low.iloc[i] == min(df.low.iloc[i - lb:i + lb + 1]):
            supports.append(df.low.iloc[i])
    return sorted(set(supports))

# =====================================================
# ENTRY VALIDATION (STEP 4B BASELINE)
# =====================================================
def valid_entry(df, stl, trend, vo):
    return trend.iloc[-1] == 1 and vo.iloc[-1] >= 0

# =====================================================
# TRADE BUILDER — DAILY EMA200 FILTER (CORE EDGE)
# =====================================================
def build_trade_opsi_a_v3(df4h, df1d):

    ema200_d = df1d.close.ewm(span=200, adjust=False).mean()
    if df1d.close.iloc[-1] < ema200_d.iloc[-1]:
        return None

    entry = df4h.close.iloc[-1]
    supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
    if not supports:
        return None

    sl = max(supports) * (1 - ZONE_BUFFER)
    risk = entry - sl
    if risk <= 0:
        return None

    tp1 = entry + risk * TP1_R
    tp2 = entry + risk * TP2_R

    return entry, sl, tp1, tp2

# =====================================================
# BACKTEST
# =====================================================
def backtest_symbol(okx, symbol):
    df = pd.DataFrame(
        okx.fetch_ohlcv(symbol, ENTRY_TF, limit=BACKTEST_LIMIT),
        columns=["t","open","high","low","close","volume"]
    )
    df1d = pd.DataFrame(
        okx.fetch_ohlcv(symbol, SR_TF, limit=LIMIT_1D),
        columns=["t","open","high","low","close","volume"]
    )

    trades = []

    for i in range(120, len(df) - MAX_FORWARD):
        slice_df = df.iloc[:i + 1]
        stl, trend = supertrend(slice_df, ATR_PERIOD, MULTIPLIER)
        vo = volume_oscillator(slice_df.volume, VO_FAST, VO_SLOW)

        if not valid_entry(slice_df, stl, trend, vo):
            continue

        if detect_candle(slice_df) not in VALID_CANDLES:
            continue

        trade = build_trade_opsi_a_v3(slice_df, df1d)
        if not trade:
            continue

        entry, sl, tp1, tp2 = trade
        hit_tp1 = False
        rr = None

        for j in range(i + 2, min(i + MAX_FORWARD, len(df))):
            if not hit_tp1 and df.high.iloc[j] >= tp1:
                hit_tp1 = True
                continue

            if hit_tp1 and df.high.iloc[j] >= tp2:
                rr = TP1_PORTION * TP1_R + TP2_PORTION * TP2_R
                break

            if not hit_tp1 and df.low.iloc[j] <= sl:
                rr = -1
                break

        if rr is not None:
            trades.append({"RR": rr, "Win": rr > 0})

    return pd.DataFrame(trades)

# =====================================================
# EQUITY CURVE
# =====================================================
def build_equity_curve(rr):
    equity = rr.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return equity, drawdown

# =====================================================
# UI — BACKTEST ONLY
# =====================================================
st.set_page_config("OPSI A PRO v3 — STEP 6A", layout="wide")
st.title("🚀 OPSI A PRO v3 — STEP 6A (TP Optimization)")

okx = ccxt.okx({"enableRateLimit": True, "timeout": 30000})

if st.button("🧪 Run Backtest"):
    symbols = get_liquid_symbols(MIN_USDT_VOLUME)
    all_bt = []

    with st.spinner("Running backtest..."):
        for s in symbols:
            try:
                bt = backtest_symbol(okx, s)
                if not bt.empty:
                    all_bt.append(bt)
            except:
                pass

    if all_bt:
        bt = pd.concat(all_bt, ignore_index=True)
        equity, dd = build_equity_curve(bt["RR"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", len(bt))
        c2.metric("Winrate %", round(bt["Win"].mean() * 100, 2))
        c3.metric("Avg RR", round(bt["RR"].mean(), 2))
        c4.metric("Max DD (R)", round(dd.min(), 2))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=equity, name="Equity", mode="lines"))
        fig.add_trace(go.Scatter(y=dd, name="Drawdown", mode="lines", line=dict(dash="dot")))

        fig.update_layout(
            title="📈 Equity Curve & Drawdown (R-based)",
            template="plotly_dark",
            height=520
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No trades found")
