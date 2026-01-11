import streamlit as st
import ccxt
import pandas as pd
import requests
import time
import os
import random
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

# =====================================================
# CONFIG — OPSI A PRO v3.2
# =====================================================
ENTRY_TF = "4h"
SR_TF = "1d"

LIMIT_4H = 200
LIMIT_1D = 200

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28
VO_MIN = 5

SR_LOOKBACK = 5
ZONE_BUFFER = 0.008

MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

VALID_CANDLES = {
    "Bullish Engulfing",
    "Hammer",
    "Strong Bullish",
    "Normal"
}

TP1_R = 0.8
TP2_R = 2.0

# =====================================================
# FILE PATH (ANTI RESET STREAMLIT)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_LOG_FILE = os.path.join(BASE_DIR, "signal_history.csv")

# =====================================================
# TIMEZONE WIB
# =====================================================
WIB = timezone(timedelta(hours=7))

def now_wib():
    return datetime.now(timezone.utc).astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# PRIORITY
# =====================================================
PAIR_PRIORITY = {
    "BCH-USDT": 5,
    "WLFI-USDT": 4,
    "ZEC-USDT": 3,
    "PEPE-USDT": 2
}

# =====================================================
# CCXT (CACHE)
# =====================================================
@st.cache_resource
def get_okx():
    return ccxt.okx({"enableRateLimit": True})

# =====================================================
# SIGNAL HISTORY
# =====================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        df = pd.DataFrame(columns=[
            "Time","Symbol","Candle",
            "Entry","SL","TP1","TP2",
            "Priority","Rating","Status"
        ])
        df.to_csv(SIGNAL_LOG_FILE, index=False)
        return df
    return pd.read_csv(SIGNAL_LOG_FILE)

def has_open_signal(symbol):
    df = load_signal_history()
    return ((df["Symbol"] == symbol) & (df["Status"] == "OPEN")).any()

def save_signal(signal):
    df = load_signal_history()
    df = pd.concat([df, pd.DataFrame([signal])], ignore_index=True)
    df.to_csv(SIGNAL_LOG_FILE, index=False)

# =====================================================
# MARKET SYMBOL FETCHER
# =====================================================
@st.cache_data(ttl=300)
def get_liquid_symbols(min_vol):
    url = "https://www.okx.com/api/v5/market/tickers"
    r = requests.get(url, params={"instType": "SPOT"}, timeout=15)
    r.raise_for_status()

    symbols = [
        d["instId"] for d in r.json()["data"]
        if d["instId"].endswith("-USDT")
        and float(d["volCcy24h"]) >= min_vol
    ]
    return random.sample(symbols, min(MAX_SCAN_SYMBOLS, len(symbols)))

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h, l, c = df.high, df.low, df.close
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (h + l) / 2

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    stl = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)

    trend.iloc[0] = 1
    stl.iloc[0] = lower.iloc[0]

    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = 1 if c.iloc[i] > stl.iloc[i] else -1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = -1 if c.iloc[i] < stl.iloc[i] else 1

    return stl, trend

def volume_oscillator(v, f, s):
    fast = v.ewm(span=f).mean()
    slow = v.ewm(span=s).mean()
    return (fast - slow) / slow * 100

# =====================================================
# PRICE ACTION
# =====================================================
def detect_candle(df):
    o, h, l, c = df.open, df.high, df.low, df.close
    body = abs(c - o)
    rng = h - l

    if rng.iloc[-1] < df.high.iloc[-20:].mean() * 0.3:
        return "Normal"

    if c.iloc[-1] > o.iloc[-1] and c.iloc[-2] < o.iloc[-2] and c.iloc[-1] > o.iloc[-2]:
        return "Bullish Engulfing"

    if c.iloc[-1] > o.iloc[-1] and (o.iloc[-1] - l.iloc[-1]) > 2 * body.iloc[-1]:
        return "Hammer"

    if body.iloc[-1] / rng.iloc[-1] > 0.65 and c.iloc[-1] > o.iloc[-1]:
        return "Strong Bullish"

    return "Normal"

# =====================================================
# SUPPORT DAILY (CLUSTERED)
# =====================================================
def find_support(df, lb):
    raw = []
    for i in range(lb, len(df) - lb):
        if df.low.iloc[i] == min(df.low.iloc[i-lb:i+lb+1]):
            raw.append(df.low.iloc[i])

    raw = sorted(set(raw))
    filtered = []
    for s in raw:
        if not filtered or abs(s - filtered[-1]) / s > 0.01:
            filtered.append(s)
    return filtered

# =====================================================
# CHART PREVIEW
# =====================================================
def render_chart(df, stl, signal):
    fig = go.Figure()

    fig.add_candlestick(
        x=df.index,
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        name="Price"
    )

    fig.add_trace(go.Scatter(
        x=df.index,
        y=stl,
        mode="lines",
        name="Supertrend",
        line=dict(color="lime", width=1)
    ))

    fig.add_hline(y=signal["Entry"], line_dash="dot", line_color="cyan", annotation_text="Entry")
    fig.add_hline(y=signal["SL"], line_dash="dash", line_color="red", annotation_text="SL")
    fig.add_hline(y=signal["TP1"], line_dash="dot", line_color="orange", annotation_text="TP1")
    fig.add_hline(y=signal["TP2"], line_dash="dot", line_color="purple", annotation_text="TP2")

    fig.update_layout(
        height=450,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False
    )

    return fig

# =====================================================
# SIGNAL CHECK
# =====================================================
def check_signal(okx, symbol):
    if has_open_signal(symbol):
        return None

    df4h = pd.DataFrame(
        okx.fetch_ohlcv(symbol, ENTRY_TF, limit=LIMIT_4H),
        columns=["t","open","high","low","close","volume"]
    )

    df1d = pd.DataFrame(
        okx.fetch_ohlcv(symbol, SR_TF, limit=LIMIT_1D),
        columns=["t","open","high","low","close","volume"]
    )

    stl, trend = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
    vo = volume_oscillator(df4h.volume, VO_FAST, VO_SLOW)
    candle = detect_candle(df4h)

    if trend.iloc[-1] != 1 or vo.iloc[-1] < VO_MIN or candle not in VALID_CANDLES:
        return None

    ema200 = df1d.close.ewm(span=200).mean()
    if ema200.isna().iloc[-1] or df1d.close.iloc[-1] < ema200.iloc[-1]:
        return None

    entry = df4h.close.iloc[-1]
    supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
    if not supports:
        return None

    sl = max(supports) * (1 - ZONE_BUFFER)
    if entry - sl < entry * 0.002:
        return None

    risk = entry - sl
    tp1 = entry + risk * TP1_R
    tp2 = entry + risk * TP2_R

    priority = PAIR_PRIORITY.get(symbol, 3)

    signal = {
        "Time": now_wib(),
        "Symbol": symbol,
        "Candle": candle,
        "Entry": round(entry, 8),
        "SL": round(sl, 8),
        "TP1": round(tp1, 8),
        "TP2": round(tp2, 8),
        "Priority": priority,
        "Rating": "⭐" * priority,
        "Status": "OPEN"
    }

    return signal, df4h.tail(100), stl.tail(100)

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO v3.2 — LIVE (WIB)", layout="wide")
st.title("🚀 OPSI A PRO v3.2 — LIVE SIGNAL + CHART PREVIEW")

tab1, tab2 = st.tabs(["📡 Live Signal", "📜 Riwayat Sinyal"])
okx = get_okx()

with tab1:
    if st.button("🔍 Scan Live Signal"):
        symbols = get_liquid_symbols(MIN_USDT_VOLUME)
        signals = []

        with st.spinner("Scanning market..."):
            for s in symbols:
                try:
                    result = check_signal(okx, s)
                    if result:
                        sig, df_chart, stl_chart = result
                        save_signal(sig)
                        sig["_chart"] = (df_chart, stl_chart)
                        signals.append(sig)
                except Exception as e:
                    st.write(f"{s} error → {e}")
                time.sleep(RATE_LIMIT_DELAY)

        if signals:
            df = pd.DataFrame(signals).drop(columns=["_chart"])
            df = df.sort_values("Priority", ascending=False)
            st.success(f"🔥 {len(df)} SIGNAL AKTIF")
            st.dataframe(df, use_container_width=True)

            for sig in signals:
                with st.expander(f"📈 {sig['Symbol']} — Chart Preview"):
                    dfc, stlc = sig["_chart"]
                    fig = render_chart(dfc, stlc, sig)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada setup valid.")

with tab2:
    history = load_signal_history().sort_values("Time", ascending=False)
    if history.empty:
        st.info("Belum ada riwayat.")
    else:
        st.dataframe(history, use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            history.to_csv(index=False),
            file_name="signal_history.csv",
            mime="text/csv"
        )
