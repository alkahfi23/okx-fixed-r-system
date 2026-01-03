import streamlit as st
import ccxt
import pandas as pd
import requests
import plotly.graph_objects as go
import time
from datetime import datetime

# =====================================================
# CONFIG — FINAL
# =====================================================
ENTRY_TF = "4h"
SR_TF = "1d"

LIMIT_4H = 200
LIMIT_1D = 200

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28

SR_LOOKBACK = 5
ZONE_BUFFER = 0.008

MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15

VALID_CANDLES = {
    "Bullish Engulfing",
    "Hammer",
    "Strong Bullish",
    "Normal"
}

# === TP FINAL (LOCK)
TP1_R = 0.8
TP2_R = 2.0
TP1_PORTION = 0.3
TP2_PORTION = 0.7

# =====================================================
# SYMBOL FETCH
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
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (h+l)/2

    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    stl = pd.Series(index=df.index)
    trend = pd.Series(index=df.index)

    stl.iloc[0] = upper.iloc[0]
    trend.iloc[0] = -1

    for i in range(1,len(df)):
        if c.iloc[i] > stl.iloc[i-1]:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = 1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = -1

    return stl, trend

def volume_oscillator(v,f,s):
    return (v.ewm(span=f).mean() - v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

# =====================================================
# PRICE ACTION
# =====================================================
def detect_candle(df):
    o,h,l,c = df.open,df.high,df.low,df.close
    po,pc = o.shift(1),c.shift(1)
    body = abs(c-o)
    rng = h-l

    if c.iloc[-1]>o.iloc[-1] and pc.iloc[-1]<po.iloc[-1] and c.iloc[-1]>po.iloc[-1]:
        return "Bullish Engulfing"
    if c.iloc[-1]>o.iloc[-1] and (o.iloc[-1]-l.iloc[-1])>2*body.iloc[-1]:
        return "Hammer"
    if rng.iloc[-1]>0 and body.iloc[-1]/rng.iloc[-1]>0.65 and c.iloc[-1]>o.iloc[-1]:
        return "Strong Bullish"
    return "Normal"

# =====================================================
# SUPPORT
# =====================================================
def find_support(df,lb):
    s=[]
    for i in range(lb,len(df)-lb):
        if df.low.iloc[i]==min(df.low.iloc[i-lb:i+lb+1]):
            s.append(df.low.iloc[i])
    return sorted(set(s))

# =====================================================
# SIGNAL CHECK
# =====================================================
def check_signal(okx,symbol):
    df4h = pd.DataFrame(
        okx.fetch_ohlcv(symbol,ENTRY_TF,limit=LIMIT_4H),
        columns=["t","open","high","low","close","volume"]
    )
    df1d = pd.DataFrame(
        okx.fetch_ohlcv(symbol,SR_TF,limit=LIMIT_1D),
        columns=["t","open","high","low","close","volume"]
    )

    stl,trend = supertrend(df4h,ATR_PERIOD,MULTIPLIER)
    vo = volume_oscillator(df4h.volume,VO_FAST,VO_SLOW)
    candle = detect_candle(df4h)

    if trend.iloc[-1]!=1 or vo.iloc[-1]<0 or candle not in VALID_CANDLES:
        return None

    ema200 = df1d.close.ewm(span=200).mean()
    if df1d.close.iloc[-1] < ema200.iloc[-1]:
        return None

    entry = df4h.close.iloc[-1]
    supports = [s for s in find_support(df1d,SR_LOOKBACK) if s<entry]
    if not supports:
        return None

    sl = max(supports)*(1-ZONE_BUFFER)
    risk = entry-sl
    tp1 = entry + risk*TP1_R
    tp2 = entry + risk*TP2_R

    return {
        "Symbol":symbol,
        "Candle":candle,
        "Entry":round(entry,6),
        "SL":round(sl,6),
        "TP1":round(tp1,6),
        "TP2":round(tp2,6)
    }

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO v3 — LIVE SIGNAL",layout="wide")
st.title("🚀 OPSI A PRO v3 — LIVE SIGNAL")

tab1,tab2 = st.tabs(["📡 Live Signal","🧪 Backtest"])

okx = ccxt.okx({"enableRateLimit":True})

# =====================================================
# LIVE SIGNAL TAB
# =====================================================
with tab1:
    if st.button("🔍 Scan Live Signal"):
        symbols = get_liquid_symbols(MIN_USDT_VOLUME)
        signals=[]

        with st.spinner("Scanning market..."):
            for s in symbols:
                try:
                    sig = check_signal(okx,s)
                    if sig:
                        sig["Time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                        signals.append(sig)
                except:
                    pass
                time.sleep(RATE_LIMIT_DELAY)

        if signals:
            st.success(f"🔥 {len(signals)} SIGNAL AKTIF")
            st.dataframe(pd.DataFrame(signals),use_container_width=True)
        else:
            st.warning("Tidak ada setup valid saat ini")

# =====================================================
# BACKTEST TAB (OPTIONAL)
# =====================================================
with tab2:
    st.info("Backtest sudah LOCK — gunakan versi sebelumnya")
