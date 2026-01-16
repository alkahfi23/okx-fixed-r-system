# =====================================================
# OPSI A PRO v4.0 — PRODUCTION CLEAN
# =====================================================

# =====================
# IMPORTS (WAJIB URUT)
# =====================
import streamlit as st
import os
import time
import json
import random
import requests
import numpy as np
import pandas as pd

import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timezone, timedelta

# =====================================================
# CONFIG GLOBAL
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
MAX_SCAN_SYMBOLS = 120
RATE_LIMIT_DELAY = 0.15

# R accounting (realistic)
TP1_R = 0.5
TP2_R = 1.5
NEW_EXPIRE_HOURS = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_LOG_FILE = os.path.join(BASE_DIR, "signal_history.csv")
TRADE_RESULT_FILE = os.path.join(BASE_DIR, "trade_results.csv")

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))

def now_wib():
    return datetime.now(timezone.utc).astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# CCXT
# =====================================================
@st.cache_resource
def get_okx():
    return ccxt.okx({"enableRateLimit": True})

# =====================================================
# FILE HANDLER
# =====================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        df = pd.DataFrame(columns=[
            "Time","CreatedAt","Symbol","Phase","Score",
            "Entry","SL","TP1","TP2",
            "Rating","Status","Label"
        ])
        df.to_csv(SIGNAL_LOG_FILE, index=False)
    return pd.read_csv(SIGNAL_LOG_FILE)

def save_signal(signal: dict):
    df = load_signal_history()
    if ((df["Symbol"] == signal["Symbol"]) & (df["Status"] == "OPEN")).any():
        return
    df = pd.concat([df, pd.DataFrame([signal])], ignore_index=True)
    df.to_csv(SIGNAL_LOG_FILE, index=False)

def load_trade_results():
    if not os.path.exists(TRADE_RESULT_FILE):
        pd.DataFrame(columns=["Time","Symbol","R"]).to_csv(
            TRADE_RESULT_FILE, index=False
        )
    return pd.read_csv(TRADE_RESULT_FILE)

# =====================================================
# EXPIRE NEW LABEL (4 JAM)
# =====================================================
def expire_new_labels():
    df = load_signal_history()
    if "CreatedAt" not in df.columns:
        return

    now = datetime.now(timezone.utc)
    changed = False

    for i,row in df.iterrows():
        if row.get("Label") != "NEW":
            continue
        try:
            created = datetime.fromisoformat(row["CreatedAt"])
            if now - created > timedelta(hours=NEW_EXPIRE_HOURS):
                df.at[i,"Label"] = ""
                changed = True
        except:
            pass

    if changed:
        df.to_csv(SIGNAL_LOG_FILE, index=False)

# =====================================================
# MARKET SYMBOL FETCH
# =====================================================
@st.cache_data(ttl=300)
def get_liquid_symbols(min_vol):
    r = requests.get(
        "https://www.okx.com/api/v5/market/tickers",
        params={"instType": "SPOT"},
        timeout=15
    )
    r.raise_for_status()

    symbols = [
        d["instId"]
        for d in r.json()["data"]
        if d["instId"].endswith("-USDT")
        and float(d.get("volCcy24h",0)) >= min_vol
    ]

    random.shuffle(symbols)
    return symbols[:MAX_SCAN_SYMBOLS]

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h,l,c = df.high, df.low, df.close
    tr = pd.concat([
        h-l,
        (h-c.shift()).abs(),
        (l-c.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (h+l)/2

    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    stl = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)

    trend.iloc[0] = 1
    stl.iloc[0] = lower.iloc[0]

    for i in range(1,len(df)):
        if trend.iloc[i-1] == 1:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = 1 if c.iloc[i] > stl.iloc[i] else -1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = -1 if c.iloc[i] < stl.iloc[i] else 1

    return stl, trend

def volume_oscillator(v,f,s):
    fast = v.ewm(span=f).mean()
    slow = v.ewm(span=s).mean()
    return (fast - slow) / slow * 100

def accumulation_distribution(df):
    h,l,c,v = df.high, df.low, df.close, df.volume
    mfm = ((c-l)-(h-c))/(h-l)
    mfm = mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

# =====================================================
# SIGNAL LOGIC + SCORE
# =====================================================
def score_signal(trend, vo, adl):
    score = 0
    if trend == 1:
        score += 3
    if vo > 10:
        score += 2
    if adl.iloc[-1] > adl.iloc[-10]:
        score += 2
    return score

def check_signal(okx, symbol):
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
    adl = accumulation_distribution(df4h)

    if trend.iloc[-1] != 1 or vo.iloc[-1] < VO_MIN:
        return None

    ema200 = df1d.close.ewm(span=200).mean()
    if df1d.close.iloc[-1] < ema200.iloc[-1]:
        return None

    entry = df4h.close.iloc[-1]
    sl = df1d.low.min() * (1-ZONE_BUFFER)
    risk = entry - sl
    if risk <= entry*0.002:
        return None

    score = score_signal(trend.iloc[-1], vo.iloc[-1], adl)
    if score < 5:
        return None

    phase = "AKUMULASI_KUAT" if score >= 6 else "AKUMULASI_LEMAH"
    rating = "⭐" * score

    return {
        "Time": now_wib(),
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Symbol": symbol,
        "Phase": phase,
        "Score": score,
        "Entry": round(entry,8),
        "SL": round(sl,8),
        "TP1": round(entry + risk*0.8,8),
        "TP2": round(entry + risk*2.0,8),
        "Rating": rating,
        "Status": "OPEN",
        "Label": "NEW"
    }

# =====================================================
# CHART
# =====================================================
def render_chart(okx, signal):
    df = pd.DataFrame(
        okx.fetch_ohlcv(signal["Symbol"], ENTRY_TF, limit=120),
        columns=["t","open","high","low","close","volume"]
    )

    stl,_ = supertrend(df, ATR_PERIOD, MULTIPLIER)
    adl = accumulation_distribution(df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])

    fig.add_candlestick(
        x=df.index,
        open=df.open, high=df.high,
        low=df.low, close=df.close,
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df.index,y=stl,line=dict(color="lime")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=adl,line=dict(color="cyan")),row=2,col=1)

    for k,c in [("Entry","cyan"),("SL","red"),("TP1","orange"),("TP2","purple")]:
        fig.add_hline(y=signal[k], line_color=c, row=1)

    fig.update_layout(template="plotly_dark", height=520, xaxis_rangeslider_visible=False)
    return fig

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO v4.0", layout="wide")
st.title("🚀 OPSI A PRO v4.0 — PRODUCTION")

okx = get_okx()
expire_new_labels()

tab1, tab2, tab3 = st.tabs(["📡 Live Scan","📜 Riwayat","🎲 Monte Carlo"])

# =====================
# TAB LIVE
# =====================
with tab1:
    if st.button("🔍 Scan Live Signal"):
        results = []
        symbols = get_liquid_symbols(MIN_USDT_VOLUME)

        prog = st.progress(0)
        stat = st.empty()

        for i,s in enumerate(symbols,1):
            stat.text(f"Scanning {s} ({i}/{len(symbols)})")
            try:
                sig = check_signal(okx, s)
                if sig:
                    save_signal(sig)
                    results.append(sig)
            except:
                pass

            prog.progress(i/len(symbols))
            time.sleep(RATE_LIMIT_DELAY)

        prog.empty()
        stat.empty()

        if results:
            df = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.success(f"🔥 {len(df)} SIGNAL DITEMUKAN")
            st.dataframe(df, use_container_width=True)

            for _,row in df.iterrows():
                with st.expander(f"📈 {row['Symbol']} | Score {row['Score']}"):
                    st.plotly_chart(render_chart(okx, row), use_container_width=True)
        else:
            st.warning("Tidak ada setup valid.")

# =====================
# TAB RIWAYAT
# =====================
with tab2:
    df = load_signal_history().sort_values("Time", ascending=False)
    st.dataframe(df, use_container_width=True)

# =====================
# TAB MONTE CARLO
# =====================
with tab3:
    df_r = load_trade_results()
    if len(df_r) < 10:
        st.warning("Data trade belum cukup.")
    else:
        r_vals = df_r["R"].values
        risk = st.slider("Risk / Trade (%)",0.2,3.0,1.0)/100
        trades = st.slider("Trades / Simulation",50,500,300)

        if st.button("🎲 Run Monte Carlo"):
            curves=[]
            for _ in range(500):
                bal=10000; eq=[bal]
                for _ in range(trades):
                    bal += bal * risk * np.random.choice(r_vals)
                    eq.append(bal)
                curves.append(eq)

            curves = np.array(curves)
            st.metric("Median Balance", f"${np.median(curves[:,-1]):,.0f}")
            st.metric("Risk of Ruin", f"{(curves[:,-1]<5000).mean()*100:.2f}%")

            fig = go.Figure()
            for i in range(min(30,len(curves))):
                fig.add_trace(go.Scatter(y=curves[i],mode="lines",opacity=0.3,showlegend=False))
            fig.update_layout(template="plotly_dark",height=400)
            st.plotly_chart(fig,use_container_width=True)
