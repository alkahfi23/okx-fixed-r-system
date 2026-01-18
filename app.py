# =====================================================
# OPSI A PRO — FINAL PRODUCTION (OKX ONLY)
# =====================================================

import streamlit as st
import ccxt
import pandas as pd
import requests
import time
import os
import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta

# =====================================================
# GLOBAL DEBUG
# =====================================================
DEBUG_LOG = []

# =====================================================
# CONFIG
# =====================================================
ENTRY_TF = "4h"
LIMIT_4H = 200

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28
VO_MIN = 5

SUPPORT_LOOKBACK = 20
SUPPORT_BUFFER = 0.995

MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

# R ACCOUNTING
TP1_PARTIAL_R = 0.5
TP2_FINAL_R   = 1.5

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
# EXCHANGE (OKX)
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

def save_signal(sig):
    df = load_signal_history()
    if ((df["Symbol"] == sig["Symbol"]) & (df["Status"] == "OPEN")).any():
        return
    df = pd.concat([df, pd.DataFrame([sig])], ignore_index=True)
    df.to_csv(SIGNAL_LOG_FILE, index=False)

def load_trade_results():
    if not os.path.exists(TRADE_RESULT_FILE):
        pd.DataFrame(columns=["Time","Symbol","R"]).to_csv(
            TRADE_RESULT_FILE, index=False
        )
    return pd.read_csv(TRADE_RESULT_FILE)

# =====================================================
# AUTO EXPIRE "NEW" (4 JAM)
# =====================================================
def expire_new():
    df = load_signal_history()
    if df.empty:
        return
    now = datetime.now(timezone.utc)
    changed = False
    for i,row in df.iterrows():
        if row["Label"] != "NEW":
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
# AUTO UPDATE TP / SL (REAL-TIME)
# =====================================================
def update_trade_outcomes(okx):
    history = load_signal_history()
    if history.empty:
        return

    results = load_trade_results()
    updated = False

    for i,row in history.iterrows():
        status = row["Status"]
        if status in ["SL HIT","TP2 HIT","BE HIT"]:
            continue

        try:
            price = okx.fetch_ticker(row["Symbol"])["last"]
        except:
            continue

        r = None
        new_status = None

        if status == "OPEN":
            if price <= row["SL"]:
                r, new_status = -1, "SL HIT"
            elif price >= row["TP1"]:
                r, new_status = TP1_PARTIAL_R, "TP1 HIT"

        elif status == "TP1 HIT":
            if price <= row["Entry"]:
                r, new_status = 0, "BE HIT"
            elif price >= row["TP2"]:
                r, new_status = TP2_FINAL_R, "TP2 HIT"

        if new_status:
            history.at[i,"Status"] = new_status
            results = pd.concat([
                results,
                pd.DataFrame([{
                    "Time": now_wib(),
                    "Symbol": row["Symbol"],
                    "R": r
                }])
            ], ignore_index=True)
            updated = True

    if updated:
        history.to_csv(SIGNAL_LOG_FILE, index=False)
        results.to_csv(TRADE_RESULT_FILE, index=False)

# =====================================================
# MARKET SYMBOLS (OKX)
# =====================================================
@st.cache_data(ttl=300)
def get_okx_symbols(min_vol):
    r = requests.get(
        "https://www.okx.com/api/v5/market/tickers",
        params={"instType": "SPOT"},
        timeout=15
    )
    r.raise_for_status()
    syms = [
        d["instId"] for d in r.json()["data"]
        if d["instId"].endswith("-USDT")
        and float(d["volCcy24h"]) >= min_vol
    ]
    random.shuffle(syms)
    return syms[:MAX_SCAN_SYMBOLS]

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h,l,c = df.high, df.low, df.close
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=period,adjust=False).mean()
    hl2 = (h+l)/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    stl = pd.Series(index=df.index,dtype=float)
    trend = pd.Series(index=df.index,dtype=int)
    trend.iloc[0]=1
    stl.iloc[0]=lower.iloc[0]

    for i in range(1,len(df)):
        if trend.iloc[i-1]==1:
            stl.iloc[i]=max(lower.iloc[i],stl.iloc[i-1])
            trend.iloc[i]=1 if c.iloc[i]>stl.iloc[i] else -1
        else:
            stl.iloc[i]=min(upper.iloc[i],stl.iloc[i-1])
            trend.iloc[i]=-1 if c.iloc[i]<stl.iloc[i] else 1
    return stl,trend

def volume_oscillator(volume, fast=14, slow=28):
    fast_ma = volume.ewm(span=fast, adjust=False).mean()
    slow_ma = volume.ewm(span=slow, adjust=False).mean()
    return (fast_ma - slow_ma) / slow_ma * 100

def accumulation_distribution(df):
    h,l,c,v=df.high,df.low,df.close,df.volume
    mfm=((c-l)-(h-c))/(h-l)
    mfm=mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

# =====================================================
# SIGNAL LOGIC (SCORE BASED + SUPPORT SL)
# =====================================================
def check_signal(okx, symbol, debug=False):
    try:
        df = pd.DataFrame(
            okx.fetch_ohlcv(symbol, ENTRY_TF, limit=LIMIT_4H),
            columns=["t","open","high","low","close","volume"]
        )

        if len(df) < 50:
            if debug: DEBUG_LOG.append({"Symbol":symbol,"Reason":"OHLCV kurang"})
            return None

        stl,trend = supertrend(df, ATR_PERIOD, MULTIPLIER)
        vo = volume_oscillator(df.volume, VO_FAST, VO_SLOW)
        adl = accumulation_distribution(df)

        score = 0
        if trend.iloc[-1] == 1: score += 3
        if vo.iloc[-1] > VO_MIN: score += 2
        if adl.iloc[-1] > adl.iloc[-10]: score += 2

        if score < 5:
            if debug: DEBUG_LOG.append({"Symbol":symbol,"Reason":"Score rendah"})
            return None

        entry = df.close.iloc[-1]
        support = df.low.rolling(SUPPORT_LOOKBACK).min().iloc[-1]
        sl = support * SUPPORT_BUFFER
        risk = entry - sl

        if risk <= entry*0.002:
            if debug: DEBUG_LOG.append({"Symbol":symbol,"Reason":"Risk kecil"})
            return None

        phase = "AKUMULASI_KUAT" if score >= 6 else "AKUMULASI_LEMAH"
        rating = "⭐"*score

        return {
            "Time": now_wib(),
            "CreatedAt": datetime.now(timezone.utc).isoformat(),
            "Symbol": symbol,
            "Phase": phase,
            "Score": score,
            "Entry": round(entry,6),
            "SL": round(sl,6),
            "TP1": round(entry + risk*0.8,6),
            "TP2": round(entry + risk*2.0,6),
            "Rating": rating,
            "Status": "OPEN",
            "Label": "NEW"
        }

    except Exception as e:
        if debug:
            DEBUG_LOG.append({"Symbol":symbol,"Reason":str(e)})
        return None

# =====================================================
# CHART
# =====================================================
def render_chart(df, stl, adl, sig):
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3])
    fig.add_candlestick(
        x=df.index,open=df.open,high=df.high,
        low=df.low,close=df.close,row=1,col=1
    )
    fig.add_trace(go.Scatter(x=df.index,y=stl,line=dict(color="lime")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=adl,line=dict(color="cyan")),row=2,col=1)

    for k,c in [("Entry","cyan"),("SL","red"),("TP1","orange"),("TP2","purple")]:
        fig.add_hline(y=sig[k],line_color=c,row=1)

    fig.update_layout(template="plotly_dark",height=520,xaxis_rangeslider_visible=False)
    return fig


def restore_signal_history_from_upload(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = [
            "Time","CreatedAt","Symbol","Phase","Score",
            "Entry","SL","TP1","TP2",
            "Rating","Status","Label"
        ]

        if not all(c in df.columns for c in required_cols):
            return False, "Format CSV tidak valid"

        df.to_csv(SIGNAL_LOG_FILE, index=False)
        return True, f"{len(df)} signal berhasil direstore"

    except Exception as e:
        return False, str(e)

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO — OKX FINAL", layout="wide")
st.title("🚀 OPSI A PRO — OKX FINAL")

with st.sidebar:
    DEBUG_MODE = st.toggle("🧪 Debug Filter", False)

okx = get_okx()

expire_new()

if "trade_updated" not in st.session_state:
    update_trade_outcomes(okx)
    st.session_state.trade_updated = True

tab1, tab2, tab3, tab4 = st.tabs(
    ["📡 Live Scan","📜 Riwayat","🎲 Monte Carlo","🧪 Debug"]
)

# =====================================================
# TAB 1 — LIVE SCAN
# =====================================================
with tab1:
    if st.button("🔍 Scan Live Signal"):
        DEBUG_LOG.clear()
        found = []
        symbols = get_okx_symbols(MIN_USDT_VOLUME)
        progress = st.progress(0)

        for i,sym in enumerate(symbols,1):
            sig = check_signal(okx, sym, DEBUG_MODE)
            if sig:
                save_signal(sig)
                found.append(sig)
            progress.progress(i/len(symbols))
            time.sleep(RATE_LIMIT_DELAY)

        progress.empty()

        if found:
            df = pd.DataFrame(found).sort_values("Score", ascending=False)
            st.success(f"🔥 {len(df)} SIGNAL DITEMUKAN")
            st.dataframe(df, use_container_width=True)

            for sig in found:
                with st.expander(f"📈 {sig['Symbol']}"):
                    dfc = pd.DataFrame(
                        okx.fetch_ohlcv(sig["Symbol"], ENTRY_TF, limit=120),
                        columns=["t","open","high","low","close","volume"]
                    )
                    stl,_ = supertrend(dfc, ATR_PERIOD, MULTIPLIER)
                    adl = accumulation_distribution(dfc)
                    st.plotly_chart(
                        render_chart(dfc,stl,adl,sig),
                        use_container_width=True
                    )
        else:
            st.warning("Tidak ada setup valid.")

# =====================================================
# TAB 2 — HISTORY
# =====================================================
with tab2:
    st.subheader("📜 Riwayat Signal")

    # ==========================
    # RESTORE SECTION
    # ==========================
    with st.expander("📤 Restore Signal History (CSV)"):
        uploaded = st.file_uploader(
            "Upload signal_history.csv",
            type=["csv"],
            help="Upload file hasil backup sebelumnya"
        )

        if uploaded:
            ok, msg = restore_signal_history_from_upload(uploaded)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    # ==========================
    # SHOW DATA
    # ==========================
    df = load_signal_history().sort_values("Score", ascending=False)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        "signal_history.csv",
        "text/csv"
    )

# =====================================================
# TAB 3 — MONTE CARLO
# =====================================================
with tab3:
    df_r = load_trade_results()
    if len(df_r) < 10:
        st.warning("Data trade belum cukup (min 10).")
    else:
        r_vals = df_r["R"].values
        risk = st.slider("Risk / Trade (%)",0.2,3.0,1.0)/100
        trades = st.slider("Trades / Simulation",50,500,300)

        if st.button("🎲 Run Monte Carlo"):
            curves=[]
            for _ in range(500):
                bal=10000; eq=[bal]
                for _ in range(trades):
                    bal += bal*risk*np.random.choice(r_vals)
                    eq.append(bal)
                curves.append(eq)
            curves=np.array(curves)

            st.metric("Median Final Balance", f"${np.median(curves[:,-1]):,.0f}")
            st.metric("Risk of Ruin (<$5k)", f"{(curves[:,-1]<5000).mean()*100:.2f}%")

            fig=go.Figure()
            for i in range(min(30,len(curves))):
                fig.add_trace(go.Scatter(y=curves[i],mode="lines",opacity=0.3,showlegend=False))
            fig.update_layout(template="plotly_dark",height=400)
            st.plotly_chart(fig,use_container_width=True)

# =====================================================
# TAB 4 — DEBUG
# =====================================================
with tab4:
    st.subheader("🧪 Debug Rejected Symbols")
    if not DEBUG_LOG:
        st.info("Belum ada data debug.")
    else:
        df_dbg = pd.DataFrame(DEBUG_LOG)
        st.dataframe(df_dbg, use_container_width=True)
        st.bar_chart(df_dbg["Reason"].value_counts())

