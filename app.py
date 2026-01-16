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

# ================== GLOBAL DEBUG ==================
DEBUG_LOG = []

# ==================================================
# CONFIG
# ==================================================
ENTRY_TF = "4h"
ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28
VO_MIN = 5

ZONE_BUFFER = 0.01
MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

TP1_R = 0.8
TP2_R = 2.0
NEW_EXPIRE_HOURS = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_LOG_FILE = os.path.join(BASE_DIR, "signal_history.csv")
TRADE_RESULT_FILE = os.path.join(BASE_DIR, "trade_results.csv")

# ==================================================
# TIMEZONE
# ==================================================
WIB = timezone(timedelta(hours=7))
def now_wib():
    return datetime.now(timezone.utc).astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")

# ==================================================
# EXCHANGES
# ==================================================
@st.cache_resource
def get_okx():
    return ccxt.okx({"enableRateLimit": True})

@st.cache_resource
def get_bitget():
    return ccxt.bitget({"enableRateLimit": True})

# ==================================================
# FILE HANDLER
# ==================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        df = pd.DataFrame(columns=[
            "Time","CreatedAt","Symbol","Source",
            "Phase","Score","Entry","SL","TP1","TP2",
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

# ==================================================
# SYMBOL SOURCES
# ==================================================
@st.cache_data(ttl=300)
def get_okx_symbols(min_vol):
    r = requests.get(
        "https://www.okx.com/api/v5/market/tickers",
        params={"instType": "SPOT"},
        timeout=15
    )
    r.raise_for_status()
    return [
        (d["instId"], "OKX")
        for d in r.json()["data"]
        if d["instId"].endswith("-USDT")
        and float(d["volCcy24h"]) >= min_vol
    ]

@st.cache_data(ttl=300)
def get_bitget_symbols(min_vol):
    r = requests.get(
        "https://api.bitget.com/api/v2/spot/market/tickers",
        timeout=15
    )
    r.raise_for_status()

    out = []
    for d in r.json().get("data", []):
        try:
            sym = d.get("symbol")
            if sym and sym.endswith("USDT"):
                vol = float(d.get("usdtVol", 0) or 0)
                if vol >= min_vol:
                    out.append((sym, "BITGET"))
        except:
            pass
    return out

def get_all_symbols(min_vol):
    merged = {}
    for s,src in get_okx_symbols(min_vol):
        merged[s] = src
    for s,src in get_bitget_symbols(min_vol):
        merged.setdefault(s, src)
    items = list(merged.items())
    random.shuffle(items)
    return items[:MAX_SCAN_SYMBOLS]

# ==================================================
# INDICATORS
# ==================================================
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

def volume_oscillator(v, fast=14, slow=28):
    fast_ma = v.ewm(span=fast, adjust=False).mean()
    slow_ma = v.ewm(span=slow, adjust=False).mean()
    return (fast_ma - slow_ma) / slow_ma * 100

def accumulation_distribution(df):
    h,l,c,v=df.high,df.low,df.close,df.volume
    mfm=((c-l)-(h-c))/(h-l)
    mfm=mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

# ==================================================
# SIGNAL LOGIC
# ==================================================
def check_signal(exchange, symbol, source, debug_log, debug_on):
    try:
        df = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, ENTRY_TF, limit=200),
            columns=["t","open","high","low","close","volume"]
        )

        if len(df) < 50:
            if debug_on:
                debug_log.append({"Symbol":symbol,"Source":source,"Reason":"OHLCV < 50"})
            return None

        stl, trend = supertrend(df, ATR_PERIOD, MULTIPLIER)
        vo = volume_oscillator(df["volume"], VO_FAST, VO_SLOW)

        if trend.iloc[-1] != 1:
            if debug_on:
                debug_log.append({"Symbol":symbol,"Source":source,"Reason":"Trend bearish"})
            return None

        if vo.iloc[-1] < VO_MIN:
            if debug_on:
                debug_log.append({"Symbol":symbol,"Source":source,"Reason":"Volume lemah"})
            return None

        entry = df["close"].iloc[-1]
        sl = df["low"].rolling(20).min().iloc[-1]*(1-ZONE_BUFFER)
        risk = entry - sl

        if risk <= entry*0.002:
            if debug_on:
                debug_log.append({"Symbol":symbol,"Source":source,"Reason":"Risk kecil"})
            return None

        score = 5 + int(vo.iloc[-1] > 10)

        return {
            "Time": now_wib(),
            "CreatedAt": datetime.now(timezone.utc).isoformat(),
            "Symbol": symbol,
            "Source": source,
            "Phase": "AKUMULASI_KUAT",
            "Score": score,
            "Entry": round(entry,6),
            "SL": round(sl,6),
            "TP1": round(entry+risk*TP1_R,6),
            "TP2": round(entry+risk*TP2_R,6),
            "Rating": "⭐"*score,
            "Status": "OPEN",
            "Label": "NEW"
        }

    except Exception as e:
        if debug_on:
            debug_log.append({"Symbol":symbol,"Source":source,"Reason":str(e)})
        return None

# ==================================================
# CHART
# ==================================================
def render_chart(df, stl, adl, sig):
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3])
    fig.add_candlestick(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=stl,line=dict(color="lime")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=adl,line=dict(color="cyan")),row=2,col=1)
    for k,c in [("Entry","cyan"),("SL","red"),("TP1","orange"),("TP2","purple")]:
        fig.add_hline(y=sig[k],line_color=c,row=1)
    fig.update_layout(template="plotly_dark",height=520,xaxis_rangeslider_visible=False)
    return fig

# ==================================================
# UI
# ==================================================
st.set_page_config("OPSI A PRO — FINAL", layout="wide")
st.title("🚀 OPSI A PRO — FINAL PRODUCTION")

with st.sidebar:
    DEBUG_MODE = st.toggle("🧪 Debug Filter", value=False)

okx = get_okx()
bitget = get_bitget()

tab1, tab2, tab3, tab4 = st.tabs(["📡 Live Scan","📜 Riwayat","🎲 Monte Carlo","🧪 Debug"])

with tab1:
    if st.button("🔍 Scan Live Signal"):
        DEBUG_LOG.clear()
        found=[]
        symbols = get_all_symbols(MIN_USDT_VOLUME)
        prog = st.progress(0)

        for i,(sym,src) in enumerate(symbols,1):
            ex = okx if src=="OKX" else bitget
            sig = check_signal(ex, sym, src, DEBUG_LOG, DEBUG_MODE)
            if sig:
                save_signal(sig)
                found.append(sig)
            prog.progress(i/len(symbols))
            time.sleep(RATE_LIMIT_DELAY)

        prog.empty()

        if found:
            df=pd.DataFrame(found).sort_values("Score",ascending=False)
            st.success(f"🔥 {len(df)} SIGNAL DITEMUKAN")
            st.dataframe(df,use_container_width=True)
        else:
            st.warning("Tidak ada setup valid.")

with tab2:
    st.dataframe(load_signal_history().sort_values("Score",ascending=False),use_container_width=True)

with tab4:
    if DEBUG_MODE and DEBUG_LOG:
        st.dataframe(pd.DataFrame(DEBUG_LOG),use_container_width=True)
    else:
        st.info("Debug OFF atau belum ada data.")
