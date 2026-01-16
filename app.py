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
# CONFIG
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
ZONE_BUFFER = 0.01

MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

# R accounting
TP1_R = 0.8
TP2_R = 2.0

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

@st.cache_resource
def get_bitget():
    return ccxt.bitget({"enableRateLimit": True})

# =====================================================
# FILE HANDLER
# =====================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        df = pd.DataFrame(columns=[
            "Time","CreatedAt","Symbol","Source",
            "Phase","Score",
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
# EXPIRE NEW LABEL (4 JAM)
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

expire_new()

# =====================================================
# SYMBOL SOURCES
# =====================================================
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

    symbols = []
    for d in r.json().get("data", []):
        try:
            symbol = d.get("symbol")
            if not symbol or not symbol.endswith("USDT"):
                continue

            # fallback volume field (AMAN)
            vol = (
                float(d.get("usdtVol", 0)) or
                float(d.get("quoteVol", 0)) or
                float(d.get("volCcy24h", 0))
            )

            if vol >= min_vol:
                symbols.append((symbol, "BITGET"))

        except Exception:
            continue

    return symbols


def get_all_symbols(min_vol):
    merged = {}
    for s,src in get_okx_symbols(min_vol):
        merged[s] = src
    for s,src in get_bitget_symbols(min_vol):
        if s not in merged:
            merged[s] = src
    items = list(merged.items())
    random.shuffle(items)
    return items[:MAX_SCAN_SYMBOLS]

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

def volume_osc(v,f,s):
    return (v.ewm(span=f).mean()-v.ewm(span=s).mean())/v.ewm(span=s).mean()*100

def accumulation_distribution(df):
    h,l,c,v=df.high,df.low,df.close,df.volume
    mfm=((c-l)-(h-c))/(h-l)
    mfm=mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def find_support(df, lb):
    levels=[]
    for i in range(lb, len(df)-lb):
        if df.low.iloc[i] == min(df.low.iloc[i-lb:i+lb+1]):
            levels.append(df.low.iloc[i])
    levels=sorted(set(levels))
    clean=[]
    for s in levels:
        if not clean or abs(s-clean[-1])/s>0.01:
            clean.append(s)
    return clean

# =====================================================
# SIGNAL LOGIC (SCORE BASED)
# =====================================================
def check_signal(exchange, symbol, source):
    df4h = pd.DataFrame(exchange.fetch_ohlcv(symbol, ENTRY_TF, limit=LIMIT_4H),
        columns=["t","open","high","low","close","volume"])
    df1d = pd.DataFrame(exchange.fetch_ohlcv(symbol, SR_TF, limit=LIMIT_1D),
        columns=["t","open","high","low","close","volume"])

    stl,trend = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
    vo = volume_osc(df4h.volume, VO_FAST, VO_SLOW)
    adl = accumulation_distribution(df4h)

    score = 0
    phase = None

    if trend.iloc[-1]==1:
        score += 2
    if vo.iloc[-1] > VO_MIN:
        score += 1
    if adl.iloc[-1] > adl.iloc[-10]:
        score += 2

    if score >= 5:
        phase = "AKUMULASI_KUAT"
    elif score >= 3:
        phase = "AKUMULASI_LEMAH"
    else:
        return None

    ema200 = df1d.close.ewm(span=200).mean()
    if df1d.close.iloc[-1] < ema200.iloc[-1]:
        return None

    entry = df4h.close.iloc[-1]
    supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
    if not supports:
        return None

    sl = max(supports)*(1-ZONE_BUFFER)
    risk = entry - sl
    if risk <= entry*0.002:
        return None

    tp1 = entry + risk*TP1_R
    tp2 = entry + risk*TP2_R

    return {
        "Time": now_wib(),
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Symbol": symbol,
        "Source": source,
        "Phase": phase,
        "Score": score,
        "Entry": round(entry,4),
        "SL": round(sl,4),
        "TP1": round(tp1,4),
        "TP2": round(tp2,4),
        "Rating": "⭐"*score,
        "Status": "OPEN",
        "Label": "NEW"
    }

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

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO — FINAL", layout="wide")
st.title("🚀 OPSI A PRO — FINAL PRODUCTION")

okx = get_okx()
bitget = get_bitget()

tab1,tab2,tab3 = st.tabs(["📡 Live Scan","📜 Riwayat","🎲 Monte Carlo"])

with tab1:
    if st.button("🔍 Scan Live Signal"):
        symbols = get_all_symbols(MIN_USDT_VOLUME)
        progress = st.progress(0)
        found = []

        for i,(sym,src) in enumerate(symbols,1):
            ex = okx if src=="OKX" else bitget
            try:
                sig = check_signal(ex, sym, src)
                if sig:
                    save_signal(sig)
                    found.append(sig)
            except:
                pass
            progress.progress(i/len(symbols))
            time.sleep(RATE_LIMIT_DELAY)

        progress.empty()

        if found:
            st.success(f"🔥 {len(found)} SIGNAL DITEMUKAN")
            df = pd.DataFrame(found).sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True)

            for sig in found:
                with st.expander(f"📈 {sig['Symbol']} ({sig['Source']})"):
                    dfc = pd.DataFrame(
                        (okx if sig["Source"]=="OKX" else bitget)
                        .fetch_ohlcv(sig["Symbol"], ENTRY_TF, limit=120),
                        columns=["t","open","high","low","close","volume"]
                    )
                    stl,_ = supertrend(dfc, ATR_PERIOD, MULTIPLIER)
                    adl = accumulation_distribution(dfc)
                    st.plotly_chart(render_chart(dfc,stl,adl,sig),use_container_width=True)
        else:
            st.warning("Tidak ada setup valid.")

with tab2:
    df = load_signal_history().sort_values("Score", ascending=False)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        "signal_history.csv",
        "text/csv"
    )

with tab3:
    df_r = load_trade_results()
    if len(df_r) < 10:
        st.warning("Data trade belum cukup.")
    else:
        r = df_r["R"].values
        risk = st.slider("Risk / Trade (%)",0.2,3.0,1.0)/100
        trades = st.slider("Trades / Simulation",50,500,300)

        if st.button("🎲 Run Monte Carlo"):
            curves=[]
            for _ in range(500):
                bal=10000; eq=[bal]
                for _ in range(trades):
                    bal += bal*risk*np.random.choice(r)
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

