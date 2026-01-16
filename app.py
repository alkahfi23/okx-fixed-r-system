import streamlit as st
import os
import ccxt
import pandas as pd
import requests
import time
import json
import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

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

ZONE_BUFFER = 0.008
MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

TP1_R = 0.8
TP2_R = 2.0

TP1_PARTIAL_R = 0.5
TP2_FINAL_R = 1.5

NEW_EXPIRE_HOURS = 4

SIGNAL_LOG_FILE = "signal_history.csv"
TRADE_RESULT_FILE = "trade_results.csv"

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))
def now_wib():
    return datetime.now(timezone.utc).astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# GOOGLE DRIVE
# =====================================================
def get_drive_service():
    try:
        info = json.loads(st.secrets["GDRIVE_SERVICE_JSON"])
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        return build("drive", "v3", credentials=creds)
    except:
        return None

def backup_csv(df, filename):
    service = get_drive_service()
    folder_id = st.secrets.get("GDRIVE_FOLDER_ID")
    if not service or not folder_id:
        return

    data = df.to_csv(index=False).encode("utf-8")
    media = MediaInMemoryUpload(data, mimetype="text/csv")

    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()
    for f in res.get("files", []):
        service.files().delete(fileId=f["id"]).execute()

    service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media
    ).execute()

def restore_csv(filename):
    service = get_drive_service()
    folder_id = st.secrets.get("GDRIVE_FOLDER_ID")
    if not service or not folder_id:
        return

    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()
    files = res.get("files", [])
    if not files:
        return

    content = service.files().get_media(fileId=files[0]["id"]).execute()
    with open(filename, "wb") as f:
        f.write(content)

# =====================================================
# RESTORE ONCE
# =====================================================
if "restored" not in st.session_state:
    restore_csv(SIGNAL_LOG_FILE)
    restore_csv(TRADE_RESULT_FILE)
    st.session_state.restored = True

# =====================================================
# FILE HANDLERS
# =====================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        pd.DataFrame(columns=[
            "Time","CreatedAt","Symbol","Source","Phase",
            "Score","Rating",
            "Entry","SL","TP1","TP2",
            "Status","R","Label"
        ]).to_csv(SIGNAL_LOG_FILE, index=False)
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
# LABEL NEW EXPIRE
# =====================================================
def expire_new():
    df = load_signal_history()
    now = datetime.now(timezone.utc)
    changed = False
    for i, r in df.iterrows():
        if r["Label"] == "NEW":
            created = datetime.fromisoformat(r["CreatedAt"])
            if now - created > timedelta(hours=NEW_EXPIRE_HOURS):
                df.at[i,"Label"] = ""
                changed = True
    if changed:
        df.to_csv(SIGNAL_LOG_FILE, index=False)

expire_new()

# =====================================================
# EXCHANGES
# =====================================================
@st.cache_resource
def get_exchanges():
    return {
        "OKX": ccxt.okx({"enableRateLimit": True}),
        "BITGET": ccxt.bitget({"enableRateLimit": True, "options":{"defaultType":"spot"}})
    }

def map_symbol(symbol, ex):
    return symbol.replace("-", "/") if ex=="BITGET" else symbol

def fetch_ohlcv_multi(exs, symbol, tf, limit):
    for name, ex in exs.items():
        try:
            s = map_symbol(symbol, name)
            data = ex.fetch_ohlcv(s, tf, limit=limit)
            if data and len(data) >= 100:
                return data, name
        except:
            continue
    return None, None

def fetch_price_multi(exs, symbol):
    for name, ex in exs.items():
        try:
            return ex.fetch_ticker(map_symbol(symbol,name))["last"]
        except:
            continue
    return None

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h,l,c = df.high,df.low,df.close
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=period,adjust=False).mean()
    hl2=(h+l)/2
    upper=hl2+mult*atr
    lower=hl2-mult*atr

    stl=[lower.iloc[0]]
    for i in range(1,len(df)):
        stl.append(
            max(lower.iloc[i],stl[-1]) if c.iloc[i]>stl[-1]
            else min(upper.iloc[i],stl[-1])
        )
    return pd.Series(stl,index=df.index)

def volume_oscillator(v,f,s):
    return (v.ewm(span=f).mean()-v.ewm(span=s).mean())/v.ewm(span=s).mean()*100

def accumulation_distribution(df):
    h,l,c,v=df.high,df.low,df.close,df.volume
    mfm=((c-l)-(h-c))/(h-l)
    mfm=mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def ad_phase(adl, lookback=10):
    slope = adl.iloc[-1] - adl.iloc[-lookback]
    avg = adl.diff().rolling(lookback).mean().iloc[-1]
    strength = slope / (abs(avg)+1e-9)
    if slope > 0:
        return "AKUMULASI_KUAT" if strength > 2 else "AKUMULASI_LEMAH"
    if slope < 0:
        return "DISTRIBUSI"
    return "NETRAL"

# =====================================================
# SCORE ENGINE (CORE)
# =====================================================
def compute_score(df4h, df1d):
    score = 0

    # Supertrend
    stl = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
    if df4h.close.iloc[-1] > stl.iloc[-1]:
        score += 25

    # Volume Oscillator
    vo = volume_oscillator(df4h.volume, VO_FAST, VO_SLOW)
    if vo.iloc[-1] > VO_MIN:
        score += 20

    # Accumulation
    adl = accumulation_distribution(df4h)
    phase = ad_phase(adl)
    if phase == "AKUMULASI_KUAT":
        score += 25
    elif phase == "AKUMULASI_LEMAH":
        score += 15

    # Daily trend
    ema200 = df1d.close.ewm(span=200).mean()
    if df1d.close.iloc[-1] > ema200.iloc[-1]:
        score += 15

    # RR quality
    entry = df4h.close.iloc[-1]
    sl = df1d.low.min() * (1 - ZONE_BUFFER)
    rr = (entry - sl) / entry
    if rr > 0.01:
        score += 15
    elif rr > 0.006:
        score += 10

    return score, phase, stl, adl

# =====================================================
# SIGNAL CHECK
# =====================================================
def check_signal(exs, symbol):
    d4h, src = fetch_ohlcv_multi(exs, symbol, ENTRY_TF, LIMIT_4H)
    d1d, _   = fetch_ohlcv_multi(exs, symbol, SR_TF, LIMIT_1D)
    if not d4h or not d1d:
        return None

    df4h=pd.DataFrame(d4h,columns=["t","open","high","low","close","volume"])
    df1d=pd.DataFrame(d1d,columns=["t","open","high","low","close","volume"])

    score, phase, stl, adl = compute_score(df4h, df1d)

    if score < 60:
        return None

    entry = df4h.close.iloc[-1]
    sl = df1d.low.min() * (1 - ZONE_BUFFER)
    risk = entry - sl
    if risk < entry*0.002:
        return None

    rating = "⭐" * min(5, max(3, score//20))

    return {
        "Time": now_wib(),
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Symbol": symbol,
        "Source": src,
        "Phase": phase,
        "Score": score,
        "Rating": rating,
        "Entry": round(entry,6),
        "SL": round(sl,6),
        "TP1": round(entry+risk*TP1_R,6),
        "TP2": round(entry+risk*TP2_R,6),
        "Status": "OPEN",
        "R": 0,
        "Label": "NEW"
    }

# =====================================================
# CHART
# =====================================================
def render_chart(df, stl, adl, sig):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3])
    fig.add_candlestick(
        x=df.index, open=df.open, high=df.high,
        low=df.low, close=df.close, row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df.index,y=stl,
                             line=dict(color="lime"),
                             name="Supertrend"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=adl,
                             line=dict(color="cyan"),
                             name="A/D"),row=2,col=1)

    for k,c in [("Entry","cyan"),("SL","red"),("TP1","orange"),("TP2","purple")]:
        fig.add_hline(y=sig[k], line_color=c, row=1)

    fig.update_layout(template="plotly_dark",
                      height=520,
                      xaxis_rangeslider_visible=False)
    return fig

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO v4.1",layout="wide")
st.title("🚀 OPSI A PRO v4.1 — SCORE RANKING")

exchanges = get_exchanges()
tab1,tab2,tab3 = st.tabs(["📡 Live Scan","📜 Riwayat","🎲 Monte Carlo"])

with tab1:
    if st.button("🔍 Scan Live Signal"):
        symbols = get_liquid_symbols(MIN_USDT_VOLUME)
        found=[]
        for s in symbols:
            sig = check_signal(exchanges, s)
            if sig:
                save_signal(sig)
                found.append(sig)
            time.sleep(RATE_LIMIT_DELAY)

        backup_csv(load_signal_history(), SIGNAL_LOG_FILE)
        backup_csv(load_trade_results(), TRADE_RESULT_FILE)

        if found:
            df = pd.DataFrame(found).sort_values("Score", ascending=False)
            st.success(f"🔥 {len(df)} SIGNAL (RANKED)")
            st.dataframe(df, use_container_width=True)

            for _, sig in df.iterrows():
                with st.expander(f"📈 {sig['Symbol']} | {sig['Phase']} | SCORE {sig['Score']}"):
                    d4h,_ = fetch_ohlcv_multi(exchanges, sig["Symbol"], ENTRY_TF, 120)
                    dfc=pd.DataFrame(d4h,columns=["t","open","high","low","close","volume"])
                    stl=supertrend(dfc,ATR_PERIOD,MULTIPLIER)
                    adl=accumulation_distribution(dfc)
                    st.plotly_chart(
                        render_chart(dfc, stl, adl, sig),
                        use_container_width=True
                    )
        else:
            st.warning("Tidak ada setup valid.")

with tab2:
    st.dataframe(load_signal_history().sort_values("Score",ascending=False),
                 use_container_width=True)

with tab3:
    df_r = load_trade_results()
    if len(df_r) < 10:
        st.warning("Trade belum cukup untuk Monte Carlo.")
    else:
        r = df_r["R"].values
        risk = st.slider("Risk / Trade (%)",0.2,3.0,1.0)/100
        trades = st.slider("Trades / Simulation",50,500,300)
        if st.button("🎲 Run Monte Carlo"):
            curves=[]
            for _ in range(500):
                bal=10000; eq=[bal]
                for _ in range(trades):
                    bal+=bal*risk*np.random.choice(r)
                    eq.append(bal)
                curves.append(eq)
            curves=np.array(curves)

            st.metric("Median Final Balance",
                      f"${np.median(curves[:,-1]):,.0f}")
            st.metric("Risk of Ruin (<$5k)",
                      f"{(curves[:,-1]<5000).mean()*100:.2f}%")

            fig=go.Figure()
            for i in range(min(30,len(curves))):
                fig.add_trace(go.Scatter(y=curves[i],mode="lines",
                                         opacity=0.3,showlegend=False))
            fig.update_layout(template="plotly_dark",height=400)
            st.plotly_chart(fig,use_container_width=True)

