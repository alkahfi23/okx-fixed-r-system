import streamlit as st
import ccxt
import pandas as pd
import requests
import time
import os
import random
import json
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

SR_LOOKBACK = 5
ZONE_BUFFER = 0.008

MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

TP1_PARTIAL_R = 0.5
TP2_FINAL_R   = 1.5

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
# GOOGLE DRIVE
# =====================================================
def get_drive_service():
    raw = os.getenv("GDRIVE_SERVICE_JSON")
    if not raw:
        return None
    info = json.loads(raw)
    creds = Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def backup_csv(df, filename):
    service = get_drive_service()
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if not service or not folder_id:
        return

    data = df.to_csv(index=False).encode("utf-8")
    media = MediaInMemoryUpload(data, mimetype="text/csv")

    # HAPUS FILE LAMA (BY NAME)
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()
    for f in res.get("files", []):
        try:
            service.files().delete(fileId=f["id"]).execute()
        except:
            pass

    # CREATE FILE BARU (AMAN)
    service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media
    ).execute()



def restore_csv(filename, path):
    service = get_drive_service()
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if not service or not folder_id:
        return

    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])

    if not files:
        return

    content = service.files().get_media(fileId=files[0]["id"]).execute()
    with open(path, "wb") as f:
        f.write(content)

# =====================================================
# RESTORE DATA (ANTI HILANG)
# =====================================================
restore_csv("signal_history.csv", SIGNAL_LOG_FILE)
restore_csv("trade_results.csv", TRADE_RESULT_FILE)

# =====================================================
# FILE HANDLERS
# =====================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        df = pd.DataFrame(columns=[
            "Time","Symbol","Phase","Candle",
            "Entry","SL","TP1","TP2",
            "Priority","Rating","Status","Label"
        ])
        df.to_csv(SIGNAL_LOG_FILE, index=False)
    return pd.read_csv(SIGNAL_LOG_FILE)

def save_signal(signal):
    df = load_signal_history()
    if ((df["Symbol"] == signal["Symbol"]) & (df["Status"] == "OPEN")).any():
        return
    df = pd.concat([df, pd.DataFrame([signal])], ignore_index=True)
    df.to_csv(SIGNAL_LOG_FILE, index=False)
    backup_csv(df, "signal_history.csv")

def load_trade_results():
    if not os.path.exists(TRADE_RESULT_FILE):
        pd.DataFrame(columns=["Time","Symbol","R"]).to_csv(TRADE_RESULT_FILE, index=False)
    return pd.read_csv(TRADE_RESULT_FILE)

# =====================================================
# CCXT
# =====================================================
@st.cache_resource
def get_okx():
    return ccxt.okx({"enableRateLimit": True})

# =====================================================
# UPDATE TRADE OUTCOMES (ONCE / SESSION)
# =====================================================
def update_trade_outcomes(okx):
    history = load_signal_history()
    if history.empty:
        return

    results = load_trade_results()
    updated = False

    for i, row in history.iterrows():
        if row["Status"] in ["SL HIT", "TP2 HIT", "BE HIT"]:
            continue

        try:
            price = okx.fetch_ticker(row["Symbol"])["last"]
        except:
            continue

        r = None
        status = None

        if row["Status"] == "OPEN" and price <= row["SL"]:
            r, status = -1, "SL HIT"
        elif row["Status"] == "OPEN" and price >= row["TP1"]:
            r, status = TP1_PARTIAL_R, "TP1 HIT"
        elif row["Status"] == "TP1 HIT" and price <= row["Entry"]:
            r, status = 0.0, "BE HIT"
        elif row["Status"] == "TP1 HIT" and price >= row["TP2"]:
            r, status = TP2_FINAL_R, "TP2 HIT"

        if r is not None:
            history.at[i, "Status"] = status
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
        backup_csv(history, "signal_history.csv")
        backup_csv(results, "trade_results.csv")

# =====================================================
# MARKET SYMBOLS
# =====================================================
@st.cache_data(ttl=300)
def get_liquid_symbols(min_vol):
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
    return random.sample(syms, min(MAX_SCAN_SYMBOLS, len(syms)))

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

def volume_oscillator(v,f,s):
    return (v.ewm(span=f).mean()-v.ewm(span=s).mean())/v.ewm(span=s).mean()*100

def accumulation_distribution(df):
    h,l,c,v=df.high,df.low,df.close,df.volume
    mfm=((c-l)-(h-c))/(h-l)
    mfm=mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def ad_phase(adl):
    return "AKUMULASI_KUAT" if adl.iloc[-1] > adl.iloc[-10] else "NETRAL"

# =====================================================
# SIGNAL CHECK
# =====================================================
def check_signal(okx, symbol):
    df4h=pd.DataFrame(okx.fetch_ohlcv(symbol,ENTRY_TF,limit=LIMIT_4H),
        columns=["t","open","high","low","close","volume"])
    df1d=pd.DataFrame(okx.fetch_ohlcv(symbol,SR_TF,limit=LIMIT_1D),
        columns=["t","open","high","low","close","volume"])

    stl,trend=supertrend(df4h,ATR_PERIOD,MULTIPLIER)
    vo=volume_oscillator(df4h.volume,VO_FAST,VO_SLOW)
    adl=accumulation_distribution(df4h)
    phase=ad_phase(adl)

    if trend.iloc[-1]!=1 or vo.iloc[-1]<VO_MIN or phase!="AKUMULASI_KUAT":
        return None

    ema200=df1d.close.ewm(span=200).mean()
    if df1d.close.iloc[-1]<ema200.iloc[-1]:
        return None

    entry=df4h.close.iloc[-1]
    sl=df1d.low.min()*(1-ZONE_BUFFER)
    risk=entry-sl
    if risk<=entry*0.002:
        return None

    return {
        "Time":now_wib(),
        "Symbol":symbol,
        "Phase":phase,
        "Candle":"Normal",
        "Entry":round(entry,8),
        "SL":round(sl,8),
        "TP1":round(entry+risk*0.8,8),
        "TP2":round(entry+risk*2.0,8),
        "Priority":4,
        "Rating":"⭐⭐⭐⭐",
        "Status":"OPEN",
        "Label":"🆕 NEW"
    }

# =====================================================
# CHART
# =====================================================
def get_chart_data(okx, symbol):
    df = pd.DataFrame(
        okx.fetch_ohlcv(symbol, ENTRY_TF, limit=120),
        columns=["t","open","high","low","close","volume"]
    )
    stl,_ = supertrend(df, ATR_PERIOD, MULTIPLIER)
    adl = accumulation_distribution(df)
    return df, stl, adl

def render_chart(df, stl, adl, signal):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
    fig.add_candlestick(x=df.index, open=df.open, high=df.high,
                        low=df.low, close=df.close, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=stl, name="Supertrend",
                             line=dict(color="lime")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=adl, name="A/D",
                             line=dict(color="cyan")), row=2, col=1)

    for k,c in [("Entry","cyan"),("SL","red"),("TP1","orange"),("TP2","purple")]:
        fig.add_hline(y=signal[k], line_color=c, row=1)

    fig.update_layout(template="plotly_dark", height=520,
                      xaxis_rangeslider_visible=False)
    return fig

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO v3.7", layout="wide")
st.title("🚀 OPSI A PRO v3.7 — NEW SIGNAL + CHART")

okx = get_okx()

if "trade_updated" not in st.session_state:
    update_trade_outcomes(okx)
    st.session_state.trade_updated = True

st.session_state.setdefault("scan_results", [])

tab1, tab2 = st.tabs(["📡 Live Scan","📜 Riwayat"])

with tab1:
    if st.button("🔍 Scan Live Signal"):
        st.session_state.scan_results = []

        symbols = get_liquid_symbols(MIN_USDT_VOLUME)
        progress = st.progress(0)
        status = st.empty()

        for i, s in enumerate(symbols, 1):
            status.text(f"Scanning {s} ({i}/{len(symbols)})")
            try:
                sig = check_signal(okx, s)
                if sig:
                    save_signal(sig)
                    st.session_state.scan_results.append(sig)
            except Exception as e:
                st.write(f"{s} error → {e}")

            progress.progress(i / len(symbols))
            time.sleep(RATE_LIMIT_DELAY)

        # 🔽 INI POSISI YANG BENAR
        progress.empty()
        status.empty()

        # ✅ BACKUP SEKALI (ANTI 404)
        backup_csv(load_signal_history(), "signal_history.csv")
        backup_csv(load_trade_results(), "trade_results.csv")

        # ==========================
        # DISPLAY RESULT
        # ==========================
        if st.session_state.scan_results:
            st.success(f"🔥 {len(st.session_state.scan_results)} NEW SIGNAL")
            df_new = pd.DataFrame(st.session_state.scan_results)
            st.dataframe(df_new, use_container_width=True)

            for sig in st.session_state.scan_results:
                with st.expander(f"📈 {sig['Symbol']} — Chart"):
                    dfc, stlc, adlc = get_chart_data(okx, sig["Symbol"])
                    st.plotly_chart(
                        render_chart(dfc, stlc, adlc, sig),
                        use_container_width=True
                    )
        else:
            st.warning("Tidak ada setup valid.")

with tab2:
    st.dataframe(load_signal_history(), use_container_width=True)


