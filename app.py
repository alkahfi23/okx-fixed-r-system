import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
import os
import requests
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

# =====================================================
# CONFIG
# =====================================================
ENTRY_TF = "4h"
DAILY_TF = "1d"

LIMIT_4H = 200
LIMIT_1D = 200

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28
VO_MIN = 5

SR_LOOKBACK = 5
ZONE_BUFFER = 0.01

TP1_R = 0.8
TP2_R = 2.0

MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

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
# OKX
# =====================================================
@st.cache_resource
def get_okx():
    return ccxt.okx({"enableRateLimit": True})

# =====================================================
# FILE HANDLER
# =====================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        pd.DataFrame(columns=[
            "Time","CreatedAt","Symbol","Phase","Score","Rating",
            "Entry","SL","TP1","TP2","Status","Label","AutoLabel"
        ]).to_csv(SIGNAL_LOG_FILE, index=False)
    return pd.read_csv(SIGNAL_LOG_FILE)

def load_trade_results():
    if not os.path.exists(TRADE_RESULT_FILE):
        pd.DataFrame(columns=["Time","Symbol","R"]).to_csv(
            TRADE_RESULT_FILE, index=False
        )
    return pd.read_csv(TRADE_RESULT_FILE)

def save_signal(sig):
    df = load_signal_history()
    if ((df["Symbol"] == sig["Symbol"]) & (df["Status"] == "OPEN")).any():
        return
    df = pd.concat([df, pd.DataFrame([sig])], ignore_index=True)
    df.to_csv(SIGNAL_LOG_FILE, index=False)

# =====================================================
# RESTORE CSV
# =====================================================
def restore_signal_csv(file):
    if file is None: return
    old = pd.read_csv(file)
    cur = load_signal_history()
    for c in cur.columns:
        if c not in old.columns:
            old[c] = ""
    merged = pd.concat([cur, old[cur.columns]], ignore_index=True)
    merged.drop_duplicates(subset=["Symbol","Entry","CreatedAt"], inplace=True)
    merged.to_csv(SIGNAL_LOG_FILE, index=False)
    st.success("✅ Signal history berhasil direstore")

def restore_trade_csv(file):
    if file is None: return
    old = pd.read_csv(file)
    cur = load_trade_results()
    merged = pd.concat([cur, old], ignore_index=True)
    merged.drop_duplicates(subset=["Symbol","Time","R"], inplace=True)
    merged.to_csv(TRADE_RESULT_FILE, index=False)
    st.success("✅ Trade result berhasil direstore")

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

def volume_osc(v,f,s):
    return (v.ewm(span=f).mean()-v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

def accumulation_distribution(df):
    h,l,c,v = df.high,df.low,df.close,df.volume
    mfm = ((c-l)-(h-c))/(h-l)
    mfm = mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def find_support(df, lb):
    levels=[]
    for i in range(lb,len(df)-lb):
        if df.low.iloc[i]==min(df.low.iloc[i-lb:i+lb+1]):
            levels.append(df.low.iloc[i])
    levels=sorted(set(levels))
    clean=[]
    for s in levels:
        if not clean or abs(s-clean[-1])/s>0.01:
            clean.append(s)
    return clean

# =====================================================
# SCORE
# =====================================================
def calculate_score(df4h, df1d):
    score=0
    ema20=df4h.close.ewm(span=20).mean()
    ema50=df4h.close.ewm(span=50).mean()
    ema200=df1d.close.ewm(span=200).mean()
    price=df4h.close.iloc[-1]

    if price>ema20.iloc[-1]: score+=1
    if ema20.iloc[-1]>ema50.iloc[-1]: score+=1
    if ema50.iloc[-1]>ema200.iloc[-1]: score+=1
    if price>ema200.iloc[-1]: score+=1

    vo=volume_osc(df4h.volume,VO_FAST,VO_SLOW).iloc[-1]
    if vo>5: score+=1
    if vo>10: score+=1
    if vo>20: score+=1

    adl=accumulation_distribution(df4h)
    if adl.iloc[-1]>adl.iloc[-5]: score+=1
    if adl.iloc[-1]>adl.iloc[-10]: score+=1
    if adl.iloc[-1]>adl.iloc[-20]: score+=1
    return score

# =====================================================
# AUTO LABEL
# =====================================================
def auto_label(row, price, df4h):
    if row["Status"] in ["TP2 HIT","SL HIT","BE HIT","CLOSED"]:
        return "NO REENTRY"

    entry=row["Entry"]
    sl=row["SL"]
    tp1=row["TP1"]

    if price < sl:
        return "INVALIDATED"

    if abs(price-entry)/entry <= 0.003:
        return "RETEST"

    if price > entry:
        if (price-entry)/entry >= 0.04:
            return "NO REENTRY"
        if price >= tp1 * 0.95:
            return "NO REENTRY"
        ema20=df4h.close.ewm(span=20).mean().iloc[-1]
        if price < ema20:
            return "NO REENTRY"
        return "HOLD"
    return "WAIT"

# =====================================================
# UPDATE LABEL & STATUS (FINAL)
# =====================================================
def update_auto_labels(okx):
    df = load_signal_history()
    changed=False

    for i,row in df.iterrows():
        try:
            price=okx.fetch_ticker(row["Symbol"])["last"]
            df4h=pd.DataFrame(
                okx.fetch_ohlcv(row["Symbol"],ENTRY_TF,limit=50),
                columns=["t","open","high","low","close","volume"]
            )
            new=auto_label(row,price,df4h)

            if row["AutoLabel"]!=new:
                df.at[i,"AutoLabel"]=new
                changed=True

            if df.at[i,"AutoLabel"] in ["NO REENTRY","INVALIDATED"] and row["Status"]=="OPEN":
                df.at[i,"Status"]="CLOSED"
                changed=True

        except:
            pass

    if changed:
        df.to_csv(SIGNAL_LOG_FILE,index=False)

def update_trade_outcomes(okx):
    df=load_signal_history()
    res=load_trade_results()
    updated=False

    for i,row in df.iterrows():
        if row["Status"]!="OPEN": continue
        try:
            price=okx.fetch_ticker(row["Symbol"])["last"]
        except:
            continue

        r=None; status=None
        if price<=row["SL"]:
            r,status=-1,"SL HIT"
            df.at[i,"AutoLabel"]="INVALIDATED"
        elif price>=row["TP2"]:
            r,status=TP2_R,"TP2 HIT"
        elif price>=row["TP1"]:
            r,status=TP1_R/2,"TP1 HIT"

        if r is not None:
            df.at[i,"Status"]=status
            res=pd.concat([res,pd.DataFrame([{
                "Time":now_wib(),"Symbol":row["Symbol"],"R":r
            }])],ignore_index=True)
            updated=True

    if updated:
        df.to_csv(SIGNAL_LOG_FILE,index=False)
        res.to_csv(TRADE_RESULT_FILE,index=False)

# =====================================================
# SYMBOLS
# =====================================================
@st.cache_data(ttl=300)
def get_okx_symbols():
    r=requests.get(
        "https://www.okx.com/api/v5/market/tickers",
        params={"instType":"SPOT"},timeout=15
    )
    r.raise_for_status()
    syms=[d["instId"] for d in r.json()["data"]
          if d["instId"].endswith("-USDT")
          and float(d["volCcy24h"])>=MIN_USDT_VOLUME]
    np.random.shuffle(syms)
    return syms[:MAX_SCAN_SYMBOLS]

# =====================================================
# SIGNAL LOGIC
# =====================================================
def check_signal(okx,symbol,debug):
    try:
        df4h=pd.DataFrame(
            okx.fetch_ohlcv(symbol,ENTRY_TF,limit=LIMIT_4H),
            columns=["t","open","high","low","close","volume"]
        )
        df1d=pd.DataFrame(
            okx.fetch_ohlcv(symbol,DAILY_TF,limit=LIMIT_1D),
            columns=["t","open","high","low","close","volume"]
        )

        stl,trend=supertrend(df4h,ATR_PERIOD,MULTIPLIER)
        vo=volume_osc(df4h.volume,VO_FAST,VO_SLOW)
        adl=accumulation_distribution(df4h)

        if trend.iloc[-1]!=1: return None
        if vo.iloc[-1]<VO_MIN: return None
        if adl.iloc[-1]<=adl.iloc[-10]: return None

        ema200=df1d.close.ewm(span=200).mean()
        if df1d.close.iloc[-1]<ema200.iloc[-1]: return None

        score=calculate_score(df4h,df1d)
        if score<6: return None

        entry=df4h.close.iloc[-1]
        ema20_4h=df4h.close.ewm(span=20).mean().iloc[-1]
        if entry > ema20_4h * 1.02: return None

        supports=[s for s in find_support(df1d,SR_LOOKBACK) if s<entry]
        if not supports: return None

        sl=max(supports)*(1-ZONE_BUFFER)
        risk=entry-sl
        if risk<=entry*0.002 or risk>=entry*0.06: return None

        return {
            "Time":now_wib(),
            "CreatedAt":datetime.now(timezone.utc).isoformat(),
            "Symbol":symbol,
            "Phase":"AKUMULASI_KUAT",
            "Score":score,
            "Rating":"⭐"*min(score,10),
            "Entry":round(entry,6),
            "SL":round(sl,6),
            "TP1":round(entry+risk*TP1_R,6),
            "TP2":round(entry+risk*TP2_R,6),
            "Status":"OPEN",
            "Label":"NEW",
            "AutoLabel":"WAIT"
        }

    except Exception as e:
        debug.append({"Symbol":symbol,"Reason":str(e)})
        return None

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO — FINAL",layout="wide")
st.title("🚀 OPSI A PRO — FINAL PRODUCTION")

okx=get_okx()
update_trade_outcomes(okx)
update_auto_labels(okx)

tab1,tab2,tab3,tab4=st.tabs(["📡 Scan","📜 History","🎲 Monte Carlo","🧪 Debug"])
DEBUG_LOG=[]

with tab1:
    if st.button("🔍 Scan Market"):
        syms=get_okx_symbols()
        prog=st.progress(0)
        status_box=st.empty()
        found=[]
        DEBUG_LOG.clear()

        total=len(syms)
        for i,s in enumerate(syms,1):
            status_box.info(f"🔄 Scanning **{s}** ({i}/{total}) | 🔥 Found: {len(found)}")
            sig=check_signal(okx,s,DEBUG_LOG)
            if sig:
                save_signal(sig)
                found.append(sig)
            prog.progress(i/total)
            time.sleep(RATE_LIMIT_DELAY)

        status_box.empty()
        prog.empty()

        if found:
            st.success(f"🔥 {len(found)} SIGNAL")
            st.dataframe(pd.DataFrame(found),use_container_width=True)
        else:
            st.warning("Tidak ada setup valid.")

with tab2:
    df=load_signal_history().sort_values("Time",ascending=False)
    with st.expander("♻️ Restore Riwayat (CSV Backup)"):
        sig_file=st.file_uploader("Restore signal_history.csv",type=["csv"])
        trade_file=st.file_uploader("Restore trade_results.csv",type=["csv"])
        c1,c2=st.columns(2)
        with c1:
            if st.button("Restore Signal"):
                restore_signal_csv(sig_file)
        with c2:
            if st.button("Restore Trade"):
                restore_trade_csv(trade_file)
    st.dataframe(df,use_container_width=True)
    st.download_button("⬇️ Download CSV",df.to_csv(index=False),"signal_history.csv")

with tab3:
    tr=load_trade_results()
    sig=load_signal_history()
    mc=tr.merge(sig[["Symbol","Phase"]],on="Symbol",how="left")
    mc=mc[mc["Phase"]=="AKUMULASI_KUAT"]

    if len(mc)<10:
        st.warning("Trade AKUMULASI_KUAT belum cukup")
    else:
        r=mc["R"].values
        risk=st.slider("Risk / Trade (%)",0.2,3.0,1.0)/100
        trades=st.slider("Trades / Simulation",50,500,300)

        if st.button("🎲 Run Monte Carlo"):
            curves=[]
            for _ in range(500):
                bal=10000
                eq=[bal]
                for _ in range(trades):
                    bal+=bal*risk*np.random.choice(r)
                    eq.append(bal)
                curves.append(eq)

            curves=np.array(curves)

            st.metric("Median Balance",f"${np.median(curves[:,-1]):,.0f}")
            st.metric("Risk of Ruin (<$5k)",f"{(curves[:,-1]<5000).mean()*100:.2f}%")

            fig=go.Figure()
            for i in range(min(30,len(curves))):
                fig.add_trace(go.Scatter(y=curves[i],mode="lines",opacity=0.3))
            fig.update_layout(template="plotly_dark",height=400)
            st.plotly_chart(fig,use_container_width=True)

with tab4:
    if DEBUG_LOG:
        st.dataframe(pd.DataFrame(DEBUG_LOG),use_container_width=True)
    else:
        st.info("Belum ada debug")
