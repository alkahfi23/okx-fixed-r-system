import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
import os
import requests
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

NEW_EXPIRE_HOURS = 4 

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
# UTIL
# =====================================================
def normalize_symbol(symbol):
    return symbol.replace("/", "-").upper().strip()

# =====================================================
# FILE HANDLER
# =====================================================
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        df = pd.DataFrame(columns=[
            "Time","CreatedAt","Symbol",
            "Phase","Score","Rating",
            "Entry","SL","TP1","TP2",
            "Status","Label","AutoLabel"
        ])
        df.to_csv(SIGNAL_LOG_FILE, index=False)
    return pd.read_csv(SIGNAL_LOG_FILE)

def save_signal(sig):
    # ⛔ hanya simpan AKUMULASI_KUAT
    if sig.get("Phase") != "AKUMULASI_KUAT":
        return

    df = load_signal_history()

    # anti duplicate OPEN
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
# ANTI DUPLICATE (FINAL)
# =====================================================
def has_open_signal(symbol):
    df = load_signal_history()
    return ((df["Symbol"] == symbol) & (df["Status"] == "OPEN")).any()

def similar_entry_exists(symbol, entry, tolerance=0.002):
    df = load_signal_history()
    rows = df[(df["Symbol"] == symbol) & (df["Status"] == "OPEN")]
    for _, r in rows.iterrows():
        if abs(r["Entry"] - entry) / entry <= tolerance:
            return True
    return False

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

def volume_osc(v, f, s):
    return (v.ewm(span=f).mean() - v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

def accumulation_distribution(df):
    h,l,c,v = df.high, df.low, df.close, df.volume
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

def calculate_score(df4h, df1d):
    score = 0

    # =========================
    # 1️⃣ EMA STRUCTURE (0–4)
    # =========================
    ema20 = df4h.close.ewm(span=20).mean()
    ema50 = df4h.close.ewm(span=50).mean()
    ema200 = df1d.close.ewm(span=200).mean()

    price = df4h.close.iloc[-1]

    if price > ema20.iloc[-1]:
        score += 1
    if ema20.iloc[-1] > ema50.iloc[-1]:
        score += 1
    if ema50.iloc[-1] > ema200.iloc[-1]:
        score += 1
    if price > ema200.iloc[-1]:
        score += 1

    # =========================
    # 2️⃣ VOLUME OSC (0–3)
    # =========================
    vo = volume_osc(df4h.volume, VO_FAST, VO_SLOW)
    vo_last = vo.iloc[-1]

    if vo_last > 5:
        score += 1
    if vo_last > 10:
        score += 1
    if vo_last > 20:
        score += 1

    # =========================
    # 3️⃣ ADL ACCUMULATION (0–3)
    # =========================
    adl = accumulation_distribution(df4h)

    if adl.iloc[-1] > adl.iloc[-5]:
        score += 1
    if adl.iloc[-1] > adl.iloc[-10]:
        score += 1
    if adl.iloc[-1] > adl.iloc[-20]:
        score += 1

    return score


# =====================================================
# AUTO LABEL ENGINE
# =====================================================
def auto_label(row, price, df4h=None):
    """
    Auto label engine with downgrade logic
    """

    # =========================
    # FINAL STATUS (LOCK)
    # =========================
    if row["Status"] in ["TP2 HIT", "SL HIT", "BE HIT"]:
        return "NO REENTRY"

    entry = row["Entry"]
    sl = row["SL"]
    tp1 = row["TP1"]

    # =========================
    # HARD STOP
    # =========================
    if price < sl:
        return "NO REENTRY"

    # =========================
    # RETEST ZONE (±0.3%)
    # =========================
    if abs(price - entry) / entry <= 0.003:
        return "RETEST"

    # =========================
    # HOLD LOGIC
    # =========================
    if price > entry:

        # 🚨 DOWNGRADE RULE 1:
        # Harga sudah terlalu jauh (> +6%)
        if (price - entry) / entry >= 0.06:
            return "NO REENTRY"

        # 🚨 DOWNGRADE RULE 2:
        # Sudah dekat TP1 tapi gagal break
        if price >= tp1 * 0.95 and price < tp1:
            return "NO REENTRY"

        # 🚨 DOWNGRADE RULE 3:
        # EMA20 4H breakdown (butuh df4h)
        if df4h is not None:
            ema20 = df4h.close.ewm(span=20).mean()
            if price < ema20.iloc[-1]:
                return "NO REENTRY"

        return "HOLD"

    # =========================
    # DEFAULT
    # =========================
    return "WAIT"

def update_auto_labels(okx):
    df = load_signal_history()
    changed=False
    for i,row in df.iterrows():
        try:
            price = okx.fetch_ticker(row["Symbol"])["last"]
        except:
            continue
        df4h = pd.DataFrame(
        okx.fetch_ohlcv(row["Symbol"], ENTRY_TF, limit=50),
        columns=["t","open","high","low","close","volume"]
        )
new = auto_label(row, price, df4h)
        if row.get("AutoLabel") != new:
            df.at[i,"AutoLabel"]=new
            changed=True
    if changed:
        df.to_csv(SIGNAL_LOG_FILE,index=False)

# =====================================================
# TP / SL UPDATE
# =====================================================
def update_trade_outcomes(okx):
    df = load_signal_history()
    results = load_trade_results()
    updated=False

    for i,row in df.iterrows():
        if row["Status"]!="OPEN":
            continue
        try:
            price = okx.fetch_ticker(row["Symbol"])["last"]
        except:
            continue

        r=None; status=None
        if price <= row["SL"]:
            r,status=-1,"SL HIT"
        elif price >= row["TP2"]:
            r,status=TP2_R,"TP2 HIT"
        elif price >= row["TP1"]:
            r,status=TP1_R,"TP1 HIT"

        if r is not None:
            df.at[i,"Status"]=status
            results=pd.concat([results,pd.DataFrame([{
                "Time":now_wib(),"Symbol":row["Symbol"],"R":r
            }])],ignore_index=True)
            updated=True

    if updated:
        df.to_csv(SIGNAL_LOG_FILE,index=False)
        results.to_csv(TRADE_RESULT_FILE,index=False)

def restore_signal_csv(uploaded_file):
    if uploaded_file is None:
        return

    try:
        old_df = pd.read_csv(uploaded_file)
        cur_df = load_signal_history()

        # normalisasi kolom
        for col in cur_df.columns:
            if col not in old_df.columns:
                old_df[col] = ""

        old_df = old_df[cur_df.columns]

        merged = pd.concat([cur_df, old_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["Symbol","Entry","CreatedAt"],
            keep="first"
        )

        merged.to_csv(SIGNAL_LOG_FILE, index=False)
        st.success("✅ Signal history berhasil direstore")

    except Exception as e:
        st.error(f"Restore gagal: {e}")

def restore_trade_csv(uploaded_file):
    if uploaded_file is None:
        return

    try:
        old_df = pd.read_csv(uploaded_file)
        cur_df = load_trade_results()

        merged = pd.concat([cur_df, old_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["Symbol","Time","R"],
            keep="first"
        )

        merged.to_csv(TRADE_RESULT_FILE, index=False)
        st.success("✅ Trade result berhasil direstore")

    except Exception as e:
        st.error(f"Restore gagal: {e}")


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
def check_signal(okx, symbol, debug):
    symbol = normalize_symbol(symbol)

    # =========================
    # ANTI DUPLICATE
    # =========================
    if has_open_signal(symbol):
        debug.append({"Symbol": symbol, "Reason": "Already OPEN"})
        return None

    try:
        # =========================
        # FETCH DATA
        # =========================
        df4h = pd.DataFrame(
            okx.fetch_ohlcv(symbol, ENTRY_TF, limit=LIMIT_4H),
            columns=["t","open","high","low","close","volume"]
        )

        df1d = pd.DataFrame(
            okx.fetch_ohlcv(symbol, DAILY_TF, limit=LIMIT_1D),
            columns=["t","open","high","low","close","volume"]
        )

        if len(df4h) < 100 or len(df1d) < 100:
            debug.append({"Symbol": symbol, "Reason": "Data OHLCV kurang"})
            return None

        # =========================
        # INDICATORS
        # =========================
        stl, trend = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
        vo = volume_osc(df4h.volume, VO_FAST, VO_SLOW)
        adl = accumulation_distribution(df4h)

        # =========================
        # HARD FILTER (WAJIB)
        # =========================
        if trend.iloc[-1] != 1:
            debug.append({"Symbol": symbol, "Reason": "Supertrend bearish"})
            return None

        if vo.iloc[-1] < VO_MIN:
            debug.append({"Symbol": symbol, "Reason": "Volume lemah"})
            return None

        if adl.iloc[-1] <= adl.iloc[-10]:
            debug.append({"Symbol": symbol, "Reason": "Tidak ada akumulasi"})
            return None

        ema200 = df1d.close.ewm(span=200).mean()
        if df1d.close.iloc[-1] < ema200.iloc[-1]:
            debug.append({"Symbol": symbol, "Reason": "Below EMA200 Daily"})
            return None

        # =========================
        # SCORE DINAMIS
        # =========================
        score = calculate_score(df4h, df1d)

        if score < 6:
            debug.append({"Symbol": symbol, "Reason": f"Score rendah ({score})"})
            return None

        # =========================
        # ENTRY & SL (SUPPORT BASED)
        # =========================
        entry = df4h.close.iloc[-1]

        supports = find_support(df1d, SR_LOOKBACK)
        supports = [s for s in supports if s < entry]

        if not supports:
            debug.append({"Symbol": symbol, "Reason": "Support tidak valid"})
            return None

        sl = max(supports) * (1 - ZONE_BUFFER)
        risk = entry - sl

        if risk <= entry * 0.002:
            debug.append({"Symbol": symbol, "Reason": "Risk terlalu kecil"})
            return None

        if similar_entry_exists(symbol, entry):
            debug.append({"Symbol": symbol, "Reason": "Duplicate entry zone"})
            return None

        # =========================
        # RATING
        # =========================
        rating = "⭐" * min(score, 10)

        # =========================
        # FINAL SIGNAL (ONLY AKUMULASI_KUAT)
        # =========================
        return {
            "Time": now_wib(),
            "CreatedAt": datetime.now(timezone.utc).isoformat(),
            "Symbol": symbol,
            "Phase": "AKUMULASI_KUAT",
            "Score": score,
            "Rating": rating,
            "Entry": round(entry, 6),
            "SL": round(sl, 6),
            "TP1": round(entry + risk * TP1_R, 6),
            "TP2": round(entry + risk * TP2_R, 6),
            "Status": "OPEN",
            "Label": "NEW",
            "AutoLabel": ""
        }

    except Exception as e:
        debug.append({"Symbol": symbol, "Reason": f"Error: {e}"})
        return None

# =====================================================
# CHART
# =====================================================
def render_chart(df,stl,adl,sig):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3])
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
st.set_page_config("OPSI A PRO — FINAL",layout="wide")
st.title("🚀 OPSI A PRO — FINAL PRODUCTION")

with st.sidebar:
    DEBUG_MODE=st.toggle("🧪 Debug",False)

okx=get_okx()

update_trade_outcomes(okx)
update_auto_labels(okx)

tab1,tab2,tab3,tab4=st.tabs(["📡 Scan","📜 History","🎲 Monte Carlo","🧪 Debug"])

DEBUG_LOG=[]

with tab1:
    if st.button("🔍 Scan Market"):
        syms=get_okx_symbols()
        prog=st.progress(0)
        found=[]
        DEBUG_LOG.clear()

        for i,s in enumerate(syms,1):
            sig=check_signal(okx,s,DEBUG_LOG)
            if sig:
                save_signal(sig)
                found.append(sig)
            prog.progress(i/len(syms))
            time.sleep(RATE_LIMIT_DELAY)

        prog.empty()

        if found:
            df=pd.DataFrame(found).sort_values("Score",ascending=False)
            st.success(f"🔥 {len(df)} SIGNAL")
            st.dataframe(df,use_container_width=True)

            for sig in found:
                with st.expander(f"📈 {sig['Symbol']}"):
                    dfc=pd.DataFrame(
                        okx.fetch_ohlcv(sig["Symbol"],ENTRY_TF,limit=120),
                        columns=["t","open","high","low","close","volume"]
                    )
                    stl,_=supertrend(dfc,ATR_PERIOD,MULTIPLIER)
                    adl=accumulation_distribution(dfc)
                    st.plotly_chart(render_chart(dfc,stl,adl,sig),use_container_width=True)
        else:
            st.warning("Tidak ada setup valid.")

with tab2:
    df = load_signal_history().sort_values("Score", ascending=False)

    with st.expander("♻️ Restore Data Lama"):
        st.markdown("Upload file CSV hasil backup lama")

        sig_file = st.file_uploader(
            "Restore Signal History (signal_history.csv)",
            type=["csv"],
            key="restore_signal"
        )

        trade_file = st.file_uploader(
            "Restore Trade Results (trade_results.csv)",
            type=["csv"],
            key="restore_trade"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("♻️ Restore Signal"):
                restore_signal_csv(sig_file)

        with col2:
            if st.button("♻️ Restore Trade"):
                restore_trade_csv(trade_file)

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        "signal_history.csv"
    )


with tab3:
    df_trades = load_trade_results()
    df_signals = load_signal_history()

    # join trade dengan phase signal
    merged = df_trades.merge(
        df_signals[["Symbol","Phase"]],
        on="Symbol",
        how="left"
    )

    # FILTER AKUMULASI_KUAT SAJA
    df_mc = merged[merged["Phase"] == "AKUMULASI_KUAT"]

    if len(df_mc) < 10:
        st.warning("Trade AKUMULASI_KUAT belum cukup untuk Monte Carlo (min 10).")
    else:
        r = df_mc["R"].values

        risk = st.slider("Risk / Trade (%)", 0.2, 3.0, 1.0) / 100
        trades = st.slider("Trades / Simulation", 50, 500, 300)

        if st.button("🎲 Run Monte Carlo"):
            curves = []

            for _ in range(500):
                bal = 10000
                eq = [bal]
                for _ in range(trades):
                    bal += bal * risk * np.random.choice(r)
                    eq.append(bal)
                curves.append(eq)

            curves = np.array(curves)

            st.metric("Median Balance", f"${np.median(curves[:,-1]):,.0f}")
            st.metric(
                "Risk of Ruin (< $5k)",
                f"{(curves[:,-1] < 5000).mean() * 100:.2f}%"
            )

            fig = go.Figure()
            for i in range(min(30, len(curves))):
                fig.add_trace(
                    go.Scatter(
                        y=curves[i],
                        mode="lines",
                        opacity=0.3,
                        showlegend=False
                    )
                )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)


with tab4:
    if not DEBUG_LOG:
        st.info("Belum ada debug data")
    else:
        st.dataframe(pd.DataFrame(DEBUG_LOG),use_container_width=True)







