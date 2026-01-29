import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
import os
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
VO_MIN = 3

SR_LOOKBACK = 5
ZONE_BUFFER = 0.01

TP1_R = 0.8
TP2_R = 2.0

RATE_LIMIT_DELAY = 0.15
MAX_SCAN_SYMBOLS = 120

# ===== FUTURES (LONG ONLY – 100x SAFE MODE) =====
FUTURES_RISK_PCT = 0.005     # 0.5% real risk
FUTURES_LEVERAGE = 100
FUTURES_MAX_RISK = 0.015    # SL max 1.5%

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_LOG_FILE = os.path.join(BASE_DIR, "signal_history.csv")
TRADE_RESULT_FILE = os.path.join(BASE_DIR, "trade_results.csv")
FUTURES_TRADE_FILE = os.path.join(BASE_DIR, "futures_trades.csv")

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))
def now_wib():
    return datetime.now(timezone.utc).astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")

def get_wib_hour():
    return datetime.now(timezone.utc).astimezone(WIB).hour

def is_danger_time():
    h = get_wib_hour()
    # jam bahaya: midnight – subuh
    return h >= 0 and h < 5

def is_safe_futures_time():
    h = get_wib_hour()
    return 19 <= h <= 23

def is_safe_spot_time():
    h = get_wib_hour()
    return (7 <= h <= 10) or (19 <= h <= 23)

# =====================================================
# OKX
# =====================================================
@st.cache_resource
def get_okx():
    ex = ccxt.okx({"enableRateLimit": True})
    ex.load_markets()
    return ex

# =====================================================
# FILE HANDLER (AUTO MIGRATION)
# =====================================================
def load_signal_history():
    cols = [
        "Time","CreatedAt","Symbol","Phase","Score","Rating",
        "Entry","SL","TP1","TP2",
        "Status","Label","AutoLabel",
        "Mode","Direction","PositionSize"
    ]
    if not os.path.exists(SIGNAL_LOG_FILE):
        pd.DataFrame(columns=cols).to_csv(SIGNAL_LOG_FILE, index=False)

    df = pd.read_csv(SIGNAL_LOG_FILE)

    for c in cols:
        if c not in df.columns:
            if c == "Mode":
                df[c] = "SPOT"
            elif c == "Direction":
                df[c] = "LONG"
            elif c == "PositionSize":
                df[c] = 0.0
            else:
                df[c] = ""

    df.to_csv(SIGNAL_LOG_FILE, index=False)
    return df

def load_trade_results():
    if not os.path.exists(TRADE_RESULT_FILE):
        pd.DataFrame(columns=["Time","Symbol","R"]).to_csv(
            TRADE_RESULT_FILE, index=False
        )
    return pd.read_csv(TRADE_RESULT_FILE)

def load_futures_trades():
    if not os.path.exists(FUTURES_TRADE_FILE):
        pd.DataFrame(columns=[
            "Time","Symbol","Direction",
            "Entry","Exit","SL",
            "Size","PnL_USDT","PnL_PCT","Reason"
        ]).to_csv(FUTURES_TRADE_FILE,index=False)
    return pd.read_csv(FUTURES_TRADE_FILE)

def save_signal(sig):
    df = load_signal_history()
    if ((df["Symbol"] == sig["Symbol"]) &
        (df["Status"] == "OPEN") &
        (df["Mode"] == sig["Mode"])).any():
        return
    df = pd.concat([df, pd.DataFrame([sig])], ignore_index=True)
    df.to_csv(SIGNAL_LOG_FILE, index=False)

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h,l,c = df.high,df.low,df.close
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

    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = 1 if c.iloc[i] > stl.iloc[i] else -1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = -1 if c.iloc[i] < stl.iloc[i] else 1

    return stl, trend

def volume_osc(v,f,s):
    return (v.ewm(span=f).mean() - v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

def accumulation_distribution(df):
    h,l,c,v = df.high,df.low,df.close,df.volume
    mfm = ((c-l)-(h-c))/(h-l)
    mfm = mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def find_support(df, lb):
    levels=[]
    for i in range(lb, len(df)-lb):
        if df.low.iloc[i] == min(df.low.iloc[i-lb:i+lb+1]):
            levels.append(df.low.iloc[i])
    return sorted(set(levels))
    
def find_resistance(df, lb):
    levels = []
    for i in range(lb, len(df)-lb):
        if df.high.iloc[i] == max(df.high.iloc[i-lb:i+lb+1]):
            levels.append(df.high.iloc[i])
    return sorted(set(levels))

# =====================================================
# SCORE
# =====================================================
def calculate_score(df4h, df1d):
    score = 0
    ema20 = df4h.close.ewm(span=20).mean()
    ema50 = df4h.close.ewm(span=50).mean()
    ema200 = df1d.close.ewm(span=200).mean()
    price = df4h.close.iloc[-1]

    if price > ema20.iloc[-1]: score+=1
    if ema20.iloc[-1] > ema50.iloc[-1]: score+=1
    if ema50.iloc[-1] > ema200.iloc[-1]: score+=1
    if price > ema200.iloc[-1]: score+=1

    vo = volume_osc(df4h.volume, VO_FAST, VO_SLOW).iloc[-1]
    if vo > 3: score+=1
    if vo > 10: score+=1
    if vo > 20: score+=1

    adl = accumulation_distribution(df4h)
    if adl.iloc[-1] > adl.iloc[-5]: score+=1
    if adl.iloc[-1] > adl.iloc[-10]: score+=1
    if adl.iloc[-1] > adl.iloc[-20]: score+=1

    return score

def calculate_score_short(df4h, df1d):
    score = 0

    ema20 = df4h.close.ewm(span=20).mean()
    ema50 = df4h.close.ewm(span=50).mean()
    ema200 = df1d.close.ewm(span=200).mean()
    price = df4h.close.iloc[-1]

    # Trend & structure (mirror long)
    if price < ema20.iloc[-1]: score += 1
    if ema20.iloc[-1] < ema50.iloc[-1]: score += 1
    if ema50.iloc[-1] < ema200.iloc[-1]: score += 1
    if price < ema200.iloc[-1]: score += 1

    # Volume expansion
    vo = volume_osc(df4h.volume, VO_FAST, VO_SLOW).iloc[-1]
    if vo > 3: score += 1
    if vo > 10: score += 1
    if vo > 20: score += 1

    # Distribution (reverse ADL)
    adl = accumulation_distribution(df4h)
    if adl.iloc[-1] < adl.iloc[-5]: score += 1
    if adl.iloc[-1] < adl.iloc[-10]: score += 1
    if adl.iloc[-1] < adl.iloc[-20]: score += 1

    return score

# =====================================================
# AUTO LABEL (SPOT ONLY)
# =====================================================
def auto_label(row, price, df4h):
    if row["Status"] in ["TP2 HIT","SL HIT","CLOSED"]:
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
        if price >= tp1*0.95:
            return "NO REENTRY"
        ema20=df4h.close.ewm(span=20).mean().iloc[-1]
        if price < ema20:
            return "NO REENTRY"
        return "HOLD"

    return "WAIT"

def update_auto_labels(okx):
    df = load_signal_history()
    changed=False

    for i,row in df.iterrows():
        if row["Mode"]!="SPOT" or row["Status"]!="OPEN":
            continue
        try:
            price = okx.fetch_ticker(row["Symbol"])["last"]
            df4h = pd.DataFrame(
                okx.fetch_ohlcv(row["Symbol"], ENTRY_TF, limit=50),
                columns=["t","open","high","low","close","volume"]
            )
            new = auto_label(row, price, df4h)
            if row["AutoLabel"] != new:
                df.at[i,"AutoLabel"] = new
                changed=True
            if new in ["NO REENTRY","INVALIDATED"]:
                df.at[i,"Status"] = "CLOSED"
                changed=True
        except:
            pass

    if changed:
        df.to_csv(SIGNAL_LOG_FILE,index=False)

# =====================================================
# FUTURES RISK ENGINE
# =====================================================
def calculate_futures_position(balance, entry, sl):
    stop_pct = abs(entry - sl) / entry
    if stop_pct <= 0 or stop_pct > FUTURES_MAX_RISK:
        return 0.0
    risk_amount = balance * FUTURES_RISK_PCT
    pos_value = risk_amount / stop_pct
    max_position = balance * FUTURES_LEVERAGE * 0.5
    return round(min(pos_value, max_position),2)

# =====================================================
# SIGNAL LOGIC
# =====================================================
def check_signal(okx, symbol, mode, balance):
    try:
        df4h = pd.DataFrame(
            okx.fetch_ohlcv(symbol, ENTRY_TF, limit=LIMIT_4H),
            columns=["t","open","high","low","close","volume"]
        )
        df1d = pd.DataFrame(
            okx.fetch_ohlcv(symbol, DAILY_TF, limit=LIMIT_1D),
            columns=["t","open","high","low","close","volume"]
        )
    except:
        return None

    stl, trend = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
    entry = df4h.close.iloc[-1]

    # =========================
    # ===== FUTURES SHORT =====
    # =========================
    if mode == "FUTURES" and trend.iloc[-1] == -1:
        score = calculate_score_short(df4h, df1d)
        if score <= 8:
            return None

        ema20 = df4h.close.ewm(span=20).mean().iloc[-1]
        if entry < ema20 * 0.97:   # avoid late short
            return None

        resistances = [r for r in find_resistance(df1d, SR_LOOKBACK) if r > entry]
        if not resistances:
            return None

        sl = min(resistances) * (1 + ZONE_BUFFER)
        risk = abs(sl - entry) / entry
        if risk <= 0.002 or risk >= FUTURES_MAX_RISK:
            return None

        pos_size = calculate_futures_position(balance, entry, sl)

        return {
            "Time": now_wib(),
            "CreatedAt": datetime.now(timezone.utc).isoformat(),
            "Symbol": symbol,
            "Phase": "DISTRIBUSI_KUAT",
            "Score": score,
            "Rating": "⭐"*score,
            "Entry": round(entry,6),
            "SL": round(sl,6),
            "TP1": round(entry - (sl-entry)*TP1_R,6),
            "TP2": round(entry - (sl-entry)*TP2_R,6),
            "Status": "OPEN",
            "Label": "NEW",
            "AutoLabel": "WAIT",
            "Mode": "FUTURES",
            "Direction": "SHORT",
            "PositionSize": pos_size
        }

    # =========================
    # ===== LONG (SPOT & FUT) ==
    # =========================
    if trend.iloc[-1] != 1:
        return None

    if volume_osc(df4h.volume, VO_FAST, VO_SLOW).iloc[-1] < VO_MIN:
        return None

    adl = accumulation_distribution(df4h)
    if adl.iloc[-1] <= adl.iloc[-10]:
        return None

    ema200 = df1d.close.ewm(span=200).mean()
    if df1d.close.iloc[-1] < ema200.iloc[-1]:
        return None

    score = calculate_score(df4h, df1d)

    if mode == "FUTURES":
        if score <= 8:
            return None
    else:
        if score < 6:
            return None

    ema20 = df4h.close.ewm(span=20).mean().iloc[-1]
    if entry > ema20 * 1.03:
        return None

    supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
    if not supports:
        return None

    sl = max(supports) * (1 - ZONE_BUFFER)
    risk = abs(entry - sl) / entry
    if risk <= 0.002 or risk >= FUTURES_MAX_RISK:
        return None

    pos_size = 0.0
    if mode == "FUTURES":
        pos_size = calculate_futures_position(balance, entry, sl)

    return {
        "Time": now_wib(),
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Symbol": symbol,
        "Phase": "AKUMULASI_KUAT",
        "Score": score,
        "Rating": "⭐"*score,
        "Entry": round(entry,6),
        "SL": round(sl,6),
        "TP1": round(entry+(entry-sl)*TP1_R,6),
        "TP2": round(entry+(entry-sl)*TP2_R,6),
        "Status": "OPEN",
        "Label": "NEW",
        "AutoLabel": "WAIT",
        "Mode": mode,
        "Direction": "LONG",
        "PositionSize": pos_size
    }

# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO — FINAL CLEAN", layout="wide")
st.title("🚀 OPSI A PRO — SPOT + FUTURES (LONG 100x)")

okx = get_okx()
update_auto_labels(okx)

MODE = st.radio("🧭 Trading Mode", ["SPOT","FUTURES"], horizontal=True)
current_hour = get_wib_hour()

# 🔴 JAM BAHAYA UNTUK SEMUA MODE
if is_danger_time():
    st.error(
        f"⛔ JAM BAHAYA ({current_hour}:00 WIB)\n\n"
        "• Likuiditas rendah\n"
        "• Risiko fake move tinggi\n"
        "• Tidak ideal untuk entry\n\n"
        "Jam ideal:\n"
        "07:00–10:00 atau 19:00–23:00 WIB"
    )

# 🟡 WARNING FUTURES
elif MODE == "FUTURES" and not is_safe_futures_time():
    st.warning(
        f"⚠️ Futures di luar jam optimal ({current_hour}:00 WIB)\n\n"
        "Jam terbaik Futures:\n"
        "19:00–23:00 WIB\n\n"
        "Gunakan hanya untuk A+ setup"
    )

# 🟡 WARNING SPOT
elif MODE == "SPOT" and not is_safe_spot_time():
    st.warning(
        f"⚠️ SPOT di luar jam ideal ({current_hour}:00 WIB)\n\n"
        "SPOT paling optimal saat:\n"
        "07:00–10:00 dan 19:00–23:00 WIB\n\n"
        "Entry masih boleh, tapi tunggu konfirmasi lebih kuat"
    )
BALANCE = st.number_input("💰 Account Balance (USDT)", value=10000.0, step=100.0)

if MODE == "FUTURES":
    st.warning(
        "⚠️ FUTURES 100x MODE\n"
        "• Score ≥ 9 only\n"
        "• Max SL 1.5%\n"
        "• Risk real 0.5%\n"
        "• A+ setup only"
    )

tab1,tab2,tab3,tab4 = st.tabs([
    "📡 Scan",
    "📜 History SPOT",
    "🎲 Monte Carlo",
    "⚡ Futures History"
])

with tab1:
    if st.button("🔍 Scan Market"):
        symbols = [
            s for s,m in okx.markets.items()
            if m.get("spot") and s.endswith("/USDT")
        ][:MAX_SCAN_SYMBOLS]

        total = len(symbols)
        found = []

        # UI elements
        progress = st.progress(0.0)
        status_box = st.empty()
        counter_box = st.empty()
        table_box = st.empty()

        start_time = time.time()

        for i, s in enumerate(symbols, 1):
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            eta = avg_time * (total - i)

            status_box.info(
                f"🔄 **Scanning:** `{s}`  \n"
                f"📊 Progress: {i}/{total}  \n"
                f"⏱ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
            )

            sig = check_signal(okx, s, MODE, BALANCE)
            if sig:
                save_signal(sig)
                found.append(sig)
                counter_box.success(f"🔥 Signal Found: {len(found)}")

                # tampilkan sementara
                table_box.dataframe(
                    pd.DataFrame(found).tail(5),
                    use_container_width=True
                )

            progress.progress(i / total)
            time.sleep(RATE_LIMIT_DELAY)

        status_box.success(f"✅ Scan selesai | Total signal: {len(found)}")
        progress.empty()

        if found:
            st.subheader("📌 Final Signals")
            st.dataframe(pd.DataFrame(found), use_container_width=True)
        else:
            st.warning("Tidak ada setup valid ditemukan.")

with tab2:
    df = load_signal_history()
    st.dataframe(df[df["Mode"]=="SPOT"].sort_values("Time", ascending=False),
                 use_container_width=True)

with tab3:
    tr = load_trade_results()
    sig = load_signal_history()
    mc = tr.merge(sig[["Symbol","Phase","Mode"]], on="Symbol", how="left")
    mc = mc[(mc["Phase"]=="AKUMULASI_KUAT") & (mc["Mode"]=="SPOT")]

    if len(mc) < 10:
        st.warning("Trade SPOT belum cukup")
    else:
        r = mc["R"].values
        risk = st.slider("Risk / Trade (%)", 0.2, 3.0, 1.0)/100
        trades = st.slider("Trades / Simulation", 50, 500, 300)

        if st.button("🎲 Run Monte Carlo"):
            curves=[]
            for _ in range(500):
                bal=10000
                eq=[bal]
                for _ in range(trades):
                    bal += bal * risk * np.random.choice(r)
                    eq.append(bal)
                curves.append(eq)
            curves=np.array(curves)
            st.metric("Median Balance", f"${np.median(curves[:,-1]):,.0f}")
            st.metric("Risk of Ruin (<$5k)", f"{(curves[:,-1]<5000).mean()*100:.2f}%")

            fig=go.Figure()
            for i in range(min(30,len(curves))):
                fig.add_trace(go.Scatter(y=curves[i], mode="lines", opacity=0.3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    df = load_futures_trades()
    if df.empty:
        st.info("Belum ada trade futures")
    else:
        st.metric("Total PnL (USDT)", f"${df['PnL_USDT'].sum():,.2f}")
        st.metric("Win Rate", f"{(df['PnL_USDT']>0).mean()*100:.2f}%")
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download Futures Trades",
                           df.to_csv(index=False),
                           "futures_trades.csv")





