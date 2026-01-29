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

def detect_market_regime(df4h, df1d, score_data):
    """
    Return:
    - REGIME_ACCUMULATION
    - REGIME_DISTRIBUTION
    - REGIME_MARKUP
    - REGIME_MARKDOWN
    """

    ema200 = df1d.close.ewm(span=200).mean().iloc[-1]
    price = df1d.close.iloc[-1]

    adl = accumulation_distribution(df4h)
    adl_up = adl.iloc[-1] > adl.iloc[-20]
    adl_down = adl.iloc[-1] < adl.iloc[-20]

    trend_score = score_data.get("TrendScore", 0)
    volume_score = score_data.get("VolumeScore", 0)

    if price < ema200 and adl_down and trend_score < 30:
        return "REGIME_MARKDOWN"

    if price > ema200 and adl_down:
        return "REGIME_DISTRIBUTION"

    if price > ema200 and adl_up and trend_score > 60:
        return "REGIME_MARKUP"

    return "REGIME_ACCUMULATION"

def detect_regime_shift(df4h, df1d, score_data):
    """
    Detects major institutional phase change
    """

    adl = accumulation_distribution(df4h)
    ema200 = df1d.close.ewm(span=200).mean().iloc[-1]
    price = df1d.close.iloc[-1]

    if adl.iloc[-1] < adl.iloc[-30] and price < ema200:
        return {
            "SignalType": "REGIME_SHIFT",
            "Regime": "SHIFT_TO_MARKDOWN",
            "Message": "⚠️ Institutional exit detected (Distribution → Markdown)"
        }

    if adl.iloc[-1] > adl.iloc[-30] and price > ema200:
        return {
            "SignalType": "REGIME_SHIFT",
            "Regime": "SHIFT_TO_MARKUP",
            "Message": "🚀 Institutional accumulation → markup phase"
        }

    return None

def execution_confirmation(df_ltf, direction):
    """
    Final execution filter (15m / 5m logic)
    """

    reasons = []
    close = df_ltf.close
    ema20 = close.ewm(span=20).mean()

    if direction == "LONG":
        if close.iloc[-1] < ema20.iloc[-1]:
            reasons.append("LTF belum reclaim EMA20")
        if close.iloc[-1] < close.iloc[-3]:
            reasons.append("Momentum belum konfirmasi")

    if direction == "SHORT":
        if close.iloc[-1] > ema20.iloc[-1]:
            reasons.append("LTF belum reject EMA20")
        if close.iloc[-1] > close.iloc[-3]:
            reasons.append("Momentum short belum valid")

    return (len(reasons) == 0), reasons
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

def calculate_institutional_score(df4h, df1d, direction="LONG"):
    price = df4h.close.iloc[-1]

    ema20 = df4h.close.ewm(span=20).mean().iloc[-1]
    ema50 = df4h.close.ewm(span=50).mean().iloc[-1]
    ema200 = df1d.close.ewm(span=200).mean().iloc[-1]

    # =========================
    # 1️⃣ MARKET STRUCTURE (40)
    # =========================
    structure = 0

    if direction == "LONG":
        if price > ema20: structure += 15
        if ema20 > ema50: structure += 10
        if ema50 > ema200: structure += 10
        if price > ema200: structure += 5
    else:  # SHORT
        if price < ema20: structure += 15
        if ema20 < ema50: structure += 10
        if ema50 < ema200: structure += 10
        if price < ema200: structure += 5

    structure = min(structure, 40)

    # =========================
    # 2️⃣ VOLUME EXPANSION (30)
    # =========================
    vo = volume_osc(df4h.volume, VO_FAST, VO_SLOW).iloc[-1]

    volume = 0
    if vo > 3: volume += 10
    if vo > 10: volume += 10
    if vo > 20: volume += 10

    volume = min(volume, 30)

    # =========================
    # 3️⃣ ADL FLOW (30)
    # =========================
    adl = accumulation_distribution(df4h)

    adl_score = 0
    if direction == "LONG":
        if adl.iloc[-1] > adl.iloc[-5]: adl_score += 10
        if adl.iloc[-1] > adl.iloc[-10]: adl_score += 10
        if adl.iloc[-1] > adl.iloc[-20]: adl_score += 10
    else:
        if adl.iloc[-1] < adl.iloc[-5]: adl_score += 10
        if adl.iloc[-1] < adl.iloc[-10]: adl_score += 10
        if adl.iloc[-1] < adl.iloc[-20]: adl_score += 10

    adl_score = min(adl_score, 30)

    # =========================
    # FINAL SCORE
    # =========================
    total_score = structure + volume + adl_score

    return {
        "TotalScore": total_score,
        "StructureScore": structure,
        "VolumeScore": volume,
        "ADLScore": adl_score
    }
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

def auto_close_signals(okx):
    """
    Auto close OPEN signals if TP1 / TP2 / SL hit
    Works for SPOT & FUTURES
    """

    df = load_signal_history()
    changed = False

    for i, row in df.iterrows():
        if row["Status"] != "OPEN":
            continue

        try:
            ticker = okx.fetch_ticker(row["Symbol"])
            price = ticker["last"]

            entry = row["Entry"]
            sl = row["SL"]
            tp1 = row["TP1"]
            tp2 = row["TP2"]
            direction = row["Direction"]

            # =========================
            # SL HIT
            # =========================
            if direction == "LONG" and price <= sl:
                df.at[i, "Status"] = "SL HIT"
                changed = True

            elif direction == "SHORT" and price >= sl:
                df.at[i, "Status"] = "SL HIT"
                changed = True

            # =========================
            # TP2 HIT (FINAL)
            # =========================
            elif direction == "LONG" and price >= tp2:
                df.at[i, "Status"] = "TP2 HIT"
                changed = True

            elif direction == "SHORT" and price <= tp2:
                df.at[i, "Status"] = "TP2 HIT"
                changed = True

            # =========================
            # TP1 HIT (PARTIAL)
            # =========================
            elif direction == "LONG" and price >= tp1:
                if row["Status"] == "OPEN":
                    df.at[i, "Status"] = "TP1 HIT"
                    changed = True

            elif direction == "SHORT" and price <= tp1:
                if row["Status"] == "OPEN":
                    df.at[i, "Status"] = "TP1 HIT"
                    changed = True

        except Exception:
            continue

    if changed:
        df.to_csv(SIGNAL_LOG_FILE, index=False)
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
    """
    FULL INSTITUTIONAL CHECK SIGNAL ENGINE

    Return:
    - dict with SignalType:
        • TRADE_EXECUTION
        • MARKET_WARNING
        • REGIME_SHIFT
    - or None
    """

    # =========================
    # 1. FETCH HTF DATA
    # =========================
    try:
        df4h = pd.DataFrame(
            okx.fetch_ohlcv(symbol, "4h", limit=200),
            columns=["t","open","high","low","close","volume"]
        )
        df1d = pd.DataFrame(
            okx.fetch_ohlcv(symbol, "1d", limit=200),
            columns=["t","open","high","low","close","volume"]
        )
    except:
        return None

    entry = df4h.close.iloc[-1]

    # =========================
    # 2. TREND & DIRECTION
    # =========================
    _, trend = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
    direction = "SHORT" if trend.iloc[-1] == -1 else "LONG"

    # =========================
    # 3. INSTITUTIONAL SCORE
    # =========================
    score_data = calculate_institutional_score(
        df4h, df1d, direction=direction
    )
    score = score_data["TotalScore"]

    # =========================
    # 4. SCORE GATE
    # =========================
    if mode == "SPOT" and score < 70:
        return None
    if mode == "FUTURES" and score < 80:
        return None

    # =========================
    # 5. MARKET REGIME
    # =========================
    regime = detect_market_regime(df4h, df1d, score_data)

    # =========================
    # 6. REGIME SHIFT ALERT
    # =========================
    shift = detect_regime_shift(df4h, df1d, score_data)
    if shift:
        return {
            "SignalType": "REGIME_SHIFT",
            "Symbol": symbol,
            "Regime": regime,
            "Details": shift
        }

    # =========================
    # 7. SPOT RULE (NO SHORT)
    # =========================
    if mode == "SPOT" and direction == "SHORT":
        return {
            "SignalType": "MARKET_WARNING",
            "Symbol": symbol,
            "Regime": regime,
            "Message": "SPOT SHORT = DISTRIBUTION WARNING (NO BUY ZONE)"
        }

    # =========================
    # 8. REGIME PERMISSION
    # =========================
    if mode == "SPOT" and regime != "REGIME_ACCUMULATION":
        return {
            "SignalType": "MARKET_WARNING",
            "Symbol": symbol,
            "Regime": regime,
            "Message": "NO BUY ZONE (Institutional Distribution)"
        }

    if mode == "FUTURES":
        if direction == "SHORT" and regime not in ["REGIME_DISTRIBUTION","REGIME_MARKDOWN"]:
            return None
        if direction == "LONG" and regime not in ["REGIME_ACCUMULATION","REGIME_MARKUP"]:
            return None

    # =========================
    # 9. EMA DISTANCE FILTER
    # =========================
    ema20 = df4h.close.ewm(span=20).mean().iloc[-1]

    if direction == "LONG" and entry > ema20 * 1.03:
        return None
    if direction == "SHORT" and entry < ema20 * 0.97:
        return None

    # =========================
    # 10. ADL CONFIRMATION
    # =========================
    adl = accumulation_distribution(df4h)
    if direction == "LONG" and adl.iloc[-1] <= adl.iloc[-10]:
        return None
    if direction == "SHORT" and adl.iloc[-1] >= adl.iloc[-10]:
        return None

    # =========================
    # 11. SL STRUCTURE
    # =========================
    if direction == "LONG":
        supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
        if not supports:
            return None
        sl = max(supports) * (1 - ZONE_BUFFER)
    else:
        resistances = [r for r in find_resistance(df1d, SR_LOOKBACK) if r > entry]
        if not resistances:
            return None
        sl = min(resistances) * (1 + ZONE_BUFFER)

    # =========================
    # 12. EXECUTION CONFIRMATION (15m)
    # =========================
    try:
        df_ltf = pd.DataFrame(
            okx.fetch_ohlcv(symbol, "15m", limit=100),
            columns=["t","open","high","low","close","volume"]
        )
    except:
        return None

    exec_ok, exec_reasons = execution_confirmation(df_ltf, direction)
    if not exec_ok:
        return None

    # =========================
    # 13. TP & PHASE
    # =========================
    if direction == "LONG":
        tp1 = entry + (entry - sl) * TP1_R
        tp2 = entry + (entry - sl) * TP2_R
        phase = "AKUMULASI_INSTITUSI"
    else:
        tp1 = entry - (sl - entry) * TP1_R
        tp2 = entry - (sl - entry) * TP2_R
        phase = "DISTRIBUSI_INSTITUSI"

    # =========================
    # 14. POSITION SIZE
    # =========================
    pos_size = 0.0
    if mode == "FUTURES":
        pos_size = calculate_futures_position(balance, entry, sl)
        if pos_size <= 0:
            return None

    # =========================
    # 15. FINAL SIGNAL
    # =========================
    return {
        "SignalType": "TRADE_EXECUTION",
        "Time": now_wib(),
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Symbol": symbol,
        "Phase": phase,
        "Regime": regime,
        "Score": score,
        "ScoreDetail": score_data,
        "Entry": round(entry, 6),
        "SL": round(sl, 6),
        "TP1": round(tp1, 6),
        "TP2": round(tp2, 6),
        "Mode": mode,
        "Direction": direction,
        "PositionSize": pos_size,
        "ExecutionReasons": exec_reasons
    }

def analyze_single_coin(okx, symbol, mode, balance):
    result = {
        "Symbol": symbol,
        "Mode": mode,
        "Trend": "NO TRADE",
        "Score": 0,
        "ScoreDetail": {},
        "Entry": None,
        "SL": None,
        "TP1": None,
        "TP2": None,
        "PositionSize": None,
        "Reasons": []
    }

    # =========================
    # DATA FETCH
    # =========================
    try:
        df4h = pd.DataFrame(
            okx.fetch_ohlcv(symbol, ENTRY_TF, limit=LIMIT_4H),
            columns=["t","open","high","low","close","volume"]
        )
        df1d = pd.DataFrame(
            okx.fetch_ohlcv(symbol, DAILY_TF, limit=LIMIT_1D),
            columns=["t","open","high","low","close","volume"]
        )
    except Exception as e:
        result["Reasons"].append(f"Data error: {e}")
        return result

    entry = df4h.close.iloc[-1]
    _, trend = supertrend(df4h, ATR_PERIOD, MULTIPLIER)

    direction = "SHORT" if trend.iloc[-1] == -1 else "LONG"

    # =========================
    # INSTITUTIONAL SCORE
    # =========================
    score_data = calculate_institutional_score(
        df4h, df1d, direction=direction
    )

    score = score_data["TotalScore"]
    result["Score"] = score
    result["ScoreDetail"] = score_data

    # =========================
    # SCORE THRESHOLD
    # =========================
    if mode == "FUTURES" and score < 80:
        result["Reasons"].append("Institutional score < 80 (Futures)")
    if mode == "SPOT" and score < 70:
        result["Reasons"].append("Institutional score < 70 (Spot)")

    # =========================
    # TREND FILTER
    # =========================
    if direction == "LONG" and trend.iloc[-1] != 1:
        result["Reasons"].append("Supertrend belum bullish")
    if direction == "SHORT" and trend.iloc[-1] != -1:
        result["Reasons"].append("Supertrend belum bearish")

    # =========================
    # EMA DISTANCE FILTER
    # =========================
    ema20 = df4h.close.ewm(span=20).mean().iloc[-1]

    if direction == "LONG" and entry > ema20 * 1.03:
        result["Reasons"].append("Harga terlalu jauh di atas EMA20 (overextended)")
    if direction == "SHORT" and entry < ema20 * 0.97:
        result["Reasons"].append("Harga terlalu jauh di bawah EMA20 (late short)")

    # =========================
    # ADL CONFIRMATION
    # =========================
    adl = accumulation_distribution(df4h)
    if direction == "LONG" and adl.iloc[-1] <= adl.iloc[-10]:
        result["Reasons"].append("Belum ada akumulasi kuat (ADL)")
    if direction == "SHORT" and adl.iloc[-1] >= adl.iloc[-10]:
        result["Reasons"].append("Belum ada distribusi kuat (ADL)")

    # =========================
    # SL STRUCTURE
    # =========================
    if direction == "LONG":
        supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
        if not supports:
            result["Reasons"].append("Tidak ada support valid untuk SL")
        else:
            sl = max(supports) * (1 - ZONE_BUFFER)

    else:  # SHORT
        resistances = [r for r in find_resistance(df1d, SR_LOOKBACK) if r > entry]
        if not resistances:
            result["Reasons"].append("Tidak ada resistance valid untuk SL")
        else:
            sl = min(resistances) * (1 + ZONE_BUFFER)

    # =========================
    # FINAL DECISION
    # =========================
    if result["Reasons"]:
        return result

    # =========================
    # TP & POSITION SIZE
    # =========================
    if direction == "LONG":
        tp1 = entry + (entry - sl) * TP1_R
        tp2 = entry + (entry - sl) * TP2_R
    else:
        tp1 = entry - (sl - entry) * TP1_R
        tp2 = entry - (sl - entry) * TP2_R

    pos_size = None
    if mode == "FUTURES":
        pos_size = calculate_futures_position(balance, entry, sl)

    result.update({
        "Trend": direction,
        "Entry": round(entry, 6),
        "SL": round(sl, 6),
        "TP1": round(tp1, 6),
        "TP2": round(tp2, 6),
        "PositionSize": pos_size
    })

    return result
# =====================================================
# UI
# =====================================================
st.set_page_config("OPSI A PRO — FINAL CLEAN", layout="wide")
st.title("🚀 OPSI A PRO — SPOT + FUTURES (LONG 100x)")

okx = get_okx()
update_auto_labels(okx)
auto_close_signals(okx)

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

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📡 Scan",
    "📜 History SPOT",
    "🎲 Monte Carlo",
    "⚡ Futures History",
    "🎯 Analisa Coin"
])

with tab1:
    st.subheader("📡 Institutional Market Scanner")

    # =========================
    # 🔄 RESTORE OPEN SIGNAL
    # =========================
    history_df = load_signal_history()
    restored = history_df[
        (history_df["Status"] == "OPEN") &
        (history_df["Mode"] == MODE)
    ]

    found = restored.to_dict("records")

    st.info(
        f"🔄 Restore {len(restored)} OPEN signal dari history "
        f"({MODE})"
    )

    if len(restored) > 0:
        st.dataframe(
            restored.sort_values("Time", ascending=False),
            use_container_width=True
        )

    st.divider()

    # =========================
    # 🚀 START SCAN
    # =========================
    if st.button("🔍 Scan Market"):
        symbols = [
            s for s, m in okx.markets.items()
            if m.get("spot") and s.endswith("/USDT")
        ][:MAX_SCAN_SYMBOLS]

        total = len(symbols)

        progress = st.progress(0.0)
        status_box = st.empty()
        counter_box = st.empty()
        table_box = st.empty()

        start_time = time.time()

        # =========================
        # SCAN LOOP
        # =========================
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

            # =========================
            # HANDLE SIGNAL
            # =========================
            if not sig:
                pass

            elif sig["SignalType"] == "TRADE_EXECUTION":
                save_signal({
                    "Time": sig["Time"],
                    "CreatedAt": sig["CreatedAt"],
                    "Symbol": sig["Symbol"],
                    "Phase": sig["Phase"],
                    "Score": sig["Score"],
                    "Rating": "⭐" * (sig["Score"] // 10),
                    "Entry": sig["Entry"],
                    "SL": sig["SL"],
                    "TP1": sig["TP1"],
                    "TP2": sig["TP2"],
                    "Status": "OPEN",
                    "Label": "INST",
                    "AutoLabel": "WAIT",
                    "Mode": sig["Mode"],
                    "Direction": sig["Direction"],
                    "PositionSize": sig["PositionSize"]
                })

                found.append(sig)

                counter_box.success(
                    f"🔥 Total OPEN Signal: {len(found)}"
                )

                table_box.dataframe(
                    pd.DataFrame(found).tail(10),
                    use_container_width=True
                )

            elif sig["SignalType"] == "MARKET_WARNING":
                status_box.warning(
                    f"⚠️ {s} | {sig['Message']} | Regime: {sig['Regime']}"
                )

            elif sig["SignalType"] == "REGIME_SHIFT":
                status_box.error(
                    f"🚨 REGIME SHIFT {s}\n"
                    f"{sig['Details']['Message']}"
                )

            progress.progress(i / total)
            time.sleep(RATE_LIMIT_DELAY)

        # =========================
        # FINAL OUTPUT
        # =========================
        status_box.success(
            f"✅ Scan selesai | Total OPEN signal: {len(found)}"
        )
        progress.empty()

        if found:
            st.subheader("📌 ALL ACTIVE SIGNALS (RESTORED + NEW)")
            st.dataframe(
                pd.DataFrame(found).sort_values(
                    "Time", ascending=False
                ),
                use_container_width=True
            )
        else:
            st.warning("Tidak ada setup A+ institutional ditemukan.")
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
        
with tab5:
    st.subheader("🎯 Analisa Coin Manual (Logic Sama dengan Scanner)")

    symbols = [
        s for s,m in okx.markets.items()
        if m.get("spot") and s.endswith("/USDT")
    ]

    symbol = st.selectbox("Pilih Coin", symbols)
    mode_an = st.radio("Mode Analisa", ["SPOT","FUTURES"], horizontal=True)
    bal_an = st.number_input("Balance (USDT)", value=10000.0, step=100.0)

    if st.button("🔍 Analyze"):
        res = analyze_single_coin(okx, symbol, mode_an, bal_an)

        st.markdown("## 📊 Hasil Analisa")
        st.markdown(f"### {symbol}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Trend", res["Trend"])
        c2.metric("Score", res["Score"])
        c3.metric("Mode", mode_an)

        if res["Trend"] == "NO TRADE":
            st.error("❌ NO TRADE")
            st.markdown("### 🔎 Alasan Tidak Masuk Kriteria:")
            for r in res["Reasons"]:
                st.write(f"• {r}")
            st.caption("⚠️ Setup belum memenuhi standar A+ system")

        else:
            st.success(f"✅ {res['Trend']} VALID")
            st.markdown("### 📌 Level Trade")
            st.json({
                "Entry": res["Entry"],
                "SL": res["SL"],
                "TP1": res["TP1"],
                "TP2": res["TP2"],
                "Position Size": res["PositionSize"]
            })





















