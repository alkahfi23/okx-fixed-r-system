import streamlit as st
import ccxt
import pandas as pd
import requests
import time
import plotly.graph_objects as go

# =====================================================
# CONFIG
# =====================================================
ENTRY_TF = "4h"
SR_TF = "1d"

LIMIT_4H = 200
LIMIT_1D = 200
BACKTEST_LIMIT = 600
MAX_FORWARD = 50

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28

SR_LOOKBACK = 5
ZONE_BUFFER = 0.008

MIN_USDT_VOLUME = 2_000_000
RATE_LIMIT_DELAY = 0.15

VALID_CANDLES = {"Bullish Engulfing", "Hammer", "Strong Bullish"}

# =====================================================
# SESSION STATE
# =====================================================
for k in ["scanned", "results", "chart_data"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "scanned" else False

# =====================================================
# HELPERS
# =====================================================
def fmt_price(x):
    if x >= 1:
        return f"{x:.4f}"
    elif x >= 0.01:
        return f"{x:.6f}"
    else:
        return f"{x:.8f}"

@st.cache_data(ttl=300)
def get_liquid_symbols(min_vol):
    url = "https://www.okx.com/api/v5/market/tickers"
    r = requests.get(url, params={"instType": "SPOT"}, timeout=15)
    r.raise_for_status()
    return [
        d["instId"]
        for d in r.json()["data"]
        if d["instId"].endswith("-USDT")
        and float(d["volCcy24h"]) >= min_vol
    ]

# =====================================================
# TELEGRAM
# =====================================================
def send_telegram(msg):
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(
            url,
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
    except:
        pass

def format_alert(symbol, candle, entry, sl, tp, risk, vo):
    return f"""
🚀 *FIXED-R SIGNAL*

*Symbol:* `{symbol}`
*Candle:* {candle}

*Entry:* `{fmt_price(entry)}`
*SL:* `{fmt_price(sl)}`
*TP (1R):* `{fmt_price(tp)}`

*Risk:* {risk}%
*VO:* {vo}

⏰ TF: 4H  
📌 Entry on candle close
"""

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h, l, c = df.high, df.low, df.close
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (h + l) / 2

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    stl = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)

    stl.iloc[0], trend.iloc[0] = upper.iloc[0], -1

    for i in range(1, len(df)):
        if c.iloc[i] > stl.iloc[i-1]:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = 1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = -1

    return stl, trend


def volume_oscillator(v, f, s):
    ef = v.ewm(span=f, adjust=False).mean()
    es = v.ewm(span=s, adjust=False).mean()
    return (ef - es) / es * 100

# =====================================================
# PRICE KNOWLEDGE
# =====================================================
def detect_candle(df):
    o, h, l, c = df.open, df.high, df.low, df.close
    po, pc = o.shift(1), c.shift(1)
    body = abs(c - o)
    rng = h - l

    if c.iloc[-1] > o.iloc[-1] and pc.iloc[-1] < po.iloc[-1] and c.iloc[-1] > po.iloc[-1]:
        return "Bullish Engulfing"
    if c.iloc[-1] > o.iloc[-1] and (o.iloc[-1] - l.iloc[-1]) > 2 * body.iloc[-1]:
        return "Hammer"
    if body.iloc[-1] / rng.iloc[-1] > 0.65 and c.iloc[-1] > o.iloc[-1]:
        return "Strong Bullish"
    return "Normal"

# =====================================================
# SUPPORT
# =====================================================
def find_support(df, lb):
    supports = []
    for i in range(lb, len(df) - lb):
        if df.low.iloc[i] == min(df.low.iloc[i-lb:i+lb+1]):
            supports.append(df.low.iloc[i])
    return sorted(set(supports))

# =====================================================
# ENTRY + TRADE
# =====================================================
def valid_entry(df, stl, trend, vo):
    return trend.iloc[-1] == 1 and trend.iloc[-2] == -1 and vo.iloc[-1] >= 5


def build_trade_fixed_r(df4h, df1d):
    entry = df4h.close.iloc[-1]
    supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
    if not supports:
        return None

    sl = max(supports) * (1 - ZONE_BUFFER)
    risk = entry - sl
    if risk <= 0:
        return None

    tp = entry + risk
    return entry, sl, tp

# =====================================================
# BACKTEST
# =====================================================
def backtest_symbol(okx, symbol):
    df = pd.DataFrame(
        okx.fetch_ohlcv(symbol, ENTRY_TF, limit=BACKTEST_LIMIT),
        columns=["t","open","high","low","close","volume"]
    )

    trades = []

    for i in range(120, len(df) - MAX_FORWARD):
        slice = df.iloc[:i+1]
        stl, trend = supertrend(slice, ATR_PERIOD, MULTIPLIER)
        vo = volume_oscillator(slice.volume, VO_FAST, VO_SLOW)

        if not valid_entry(slice, stl, trend, vo):
            continue

        candle = detect_candle(slice)
        if candle not in VALID_CANDLES:
            continue

        df1d = pd.DataFrame(
            okx.fetch_ohlcv(symbol, SR_TF, limit=LIMIT_1D),
            columns=["t","open","high","low","close","volume"]
        )

        trade = build_trade_fixed_r(slice, df1d)
        if not trade:
            continue

        entry, sl, tp = trade
        rr = None

        for j in range(i+2, min(i+MAX_FORWARD, len(df))):
            if df.high.iloc[j] >= tp:
                rr = 1; break
            if df.low.iloc[j] <= sl:
                rr = -1; break

        if rr is not None:
            trades.append({"Symbol": symbol, "RR": rr, "Win": rr > 0})

    return pd.DataFrame(trades)

# =====================================================
# EQUITY CURVE
# =====================================================
def plot_equity_curve(bt):
    bt = bt.copy()
    bt["Trade #"] = range(1, len(bt) + 1)
    bt["Equity (R)"] = bt["RR"].cumsum()

    fig = go.Figure(go.Scatter(
        x=bt["Trade #"], y=bt["Equity (R)"],
        mode="lines+markers", line=dict(width=3)
    ))

    fig.update_layout(
        title="📈 Equity Curve (Fixed-R System)",
        xaxis_title="Trade #",
        yaxis_title="Cumulative R",
        template="plotly_dark",
        height=500
    )
    return fig

# =====================================================
# UI
# =====================================================
st.set_page_config("OKX Fixed-R System", layout="wide")
st.title("🚀 OKX Spot Screener & Backtest — Fixed-R System")
st.success("✅ Run screener di 23:00 WIB (WAJIB) + optional 19:00 WIB")

tab1, tab2 = st.tabs(["🔍 Screener", "🧪 Backtest"])
okx = ccxt.okx({"enableRateLimit": True, "timeout": 30000})

# =====================================================
# TAB 1 — SCREENER
# =====================================================
with tab1:
    if st.button("🔍 Run Screener"):
        symbols = get_liquid_symbols(MIN_USDT_VOLUME)
        results, charts = [], {}

        with st.spinner("Scanning market..."):
            for s in symbols:
                try:
                    df4h = pd.DataFrame(
                        okx.fetch_ohlcv(s, ENTRY_TF, limit=LIMIT_4H),
                        columns=["t","open","high","low","close","volume"]
                    )
                    stl, trend = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
                    vo = volume_oscillator(df4h.volume, VO_FAST, VO_SLOW)

                    if not valid_entry(df4h, stl, trend, vo):
                        continue

                    candle = detect_candle(df4h)
                    if candle not in VALID_CANDLES:
                        continue

                    df1d = pd.DataFrame(
                        okx.fetch_ohlcv(s, SR_TF, limit=LIMIT_1D),
                        columns=["t","open","high","low","close","volume"]
                    )

                    trade = build_trade_fixed_r(df4h, df1d)
                    if not trade:
                        continue

                    entry, sl, tp = trade
                    risk = round((entry - sl) / entry * 100, 2)

                    results.append({
                        "Symbol": s,
                        "Candle": candle,
                        "Entry": fmt_price(entry),
                        "SL": fmt_price(sl),
                        "TP (1R)": fmt_price(tp),
                        "Risk %": risk,
                        "VO %": round(vo.iloc[-1], 2)
                    })

                    send_telegram(format_alert(
                        s, candle, entry, sl, tp, risk, round(vo.iloc[-1], believe=2)
                    ))

                except:
                    pass
                time.sleep(RATE_LIMIT_DELAY)

        if results:
            st.session_state.results = pd.DataFrame(results)
            st.success(f"Found {len(results)} setups")
            st.dataframe(st.session_state.results, use_container_width=True)
        else:
            st.warning("No valid setup found")

# =====================================================
# TAB 2 — BACKTEST
# =====================================================
with tab2:
    if st.button("🧪 Run Backtest"):
        symbols = get_liquid_symbols(MIN_USDT_VOLUME)
        all_bt = []

        with st.spinner("Running backtest..."):
            for s in symbols:
                try:
                    bt = backtest_symbol(okx, s)
                    if not bt.empty:
                        all_bt.append(bt)
                except:
                    pass

        if all_bt:
            bt = pd.concat(all_bt, ignore_index=True)
            st.metric("Total Trades", len(bt))
            st.metric("Winrate %", round(bt["Win"].mean()*100, 2))
            st.metric("Avg RR", round(bt["RR"].mean(), 2))
            st.plotly_chart(plot_equity_curve(bt), use_container_width=True)
        else:
            st.warning("No trades found")
