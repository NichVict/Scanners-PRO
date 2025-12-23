from __future__ import annotations
import os, math
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf

import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq



# =========================================================
# 🔐 ENV — OPLAB
# =========================================================
def getenv(key: str) -> str:
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        return str(st.secrets.get(key, "")).strip()
    except:
        return ""

OPLAB_API_KEY  = getenv("OPLAB_API_KEY")
OPLAB_BASE_URL = getenv("OPLAB_BASE_URL") or "https://api.oplab.com.br/v3"

def _headers():
    return {
        "Access-Token": OPLAB_API_KEY,
        "Accept": "application/json"
    }

# =========================================================
# 📊 DATA FETCH
# =========================================================
@st.cache_data(ttl=300)
def fetch_options(symbol: str) -> pd.DataFrame:
    url = f"{OPLAB_BASE_URL}/market/options/{symbol}"
    r = requests.get(url, headers=_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data if isinstance(data, list) else data.get("data", []))
    if df.empty:
        return df

    df.rename(columns={
        "option_symbol": "symbol",
        "strike_price": "strike",
        "due_date": "expiration",
        "last_price": "last",
        "spot_price": "ref_price"
    }, inplace=True)

    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["last"] = pd.to_numeric(df["last"], errors="coerce")
    df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce")
    df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df["type"] = df.get("type", df.get("category")).astype(str).str.upper()

    return df.dropna(subset=["symbol"])

@st.cache_data(ttl=600)
def fetch_candles(symbol: str, days: int = 180) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(f"{symbol}.SA", start=start, end=end, progress=False)
    if df.empty:
        return df
    df.reset_index(inplace=True)
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    return df

# =========================================================
# 🧮 BLACK & SCHOLES
# =========================================================
def bs_price(S, K, T, r, sigma, call=True):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if call:
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def implied_vol(S, K, T, r, premium, call=True):
    try:
        return brentq(lambda x: bs_price(S, K, T, r, x, call) - premium, 0.001, 5)
    except:
        return np.nan

# =========================================================
# 🧠 SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Parâmetros")

    symbols = st.multiselect(
        "Ativos",
        ["BOVA11","PETR4","VALE3","ITUB4","WEGE3"],
        default=["BOVA11"]
    )

    tipo = st.radio("Tipo", ["CALL", "PUT", "Ambas"], horizontal=True)

    delta_min, delta_max = st.slider("Delta", 0.0, 1.0, (0.3, 0.6), 0.01)
    iv_max = st.slider("IV % Máx", 0, 100, 60)
    spread_max = st.slider("Spread Máx", 0.0, 2.0, 0.05)

    juros = st.number_input("Juros (%)", 0.0, 50.0, 14.9) / 100
    top_n = st.number_input("Top N", 1, 10, 5)

    run = st.button("🌀 Rodar Scanner", type="primary", use_container_width=True)

# =========================================================
# 🚀 EXECUÇÃO
# =========================================================
if run and symbols:

    all_opts = []
    all_candles = []

    for s in symbols:
        all_opts.append(fetch_options(s))
        all_candles.append(fetch_candles(s))

    opts = pd.concat(all_opts, ignore_index=True)
    candles = pd.concat(all_candles, ignore_index=True)

    opts["T"] = (opts["expiration"] - pd.Timestamp.today()).dt.days.clip(lower=1) / 252
    opts["mid"] = (opts["bid"] + opts["ask"]) / 2
    opts["premium"] = opts["mid"].fillna(opts["last"])

    opts["iv"] = opts.apply(
        lambda r: implied_vol(
            r["ref_price"], r["strike"], r["T"], juros, r["premium"],
            call=r["type"] == "CALL"
        ), axis=1
    )

    opts["score"] = (
        0.4 * (1 - opts["iv"].rank(pct=True)) +
        0.3 * opts["volume"].rank(pct=True) +
        0.2 * (1 - abs(opts["strike"] - opts["ref_price"]) / opts["ref_price"]) +
        0.1 * (1 - (opts["ask"] - opts["bid"]) / opts["last"])
    )

    opts = opts.sort_values("score", ascending=False).head(top_n)

    # =========================
    # 🏆 CARDS
    # =========================
    st.subheader("🏆 Top Oportunidades")

    cols = st.columns(len(opts))
    for col, (_, r) in zip(cols, opts.iterrows()):
        color = "#1f7a1f" if r["type"] == "CALL" else "#7a1f1f"
        col.markdown(
            f"""
            <div style="
                background:{color};
                padding:16px;
                border-radius:16px;
                text-align:center;
                color:white">
                <b>{r['symbol']}</b><br>
                {r['type']}<br>
                Score {r['score']:.2f}<br>
                Strike {r['strike']}<br>
                Venc {r['expiration'].date()}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # =========================
    # 📋 TABELA
    # =========================
    st.dataframe(
        opts[["symbol","type","strike","expiration","score","iv","volume"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # =========================
    # 📈 CANDLES
    # =========================
    for s in symbols:
        df = candles[candles["Symbol"] == s] if "Symbol" in candles else candles
        if df.empty:
            continue

        fig = go.Figure(go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        ))
        fig.update_layout(template="plotly_dark", title=s, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📦 Dados brutos"):
        st.dataframe(opts, use_container_width=True)
