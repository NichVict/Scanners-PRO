# =========================================================
# 🧠 SCANNER DE OPÇÕES — FÊNIX (VERSÃO ASSINANTE / APIMEC)
# =========================================================
from __future__ import annotations
import streamlit as st
st.set_page_config(
    page_title="Scanner de Opções",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# Imports
# =========================================================
import os, math, calendar
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests, yfinance as yf
import plotly.graph_objects as go

from scipy.stats import norm
from scipy.optimize import brentq

# =========================================================
# ENV helpers
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
    return {"Access-Token": OPLAB_API_KEY, "Accept": "application/json"}

def _to_num(x): 
    return pd.to_numeric(x, errors="coerce")

# =========================================================
# Sidebar — parâmetros (IGUAL AO ORIGINAL)
# =========================================================
with st.sidebar:
    st.title("⚙️ Parâmetros do Scanner")

    symbols = st.multiselect(
        "Ativos (subjacentes)",
        ["BOVA11","PETR4","VALE3","ITUB4","WEGE3","ABEV3","BBDC4","BBAS3","EMBR3"],
        default=["BOVA11"]
    )

    days = st.number_input("Dias de histórico (candles)", 30, 365, 180, 5)
    taxa_juros = st.number_input("Taxa de juros anual (%)", 0.0, 50.0, 14.90, 0.1) / 100

    tipo_opcao = st.radio("Tipo de opção", ["Ambas","CALL","PUT"], horizontal=True)

    def proximo_vencimento():
        hoje = date.today()
        c = calendar.Calendar()
        sextas = [d for d in c.itermonthdates(hoje.year, hoje.month) if d.weekday()==4 and d.month==hoje.month]
        if len(sextas)>=3 and hoje<=sextas[2]:
            return sextas[2]
        mes = hoje.month+1 if hoje.month<12 else 1
        ano = hoje.year if hoje.month<12 else hoje.year+1
        sextas = [d for d in c.itermonthdates(ano, mes) if d.weekday()==4 and d.month==mes]
        return sextas[2]

    col1, col2 = st.columns(2)
    with col1:
        venc_ini = st.date_input("Venc. inicial", proximo_vencimento())
    with col2:
        venc_fim = st.date_input("Venc. final", venc_ini)

    delta_min = st.slider("Delta mínimo (abs)", 0.0, 1.0, 0.30, 0.01)
    delta_max = st.slider("Delta máximo (abs)", 0.0, 1.0, 0.60, 0.01)
    iv_pct_max = st.slider("IV percentil local máx.", 0, 100, 60)
    min_vol = st.number_input("Volume mínimo (opção)", 0, 200000, 0)
    max_spread = st.slider("Spread relativo máx.", 0.0, 2.0, 0.05, 0.01)

    delta_target = st.slider("Delta alvo p/ score", 0.0, 1.0, 0.45, 0.01)
    top_n = st.number_input("Top por vencimento", 1, 10, 5)

    run = st.button("🌀 Rodar Scanner", type="primary", width="stretch")

# =========================================================
# Fetch data
# =========================================================
@st.cache_data(ttl=600)
def fetch_candles(symbol, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(f"{symbol}.SA", start=start, end=end, progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    df["vol_fin"] = df["Close"] * df["Volume"]
    df["mm20"] = df["vol_fin"].rolling(20).mean()
    return df

@st.cache_data(ttl=300, show_spinner=True)
def fetch_options(symbol: str) -> pd.DataFrame:
    url = f"{OPLAB_BASE_URL}/market/options/{symbol}"
    try:
        r = requests.get(url, headers=_headers(), timeout=30)

        # ❌ NÃO explode a app
        if r.status_code != 200:
            st.warning(f"⚠️ Oplab retornou {r.status_code} para {symbol}")
            return pd.DataFrame()

        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            st.warning(f"⚠️ Snapshot vazio para {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Normalizações mínimas
        df["expiration"] = pd.to_datetime(df.get("due_date"), errors="coerce")
        df["strike"] = pd.to_numeric(df.get("strike_price"), errors="coerce")
        df["type"] = df.get("category", "").astype(str).str.upper()
        df["last"] = pd.to_numeric(df.get("last_price"), errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
        df["ref_price"] = pd.to_numeric(df.get("spot_price"), errors="coerce")
        df["symbol"] = df.get("option_symbol")

        return df.dropna(subset=["symbol"]).reset_index(drop=True)

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erro de rede ao buscar opções de {symbol}: {e}")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Erro inesperado ({symbol}): {e}")
        return pd.DataFrame()


# =========================================================
# Main
# =========================================================
st.title("🧠 Scanner de Opções")

if run:
    if not symbols:
        st.warning("Selecione ao menos um ativo.")
        st.stop()

    dfs_candles, dfs_opts = [], []
    for s in symbols:
        dfs_candles.append(fetch_candles(s, days))
        dfs_opts.append(fetch_options(s))

    candles = pd.concat(dfs_candles)
    opts = pd.concat(dfs_opts)

    opts = opts[
        (opts["expiration"].dt.date.between(venc_ini, venc_fim)) &
        (opts["volume"] >= min_vol)
    ]

    if tipo_opcao != "Ambas":
        opts = opts[opts["type"] == tipo_opcao]

    opts["score"] = (
        0.4 * (1 - opts["volume"].rank(pct=True)) +
        0.3 * (1 - (opts["strike"] / opts["ref_price"] - 1).abs()) +
        0.2 * (1 - (opts["type"]=="PUT").astype(int)) +
        0.1 * (1 - opts["volume"].rank(pct=True))
    )

    top = (
        opts.sort_values("score", ascending=False)
            .groupby(opts["expiration"].dt.date)
            .head(top_n)
    )

    st.subheader("🏆 Top Oportunidades por Vencimento 💎")
    st.dataframe(
        top[["symbol","type","strike","expiration","score","volume"]],
        hide_index=True,
        width="stretch"
    )

    st.markdown("---")
    st.subheader("📈 Candles — OHLCV")

    for s in symbols:
        df = candles[candles["Ticker"]==s] if "Ticker" in candles else candles
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name=s
        ))
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, width="stretch")
