# -*- coding: utf-8 -*-
"""
Buscador de Opções — Oplab v3 + Yahoo (fallback) + IV/Greeks locais
Versão 2025-10-28

Recursos:
- Interface Streamlit (sidebar completa, filtros, um único botão "Rodar buscador")
- Baixa OHLCV do ativo (Oplab -> fallback Yahoo)
- Snapshot de opções (Oplab /market/options/{symbol})
- IV e gregas locais (Black–Scholes + Brent)
- Filtros: CALL/PUT, janela de vencimento, delta, IV %, volume, spread, e
  "exigir volume do ativo acima da MM20" (volume financeiro > MM20 do dia mais recente)
- Gráfico: candles + barras de volume financeiro + MM20 branca

Requer .env com:
  OPLAB_API_KEY="seu_token"
  (opcional) OPLAB_BASE_URL="https://api.oplab.com.br/v3/"
"""

from __future__ import annotations
import os, math
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests, yfinance as yf
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq

# ===============================
# Config inicial
# ===============================
# ===============================
# Config inicial e layout responsivo
# ===============================
st.set_page_config(
    page_title="Scanner de Opções",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CSS para responsividade (apenas layout)
# ===============================
# ===============================
# CSS — Sidebar Slim Mode (fonte e espaçamento reduzidos)
# ===============================
# ===============================
# CSS — Reduzir tamanho da fonte da sidebar
# ===============================
st.markdown("""
<style>
/* 🔹 Sidebar: fonte geral menor e mais compacta */
section[data-testid="stSidebar"] {
    font-size: 0.85rem !important;      /* reduz o texto geral */
}

/* 🔹 Ajusta inputs, sliders, radios e checkboxes */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.82rem !important;
}

/* 🔹 Sliders e valores numéricos */
section[data-testid="stSidebar"] .stSlider {
    font-size: 0.8rem !important;
}

/* 🔹 Botões */
section[data-testid="stSidebar"] button {
    font-size: 0.85rem !important;
    padding: 0.4rem 0.6rem !important;
}

/* 🔹 Título da sidebar (ex: "⚙️ Parâmetros do Scanner") */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)





load_dotenv(find_dotenv(), override=True)
OPLAB_API_KEY  = os.getenv("OPLAB_API_KEY", "")
OPLAB_BASE_URL = os.getenv("OPLAB_BASE_URL", "https://api.oplab.com.br/v3/").rstrip("/")

def _headers():
    return {"Access-Token": OPLAB_API_KEY, "accept": "application/json"}

def _to_num(x): 
    return pd.to_numeric(x, errors="coerce")

def err(msg: str):
    st.error(f"❌ {msg}")

def warn(msg: str):
    st.warning(f"⚠️ {msg}")

# ===============================
# Yahoo helper
# ===============================
def _yf_download_one(ticker_sa: str, start: datetime, end: datetime) -> pd.DataFrame:
    data = yf.download(ticker_sa, start=start, end=end, progress=False, auto_adjust=False)
    if data is None or data.empty:
        raise ValueError("Yahoo sem dados")
    data = data.reset_index().rename(columns={
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close",
        "Adj Close":"adj_close","Volume":"volume"
    })
    for c in ["open","high","low","close","volume"]:
        data[c] = _to_num(data[c])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data[["date","open","high","low","close","volume"]]

# ===============================
# Fetch candles (Oplab -> Yahoo)
# ===============================
@st.cache_data(ttl=600, show_spinner=True)
def fetch_candles(symbol: str, days: int = 180) -> pd.DataFrame:
    symbol = str(symbol).strip().upper()
    end = datetime.today()
    start = end - timedelta(days=days)

    # Corrige ticker para o formato do Yahoo (".SA" apenas se for brasileiro)
    ticker_yf = f"{symbol}.SA" if not symbol.endswith(".SA") else symbol

    try:
        df = yf.download([ticker_yf], start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError("Yahoo sem dados")

        # Se vier MultiIndex (caso da lista), "aplana" o DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["underlying_symbol"] = symbol
        df = df[["underlying_symbol", "date", "open", "high", "low", "close", "volume"]]
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    except Exception as e:
        err(f"Yahoo falhou ({symbol}): {e}")
        cols = ["underlying_symbol", "date", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols)


# ===============================
# Fetch opções (Oplab)
# ===============================
@st.cache_data(ttl=300, show_spinner=True)
def fetch_options_snapshot(symbol: str) -> pd.DataFrame:
    url = f"{OPLAB_BASE_URL}/market/options/{symbol}"
    try:
        r = requests.get(url, headers=_headers(), timeout=45)
        r.raise_for_status()
        raw = r.json()
        data = raw if isinstance(raw, list) else raw.get("data", [])
        df = pd.DataFrame(data)
        if df.empty:
            raise RuntimeError("Snapshot vazio")

        rename = {
            "parent_symbol": "underlying_symbol",
            "underlying": "underlying_symbol",
            "due_date": "expiration",
            "expiration_date": "expiration",
            "strike_price": "strike",
            "last_price": "last",
            "spot_price": "ref_price",
            "option_symbol": "symbol",
        }
        for k, v in rename.items():
            if k in df.columns and v != k:
                df.rename(columns={k: v}, inplace=True)

        # Campos essenciais
        needed = ["symbol","underlying_symbol","expiration","type","category","strike",
                  "bid","ask","last","close","volume","open_interest",
                  "ref_price"]
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan

        # Normalizações
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
        for c in ["strike","bid","ask","last","close","volume","open_interest","ref_price"]:
            df[c] = _to_num(df[c])

        # Tipagem CALL/PUT (type pode vir em 'category')
        if "type" not in df or df["type"].isna().all():
            df["type"] = df["category"]
        df["type"] = df["type"].astype(str).str.upper().replace({"C":"CALL","P":"PUT"})

        df["underlying_symbol"] = df["underlying_symbol"].astype(str).str.upper().fillna(symbol)

        return df.dropna(subset=["symbol"]).reset_index(drop=True)

    except Exception as e:
        warn(f"Falha ao buscar opções de {symbol}: {e}")
        cols = ["symbol","underlying_symbol","expiration","type","strike","bid","ask","last","close","volume","open_interest","ref_price"]
        return pd.DataFrame(columns=cols)

# ===============================
# Black-Scholes + IV local
# ===============================
def _bs_price_greeks(S: float, K: float, T: float, r: float, sigma: float, call_put: str):
    """Retorna (price, delta, gamma, vega, theta, rho). r e sigma anuais (decimais)."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return (np.nan,)*6
    cp = 1 if str(call_put).upper() == "CALL" else -1
    try:
        sqrtT = math.sqrt(T)
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrtT)
        d2 = d1 - sigma*sqrtT

        price = cp*(S*norm.cdf(cp*d1) - K*math.exp(-r*T)*norm.cdf(cp*d2))
        delta = cp*norm.cdf(cp*d1)
        gamma = norm.pdf(d1) / (S*sigma*sqrtT)
        vega  = S*norm.pdf(d1)*sqrtT
        theta = (-(S*norm.pdf(d1)*sigma)/(2*sqrtT) - cp*r*K*math.exp(-r*T)*norm.cdf(cp*d2))
        rho   = cp*K*T*math.exp(-r*T)*norm.cdf(cp*d2)
        return price, delta, gamma, vega, theta, rho
    except Exception:
        return (np.nan,)*6

def _implied_vol(S, K, T, r, premium, call_put):
    """IV via Brent. Retorna sigma (decimal)."""
    if not all(pd.notna([S, K, T, r, premium])) or S <= 0 or K <= 0 or T <= 0 or premium <= 0:
        return np.nan
    try:
        return brentq(lambda s: _bs_price_greeks(S, K, T, r, s, call_put)[0] - premium,
                      1e-3, 5.0, maxiter=100, disp=False)
    except Exception:
        return np.nan

# ===============================
# Contexto de volume do ativo (volume financeiro + MM20)
# ===============================
def preparar_contexto_ativos(df_at: pd.DataFrame, ma: int = 20) -> pd.DataFrame:
    """
    Calcula, por ativo, o último registro com:
      - volume_fin: close * volume
      - volfin_ma: MM20 de volume financeiro
      - vol_acima_ma: 1 se volume_fin > volfin_ma (no último dia), senão 0
      - last_close
    """
    if df_at is None or df_at.empty:
        return pd.DataFrame(columns=["underlying_symbol","volume_fin","volfin_ma","vol_acima_ma","last_close"])

    d = df_at.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d.sort_values(["underlying_symbol","date"], inplace=True)

    d["volume"] = _to_num(d.get("volume"))
    d["close"]  = _to_num(d.get("close"))

    d["volume_fin"] = d["close"] * d["volume"]
    d["volfin_ma"] = (
        d.groupby("underlying_symbol", group_keys=False)["volume_fin"]
         .transform(lambda s: pd.Series(s).rolling(ma, min_periods=1).mean().values)
    )
    d["vol_acima_ma"] = (d["volume_fin"] > d["volfin_ma"]).astype(int)

    last = (
        d.groupby("underlying_symbol", as_index=False)
         .tail(1)[["underlying_symbol","volume_fin","volfin_ma","vol_acima_ma","close"]]
         .rename(columns={"close":"last_close"})
         .reset_index(drop=True)
    )
    return last

# ===============================
# Enriquecimento: mid/spread, DTE/tempo, IV/greeks, percentis
# ===============================
def add_features_and_iv(df_opts: pd.DataFrame, price_lookup: dict[str, float] | None, r_annual: float) -> pd.DataFrame:
    if df_opts is None or df_opts.empty:
        return df_opts

    d = df_opts.copy()

    # Mid / spread
    for c in ["bid","ask","last","close","strike","volume","open_interest","ref_price"]:
        d[c] = _to_num(d.get(c))

    d["mid"] = np.where(
        pd.notna(d["bid"]) & pd.notna(d["ask"]) & (d["bid"]>0) & (d["ask"]>0),
        (d["bid"] + d["ask"]) / 2.0,
        np.where(pd.notna(d["last"]) & (d["last"]>0), d["last"], d["close"])
    )
    d["spread"] = (d["ask"] - d["bid"]).clip(lower=0).fillna(0.0)
    d["spread_rel"] = (d["spread"] / d["mid"]).replace([np.inf, -np.inf], np.nan)

    # Datas e tempo (T)
    # Datas e tempo (T) — corrigido para evitar erro de conversão timedelta64[D]
    hoje = date.today()
    d["expiration"] = pd.to_datetime(d.get("expiration"), errors="coerce")

    # calcula dias até o vencimento de forma robusta
    d["dte_calendar"] = (d["expiration"].dt.date - hoje).apply(
        lambda x: x.days if pd.notna(x) else np.nan
    )

    d["dte_bus"] = d["dte_calendar"].clip(lower=1)  # úteis aproximados
    d["T"] = (d["dte_bus"] / 252.0).clip(lower=1 / 365.0)

    # Preço de referência do subjacente
    if "ref_price" not in d.columns:
        d["ref_price"] = np.nan
    if price_lookup:
        mask_na = d["ref_price"].isna()
        if mask_na.any():
            d.loc[mask_na, "ref_price"] = d.loc[mask_na, "underlying_symbol"].map(price_lookup)

    # Premium a usar
    d["premium_used"] = np.where(d["last"]>0, d["last"], np.where(d["mid"]>0, d["mid"], d["close"]))

    # Tipagem
    d["type"] = d["type"].astype(str).str.upper().replace({"C":"CALL","P":"PUT"})
    d["option_type"] = np.where(d["type"].isin(["CALL","PUT"]), d["type"], "CALL")

    # IV
    d["iv_local"] = d.apply(lambda r: _implied_vol(r["ref_price"], r["strike"], r["T"], r_annual, r["premium_used"], r["option_type"]), axis=1)
    d["iv_local_pct"] = d["iv_local"] * 100.0

    # Greeks
    greeks = d.apply(lambda r: pd.Series(_bs_price_greeks(r["ref_price"], r["strike"], r["T"], r_annual,
                                                         r["iv_local"] if pd.notna(r["iv_local"]) and r["iv_local"]>0 else np.nan,
                                                         r["option_type"]),
                                         index=["bs_price","delta","gamma","vega","theta","rho"]), axis=1)
    d = pd.concat([d, greeks], axis=1)
    d["delta_abs"] = d["delta"].abs()

    # Percentil local de IV por ativo+vencimento
    d["iv_pct_local"] = (
        d.groupby(["underlying_symbol","expiration"])["iv_local_pct"]
         .transform(lambda s: 100*s.rank(pct=True, method="average"))
    )

    # Limpeza
    d["spread_rel"] = d["spread_rel"].fillna(1.0).clip(0, 5)

    return d

# ===============================
# Filtros e ranking
# ===============================
def aplicar_filtros(
    d: pd.DataFrame,
    tipo_opcao: str,
    venc_ini: date,
    venc_fim: date,
    delta_min: float,
    delta_max: float,
    iv_pct_max: float,
    min_volume_opt: float,
    max_spread_rel: float,
    exigir_vol_acima: bool
) -> pd.DataFrame:
    if d is None or d.empty:
        return d

    x = d.copy()

    # Janela de vencimento
    x = x[x["expiration"].between(pd.to_datetime(venc_ini), pd.to_datetime(venc_fim))]

    # Tipo CALL/PUT
    if tipo_opcao in ("CALL","PUT"):
        x = x[x["type"] == tipo_opcao]

    # Garantias de colunas
    if "volume" not in x.columns:
        if "volume_x" in x.columns: x["volume"] = x["volume_x"]
        elif "volume_y" in x.columns: x["volume"] = x["volume_y"]
        else: x["volume"] = np.nan
    x["volume"] = _to_num(x["volume"])

    # Condições
    cond = (
        x["delta_abs"].between(delta_min, delta_max, inclusive="both")
        & (x["iv_pct_local"] <= iv_pct_max)
        & (x["volume"].fillna(0) >= min_volume_opt)
        & (x["spread_rel"].fillna(1.0) <= max_spread_rel)
        & x["T"].gt(0)
    )

    # Exigir volume do ativo acima da MM20 (volume financeiro)
    #if exigir_vol_acima and "vol_acima_ma" in x.columns:
        #cond &= (x["vol_acima_ma"] == 1)

    return x.loc[cond.fillna(False)].copy()

def _norm01(s: pd.Series, invert: bool = False) -> pd.Series:
    s = _to_num(s)
    if s.nunique(dropna=True) <= 1:
        n = pd.Series(0.5, index=s.index)
    else:
        n = (s - s.min()) / (s.max() - s.min() + 1e-12)
    n = n.fillna(0.5)
    return (1 - n) if invert else n

def rankear(d: pd.DataFrame, delta_target=0.45, exigir_vol_acima=False) -> pd.DataFrame:
    if d is None or d.empty:
        return d

    x = d.copy().reset_index(drop=True)

    # Normalizações auxiliares
    def _norm01(s: pd.Series, invert: bool = False) -> pd.Series:
        s = _to_num(s)
        if s.nunique(dropna=True) <= 1:
            n = pd.Series(0.5, index=s.index)
        else:
            n = (s - s.min()) / (s.max() - s.min() + 1e-12)
        n = n.fillna(0.5)
        return (1 - n) if invert else n

    # Score base (sem volume acima da MM20 ainda)
    x["score_base"] = (
        0.40 * _norm01(x["iv_pct_local"], invert=True) +     # preferir IV % local baixo
        0.30 * _norm01(x["volume"]) +                        # preferir volume alto
        0.20 * _norm01((x["delta_abs"] - delta_target).abs(), invert=True) +  # delta próximo
        0.10 * _norm01(x["spread_rel"], invert=True)          # preferir spread pequeno
    )

    # 💡 Bônus proporcional baseado no volume relativo ao MM20
    if exigir_vol_acima and {"volume_fin", "volfin_ma"}.issubset(x.columns):
        ratio = (x["volume_fin"] / x["volfin_ma"]).replace([np.inf, -np.inf], np.nan)
        ratio = ratio.clip(lower=0.5, upper=2.0)  # evita distorções
        bonus = (ratio - 1.0) * 0.25              # até ±25%
        x["score"] = np.clip(x["score_base"] * (1 + bonus), 0, None)
    else:
        x["score"] = x["score_base"]

    return x.sort_values("score", ascending=False)


def top_por_venc(d: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if d is None or d.empty:
        return d
    return (
        d.sort_values(["expiration","score"], ascending=[True, False])
         .groupby("expiration", as_index=False)
         .head(n)
    )

# ===============================
# UI — Sidebar
# ===============================
with st.sidebar:
    st.title("⚙️ Parâmetros do Scanner")

    symbols = st.multiselect(
        "Ativos (subjacentes)",
        ["PETR4","BOVA11","VALE3","ITUB4","WEGE3","ABEV3","BBDC4","BBAS3","EMBR3","MGLU3"],
        default=["BOVA11"]
    )
    days = st.number_input("Dias de histórico (candles)", min_value=30, max_value=365, value=180, step=5)

    st.markdown("---")
    taxa_juros = st.number_input("Taxa de juros anual (%)", min_value=0.0, max_value=50.0, value=14.90, step=0.10) / 100.0

    st.markdown("---")
    tipo_opcao = st.radio("Tipo de opção", options=["Ambas","CALL","PUT"], index=0, horizontal=True)

    col_v1, col_v2 = st.columns(2)
    # ===============================
    # 🗓️ Cálculo automático do vencimento mais próximo
    # ===============================

    import calendar
    
    def proximo_vencimento_opcoes(base: date | None = None) -> date:
        """Retorna a 3ª sexta-feira do mês atual (se ainda não passou) ou a do mês seguinte."""
        if base is None:
            base = datetime.today().date()
    
        ano, mes = base.year, base.month
        c = calendar.Calendar(firstweekday=calendar.MONDAY)
    
        # Lista todas as sextas-feiras do mês
        sextas = [d for d in c.itermonthdates(ano, mes) if d.weekday() == 4 and d.month == mes]
    
        # Se a 3ª sexta ainda não passou neste mês, usa ela
        if len(sextas) >= 3 and base <= sextas[2]:
            return sextas[2]
        else:
            # Caso já tenha passado, pega a 3ª sexta do mês seguinte
            mes = 1 if mes == 12 else mes + 1
            ano = ano + 1 if mes == 1 else ano
            sextas = [d for d in c.itermonthdates(ano, mes) if d.weekday() == 4 and d.month == mes]
            return sextas[2] if len(sextas) >= 3 else base + timedelta(days=30)
    
    # ===== Sidebar =====
    prox_venc = proximo_vencimento_opcoes()
    
    with col_v1:
        venc_ini = st.date_input("Venc. inicial", prox_venc)
    with col_v2:
        venc_fim = st.date_input("Venc. final", prox_venc)



    st.markdown("---")
    delta_min = st.slider("Delta mínimo (abs)", 0.0, 1.0, 0.30, 0.01)
    delta_max = st.slider("Delta máximo (abs)", 0.0, 1.0, 0.60, 0.01)
    iv_pct_max = st.slider("IV percentil local máx. (%)", 0, 100, 60, 1)
    min_vol_opt = st.number_input("Volume mínimo (opção)", 0, 200000, 0, 100)
    max_spread_rel = st.slider("Spread relativo máx.", 0.0, 5.0, 1.0, 0.05)
    exigir_vol_acima = st.checkbox("Exigir volume do ativo acima da MM20 (volume financeiro)", value=False)

    st.markdown("---")
    delta_target = st.slider("Delta alvo p/ score", 0.0, 1.0, 0.45, 0.01)
    top_n = st.number_input("Top por vencimento", 1, 10, 5)

    st.markdown("---")
    btn_run = st.button("🌀 Rodar Scanner", type="primary", use_container_width=True)

# ===============================
# UI — Main
# ===============================
st.title("🧠Scanner de Opções")

# ===============================
# Explicação — Score de Oportunidade (posicionado logo abaixo do título)
# ===============================
st.markdown("""
<style>
.expander-dark {
    background-color: #1A1D23 !important;
    border: 1px solid #2B2F36 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
}
.expander-dark div[role='button'] {
    color: #00C896 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
.expander-dark p, .expander-dark li {
    color: #E0E0E0 !important;
    font-size: 0.9rem !important;
    line-height: 1.5em;
}
</style>
""", unsafe_allow_html=True)

with st.expander("📘 Entendendo o Score de Oportunidade", expanded=False):
    st.markdown("""
O **Score de Oportunidade** classifica as opções conforme sua **atratividade relativa**, 
combinando **preço justo, liquidez e eficiência de volatilidade**.  
Ele é calculado com base em quatro fatores principais:

- 📉 **Volatilidade implícita (IV%) local:** prioriza opções com IV mais baixa dentro do vencimento (menor sobrepreço);  
- ⚡ **Volume de negociação:** valoriza contratos com maior liquidez;  
- 🎯 **Delta:** favorece deltas próximos do alvo definido (ex: 0,45);  
- 💸 **Spread relativo:** penaliza opções com diferença grande entre bid e ask.

O resultado é um **Score entre 0 e 1**, onde valores mais altos indicam melhor equilíbrio entre **risco, liquidez e eficiência**.
""")

    import plotly.graph_objects as go
    fatores = ["IV% (baixa)", "Volume", "Delta (alvo)", "Spread (baixo)"]
    pesos = [0.40, 0.30, 0.20, 0.10]

    fig_score = go.Figure(
        go.Bar(
            x=pesos,
            y=fatores,
            orientation='h',
            text=[f"{p*100:.0f}%" for p in pesos],
            textposition='outside',
            marker=dict(
                color=['#00E6A8', '#00C896', '#009F80', '#007B66'],
                line=dict(color='#0E1117', width=1)
            )
        )
    )
    fig_score.update_layout(
        template='plotly_dark',
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        xaxis=dict(title="Peso (%)", range=[0, 0.5]),
        yaxis=dict(title=""),
        showlegend=False,
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#FFFFFF', size=12)
    )

    st.plotly_chart(fig_score, use_container_width=True)


# Estado
if "ativos" not in st.session_state:  st.session_state["ativos"] = pd.DataFrame()
if "opcoes" not in st.session_state:  st.session_state["opcoes"] = pd.DataFrame()

# ===============================
# Estado e execução automática
# ===============================
if "primeira_execucao" not in st.session_state:
    st.session_state["primeira_execucao"] = True
else:
    st.session_state["primeira_execucao"] = False


# Executa automaticamente na primeira abertura da página
if btn_run or st.session_state["primeira_execucao"]:

    if not symbols:
        err("Selecione ao menos um ativo.")
        st.stop()

    with st.status("Baixando e preparando dados...", expanded=True) as status:
        try:
            # 1) Download
            dfs_at, dfs_op = [], []
            for sym in symbols:
                with st.spinner(f"Baixando dados de {sym}..."):
                    dfs_at.append(fetch_candles(sym, int(days)))
                    dfs_op.append(fetch_options_snapshot(sym))


            at = pd.concat(dfs_at, ignore_index=True) if dfs_at else pd.DataFrame()
            op = pd.concat(dfs_op, ignore_index=True) if dfs_op else pd.DataFrame()

            if at.empty or op.empty:
                err("Sem dados suficientes (ativos ou opções).")
                st.stop()

            # 2) Contexto volume (financeiro) e merge
            ctx = preparar_contexto_ativos(at, ma=20)
            last_close_map = dict(zip(ctx["underlying_symbol"], ctx["last_close"]))

            # injeta flag no book
            book_raw = op.merge(
                ctx.rename(columns={"volume_fin":"volume_fin_acao","volfin_ma":"volfin_ma_acao"}),
                on="underlying_symbol", how="left"
            )

            # 3) Enriquecer com IV/greeks etc.
            book = add_features_and_iv(book_raw, price_lookup=last_close_map, r_annual=taxa_juros)

            # 4) Aplicar filtros
            flt = aplicar_filtros(
                book,
                tipo_opcao=tipo_opcao,
                venc_ini=venc_ini, venc_fim=venc_fim,
                delta_min=delta_min, delta_max=delta_max,
                iv_pct_max=float(iv_pct_max),
                min_volume_opt=float(min_vol_opt),
                max_spread_rel=float(max_spread_rel),
                exigir_vol_acima=bool(exigir_vol_acima)
            )

            # 5) Ranking e top por vencimento
            ranked = rankear(flt, delta_target=delta_target, exigir_vol_acima=exigir_vol_acima)
            top = top_por_venc(ranked, n=int(top_n))

            status.update(label="Concluído", state="complete")

            # ====== Saída principal ======
            # ====== Saída principal ======
            # ====== Saída principal ======
            # ====== Saída principal ======


          
        
            st.subheader("🏆 Top Oportunidades por Vencimento 💎")

            


            if top.empty:
                warn("Nenhuma oportunidade encontrada com os critérios atuais. Afrouxe IV %, delta ou spread.")
            else:
                # 🔢 Ordena por score decrescente
                top = top.sort_values("score", ascending=False).reset_index(drop=True)

                # 🧾 Ajustes visuais
                top["expiration"] = pd.to_datetime(top["expiration"], errors="coerce").dt.date  # só data
                num_cols = top.select_dtypes(include=["float", "float64", "int", "int64"]).columns
                top[num_cols] = top[num_cols].apply(lambda x: np.round(x, 2))  # arredonda 2 casas

                # Coloca 'score' na 1ª coluna
                cols = ["score"] + [c for c in top.columns if c != "score"]

                # 💎 Destaques — Top 5 Cards com gradiente dinâmico
               
                #st.markdown("### 💎 Destaques (Top 5 Scores Globais)")#

                # 🔹 Garante que o número de cards nunca passe de 10, mesmo que o usuário altere o código
                num_cards = min(int(top_n), 10)
                top5 = top.head(num_cards)[["symbol", "score", "type", "strike", "delta", "expiration"]]


                def get_card_gradient(score, tipo):
                    """Gradiente dinâmico de cor baseado no score e tipo (CALL/PUT)."""
                    s = float(score)
                    if tipo == "CALL":
                        start, end = "#004d00", "#66ff66"  # verde escuro → verde claro
                    else:
                        start, end = "#7f0000", "#ff6666"  # vermelho escuro → vermelho claro
                    return f"linear-gradient(135deg, {start} {(s*100):.0f}%, {end})"

                # 🟢 inicializa a variável antes do loop
                card_html = ""

                for _, row in top5.iterrows():
                    grad = get_card_gradient(row["score"], row["type"])
                    delta_color = "lime" if row["type"] == "CALL" else "salmon"

                    card_html += f"""
                    <div class="card" style="background-image: {grad};">
                        <div class="symbol">{row['symbol']} ({row['type']})</div>
                        <div class="score-label">Score</div>
                        <div class="score">{row['score']:.2f}</div>
                        <div class="details">Strike {row['strike']:.2f} • Venc. {row['expiration']}</div>
                        <div class="delta-line">
                            <span style='color:{delta_color}; font-weight:600;'>Δ {row['delta']:.2f}</span>
                        </div>
                    </div>
                    """





                st.markdown(
                    f"""
                    <style>
                    .card {{
                        display: inline-block;
                        border-radius: 16px;
                        padding: 16px 18px;
                        margin: 8px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                        color: white;
                        transition: all 0.25s ease;
                        width: 16.5%; /* 🔹 menor largura — garante 5 por linha */
                        text-align: center;
                        min-height: 180px;
                    }}
                    .card:hover {{
                        transform: translateY(-4px) scale(1.03);
                        box-shadow: 0 6px 14px rgba(0,0,0,0.6);
                        cursor: pointer;
                    }}
                    .symbol {{
                        font-weight: 600;
                        font-size: 1rem;
                        margin-bottom: 6px;
                    }}
                    .score-label {{
                        font-size: 0.8rem;
                        color: rgba(255,255,255,0.9);
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }}
                    .score {{
                        font-size: 1.8rem;
                        font-weight: 700;
                        margin-bottom: 6px;
                        color: #fff;
                        text-shadow: 0 0 10px rgba(255,255,255,0.5);
                    }}
                    .details {{
                        font-size: 0.85rem;
                        color: rgba(255,255,255,0.85);
                    }}
                    .delta-line {{
                        margin-top: 4px;
                    }}
                
                    /* 🔹 Responsividade — ajustado */
                    @media (max-width: 1800px) {{
                        .card {{ width: 17%; }}
                    }}
                    @media (max-width: 1300px) {{
                        .card {{ width: 22%; }} /* 4 por linha */
                    }}
                    @media (max-width: 1000px) {{
                        .card {{ width: 45%; }} /* 2 por linha */
                    }}
                    @media (max-width: 600px) {{
                        .card {{
                            width: 90%;
                            padding: 14px 16px;
                        }}
                        .symbol {{ font-size: 0.95rem; }}
                        .score-label {{ font-size: 0.7rem; }}
                        .score {{ font-size: 1.5rem; }}
                        .details {{ font-size: 0.8rem; }}
                    }}
                    @media (max-width: 400px) {{
                        .card {{
                            width: 95%;
                            padding: 12px 14px;
                        }}
                        .symbol {{ font-size: 0.9rem; }}
                        .score {{ font-size: 1.3rem; }}
                        .details {{ font-size: 0.75rem; }}
                    }}
                
                    /* 🔹 Container flexível centralizado e alinhamento perfeito */
                    .cards-container {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: center;
                        align-items: stretch;
                        gap: 10px;
                    }}
                    </style>
                
                    <div class="cards-container">
                        {card_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )




                st.markdown("---")

                # 🧠 Coloração condicional da tabela com gradiente idêntico
                # 🧠 Coloração condicional da tabela com gradiente idêntico
                def score_color(val, tipo):
                    if pd.isna(val):
                        return ""
                    s = float(val)
                    s = max(0, min(s, 1))
                    if tipo == "CALL":
                        dark = np.array([0, 77, 0])      # #004d00
                        light = np.array([102, 255, 102])  # #66ff66
                    else:
                        dark = np.array([127, 0, 0])     # #7f0000
                        light = np.array([255, 102, 102])  # #ff6666
                    rgb = (dark * s + light * (1 - s)).astype(int)
                    color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
                    return f"background-color: {color}; color: black; font-weight: 700;"

                # Determina colunas financeiras
                financial_cols = [
                    c for c in top.columns if any(k in c.lower() for k in ["price", "premium", "strike", "ref_", "volfin", "volume"])
                ]

                # Formata todas as colunas numéricas para 2 casas e adiciona R$ em valores financeiros
                fmt = {}
                for c in top.columns:
                    if c in financial_cols:
                        fmt[c] = "R$ {:,.2f}".format
                    elif top[c].dtype.kind in "fi":
                        fmt[c] = "{:.2f}".format

                styled_df = (
                    top[cols]
                    .style
                    .format(fmt)
                    .apply(
                        lambda r: [score_color(r["score"], r["type"])] + ["" for _ in range(len(r) - 1)],
                        axis=1
                    )
                )

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True
                )





            # ====== Gráficos de candles ======
            # ====== Gráficos de candles ======
            st.markdown("---")
            st.subheader("📈 Candles (últimos dias) — OHLCV")
            st.caption("OHLCV = Open, High, Low, Close, Volume. Volume abaixo é financeiro (Close × Volume) com MM20 branca.")

            if at.empty:
                warn("Candles indisponíveis.")
            else:
                for sym in sorted(set(at["underlying_symbol"])):
                    d = at[at["underlying_symbol"] == sym].sort_values("date").tail(180)
                    if d.empty:
                        continue

                    # Volume financeiro + MM20
                    d["vol_fin"] = _to_num(d["close"]) * _to_num(d["volume"])
                    d["volfin_ma20"] = d["vol_fin"].rolling(20, min_periods=1).mean()

                    fig = go.Figure()

                    # 📊 Candles (eixo Y principal)
                    fig.add_trace(go.Candlestick(
                        x=d["date"],
                        open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                        name=f"{sym} OHLC",
                        increasing_line_color="lime",
                        decreasing_line_color="red",
                        yaxis="y1"
                    ))

                    # 💙 Volume financeiro — barras azuis sólidas
                    fig.add_trace(go.Bar(
                        x=d["date"],
                        y=d["vol_fin"],
                        name="Volume financeiro",
                        marker_color="deepskyblue",
                        yaxis="y2",
                        opacity=0.6
                    ))

                    # 💬 (opcional) Volume verde/vermelho conforme variação:
                    # colors = np.where(d["close"] >= d["open"], "limegreen", "crimson")
                    # fig.add_trace(go.Bar(
                    #     x=d["date"], y=d["vol_fin"],
                    #     name="Volume financeiro",
                    #     marker_color=colors,
                    #     yaxis="y2", opacity=0.6
                    # ))

                    # Linha da MM20 branca
                    fig.add_trace(go.Scatter(
                        x=d["date"],
                        y=d["volfin_ma20"],
                        name="MM20 Vol (R$)",
                        mode="lines",
                        line=dict(color="white", width=1.5),
                        yaxis="y2"
                    ))

                    fig.update_layout(
                        title=f"{sym}",
                        height=550,
                        template="plotly_dark",
                        xaxis=dict(
                            domain=[0.0, 1.0],
                            rangeslider=dict(visible=False),
                            showline=True,
                            linecolor="#555",
                            mirror=True
                        ),
                        # Preço (candles) — painel superior
                        yaxis=dict(
                            title="Preço",
                            domain=[0.35, 1.0],   # 65% do topo
                            side="left",
                            showgrid=True
                        ),
                        # Volume — painel inferior (separado dos candles)
                        yaxis2=dict(
                            title="Volume (R$)",
                            domain=[0.0, 0.30],   # 30% da parte inferior
                            showgrid=False
                            # (sem overlaying!)
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=40, r=40, t=50, b=20)
                    )


                    st.plotly_chart(fig, use_container_width=True)


            # ====== Dados brutos (opcional para debug) ======
            with st.expander("📦 Dados brutos (opcional)"):
                st.caption("Ativos (OHLCV)")
                st.dataframe(at, use_container_width=True, hide_index=True)

                st.caption("Opções (processadas com IV/greeks)")
                show_cols = [
                    "symbol","underlying_symbol","type","expiration","strike",
                    "bid","ask","last","close","premium_used",
                    "ref_price","T","dte_bus",
                    "iv_local_pct","iv_pct_local",
                    "delta","gamma","vega","theta","rho",
                    "volume","open_interest","spread","spread_rel","moneyness",
                    "vol_acima_ma","score"
                ]
                show_cols = [c for c in show_cols if c in book.columns]
                st.dataframe(book[show_cols], use_container_width=True, hide_index=True)

        except Exception as e:
            status.update(label="Erro no processamento", state="error")
            err(str(e))
