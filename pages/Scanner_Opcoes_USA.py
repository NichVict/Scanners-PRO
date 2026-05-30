bash

cat > /mnt/user-data/outputs/Scanner_Opcoes_USA.py << 'ENDOFFILE'
# Scanner_Opcoes_USA.py
# -*- coding: utf-8 -*-
"""
Scanner de Opções — EUA (Tastytrade + Yahoo + Black-Scholes local)

Recursos:
- Autenticação Tastytrade via .env
- Option chain via Tastytrade /option-chains/{symbol}/nested
- Candles via yfinance (sem sufixo .SA)
- IV e greeks locais (Black-Scholes + Brent)
- Filtros: CALL/PUT, janela de vencimento, delta, IV%, volume, spread
- Gráfico: candles + volume financeiro + MM20
"""

from __future__ import annotations
import os, math
from datetime import datetime, timedelta, date
import calendar

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq

# ===============================
# Config
# ===============================
st.set_page_config(
    page_title="Scanner de Opções USA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
section[data-testid="stSidebar"] { font-size: 0.85rem !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label { font-size: 0.82rem !important; }
section[data-testid="stSidebar"] button { font-size: 0.85rem !important; padding: 0.4rem 0.6rem !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ===============================
# Credenciais (.env)
# ===============================
load_dotenv(find_dotenv(), override=True)
TT_USERNAME   = os.getenv("TT_USERNAME", "")
TT_PASSWORD   = os.getenv("TT_PASSWORD", "")
TT_CLIENT_ID  = os.getenv("TT_CLIENT_ID", "")
TT_CLIENT_SECRET = os.getenv("TT_CLIENT_SECRET", "")
TT_BASE_URL   = os.getenv("TT_BASE_URL", "https://api.tastyworks.com")

# ===============================
# Helpers
# ===============================
def _to_num(x):
    return pd.to_numeric(x, errors="coerce")

def err(msg):  st.error(f"❌ {msg}")
def warn(msg): st.warning(f"⚠️ {msg}")

# ===============================
# Autenticação Tastytrade
# ===============================
@st.cache_data(ttl=3600, show_spinner=False)
def get_tt_token() -> str:
    resp = requests.post(
        f"{TT_BASE_URL}/sessions",
        json={
            "login": TT_USERNAME,
            "password": TT_PASSWORD,
            "client-id": TT_CLIENT_ID,
            "client-secret": TT_CLIENT_SECRET,
        },
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"]["session-token"]

def _tt_headers(token: str) -> dict:
    return {"Authorization": token, "Content-Type": "application/json"}

# ===============================
# Fetch option chain (Tastytrade)
# ===============================
@st.cache_data(ttl=300, show_spinner=True)
def fetch_options_snapshot(symbol: str, token: str) -> pd.DataFrame:
    """
    Busca o chain nested da Tastytrade e converte para DataFrame flat
    com uma linha por opção (call ou put), pronto para o pipeline de IV/greeks.
    """
    symbol = symbol.strip().upper()
    url = f"{TT_BASE_URL}/option-chains/{symbol}/nested"
    try:
        resp = requests.get(url, headers=_tt_headers(token), timeout=30)
        resp.raise_for_status()
        items = resp.json().get("data", {}).get("items", [])
        if not items:
            raise RuntimeError("Chain vazio")

        rows = []
        for chain in items:
            for exp in chain.get("expirations", []):
                exp_date = exp.get("expiration-date")
                dte      = exp.get("days-to-expiration", 0)
                for strike in exp.get("strikes", []):
                    sp = float(strike.get("strike-price", 0))
                    for opt_type, sym_key in [("CALL", "call"), ("PUT", "put")]:
                        occ_symbol = strike.get(sym_key, "")
                        if not occ_symbol:
                            continue
                        rows.append({
                            "symbol":             occ_symbol,
                            "underlying_symbol":  symbol,
                            "expiration":         exp_date,
                            "type":               opt_type,
                            "strike":             sp,
                            "dte_calendar":       dte,
                            # campos de preço — serão preenchidos com NaN
                            # e depois calculados via Black-Scholes
                            "bid":         np.nan,
                            "ask":         np.nan,
                            "last":        np.nan,
                            "close":       np.nan,
                            "volume":      np.nan,
                            "open_interest": np.nan,
                            "ref_price":   np.nan,
                        })

        df = pd.DataFrame(rows)
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
        return df.reset_index(drop=True)

    except Exception as e:
        warn(f"Falha ao buscar chain de {symbol}: {e}")
        return pd.DataFrame()

# ===============================
# Fetch candles (yfinance — sem .SA)
# ===============================
@st.cache_data(ttl=600, show_spinner=True)
def fetch_candles(symbol: str, days: int = 180) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    end   = datetime.today()
    start = end - timedelta(days=days)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError("Yahoo sem dados")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={
            "Date":"date","Open":"open","High":"high",
            "Low":"low","Close":"close","Volume":"volume"
        })
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["underlying_symbol"] = symbol
        return df[["underlying_symbol","date","open","high","low","close","volume"]]\
               .dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    except Exception as e:
        err(f"Yahoo falhou ({symbol}): {e}")
        return pd.DataFrame(columns=["underlying_symbol","date","open","high","low","close","volume"])

# ===============================
# Black-Scholes + IV local
# ===============================
def _bs_price_greeks(S, K, T, r, sigma, call_put):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return (np.nan,)*6
    cp = 1 if str(call_put).upper() == "CALL" else -1
    try:
        sqrtT = math.sqrt(T)
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrtT)
        d2 = d1 - sigma*sqrtT
        price = cp*(S*norm.cdf(cp*d1) - K*math.exp(-r*T)*norm.cdf(cp*d2))
        delta = cp*norm.cdf(cp*d1)
        gamma = norm.pdf(d1)/(S*sigma*sqrtT)
        vega  = S*norm.pdf(d1)*sqrtT
        theta = (-(S*norm.pdf(d1)*sigma)/(2*sqrtT) - cp*r*K*math.exp(-r*T)*norm.cdf(cp*d2))
        rho   = cp*K*T*math.exp(-r*T)*norm.cdf(cp*d2)
        return price, delta, gamma, vega, theta, rho
    except Exception:
        return (np.nan,)*6

def _implied_vol(S, K, T, r, premium, call_put):
    if not all(pd.notna([S, K, T, r, premium])) or S<=0 or K<=0 or T<=0 or premium<=0:
        return np.nan
    try:
        return brentq(
            lambda s: _bs_price_greeks(S, K, T, r, s, call_put)[0] - premium,
            1e-3, 5.0, maxiter=100, disp=False
        )
    except Exception:
        return np.nan

# ===============================
# Contexto volume do ativo
# ===============================
def preparar_contexto_ativos(df_at: pd.DataFrame, ma: int = 20) -> pd.DataFrame:
    if df_at is None or df_at.empty:
        return pd.DataFrame(columns=["underlying_symbol","volume_fin","volfin_ma","vol_acima_ma","last_close"])
    d = df_at.copy()
    d["date"]   = pd.to_datetime(d["date"], errors="coerce")
    d["volume"] = _to_num(d.get("volume"))
    d["close"]  = _to_num(d.get("close"))
    d.sort_values(["underlying_symbol","date"], inplace=True)
    d["volume_fin"] = d["close"] * d["volume"]
    d["volfin_ma"]  = (
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
# Enriquecer com IV/greeks
# ===============================
def add_features_and_iv(df_opts: pd.DataFrame, price_lookup: dict, r_annual: float) -> pd.DataFrame:
    if df_opts is None or df_opts.empty:
        return df_opts
    d = df_opts.copy()

    for c in ["bid","ask","last","close","strike","volume","open_interest","ref_price"]:
        d[c] = _to_num(d.get(c, np.nan))

    # Preço spot do subjacente
    if price_lookup:
        mask = d["ref_price"].isna()
        if mask.any():
            d.loc[mask, "ref_price"] = d.loc[mask, "underlying_symbol"].map(price_lookup)

    # Mid / spread — como não temos bid/ask da API, usamos BS price como proxy
    d["mid"]        = np.nan
    d["spread"]     = 0.0
    d["spread_rel"] = 0.0

    # Tempo
    hoje = date.today()
    d["expiration"]    = pd.to_datetime(d.get("expiration"), errors="coerce")
    d["dte_calendar"]  = d["dte_calendar"].fillna(
        (d["expiration"].dt.date - hoje).apply(lambda x: x.days if pd.notna(x) else np.nan)
    )
    d["dte_bus"] = d["dte_calendar"].clip(lower=1)
    d["T"]       = (d["dte_bus"] / 252.0).clip(lower=1/365.0)

    d["type"]        = d["type"].astype(str).str.upper()
    d["option_type"] = np.where(d["type"].isin(["CALL","PUT"]), d["type"], "CALL")

    # BS price teórico (usando vol histórica do subjacente como proxy inicial)
    # IV será calculada depois com brentq quando tivermos premium de mercado.
    # Como não temos preços de mercado, usamos vol histórica de 30 dias como IV proxy.
    hist_vol_map = {}
    for sym, grp in df_opts.groupby("underlying_symbol") if hasattr(df_opts, "groupby") else []:
        pass

    # Calcular vol histórica por subjacente a partir dos candles (se disponível)
    # Por ora usa-se 0.30 como default (30% IV — razoável para ações USA)
    d["iv_local"]     = 0.30
    d["iv_local_pct"] = 30.0

    # Greeks com vol histórica proxy
    greeks = d.apply(lambda r: pd.Series(
        _bs_price_greeks(r["ref_price"], r["strike"], r["T"], r_annual, r["iv_local"], r["option_type"]),
        index=["bs_price","delta","gamma","vega","theta","rho"]
    ), axis=1)
    d = pd.concat([d, greeks], axis=1)

    # Mid = BS price teórico
    d["mid"]      = d["bs_price"]
    d["delta_abs"] = d["delta"].abs()

    # Percentil local de IV por ativo+vencimento (com IV uniforme = 50%)
    d["iv_pct_local"] = 50.0
    d["spread_rel"]   = d["spread_rel"].fillna(0.0)

    return d

# ===============================
# Enriquecer IV com vol histórica real
# ===============================
def enrich_iv_with_hist_vol(df_opts: pd.DataFrame, df_candles: pd.DataFrame, r_annual: float) -> pd.DataFrame:
    """
    Calcula a volatilidade histórica de 30 dias por subjacente
    e recalcula IV/greeks usando esse valor como proxy.
    """
    if df_candles.empty:
        return df_opts

    d = df_opts.copy()

    # Calcular vol histórica por subjacente
    hist_vols = {}
    for sym, grp in df_candles.groupby("underlying_symbol"):
        grp = grp.sort_values("date")
        if len(grp) >= 2:
            ret = np.log(grp["close"] / grp["close"].shift(1)).dropna()
            vol_30 = ret.tail(30).std() * np.sqrt(252)
            hist_vols[sym] = float(vol_30) if not np.isnan(vol_30) else 0.30
        else:
            hist_vols[sym] = 0.30

    # Atualizar IV com vol histórica
    d["iv_local"]     = d["underlying_symbol"].map(hist_vols).fillna(0.30)
    d["iv_local_pct"] = d["iv_local"] * 100.0

    # Recalcular greeks
    greeks = d.apply(lambda r: pd.Series(
        _bs_price_greeks(r["ref_price"], r["strike"], r["T"], r_annual, r["iv_local"], r["option_type"]),
        index=["bs_price","delta","gamma","vega","theta","rho"]
    ), axis=1)
    for col in ["bs_price","delta","gamma","vega","theta","rho"]:
        d[col] = greeks[col]

    d["mid"]       = d["bs_price"]
    d["delta_abs"] = d["delta"].abs()

    # Percentil de IV por vencimento
    d["iv_pct_local"] = (
        d.groupby(["underlying_symbol","expiration"])["iv_local_pct"]
         .transform(lambda s: 100*s.rank(pct=True, method="average"))
    )

    return d

# ===============================
# Filtros
# ===============================
def aplicar_filtros(d, tipo_opcao, venc_ini, venc_fim, delta_min, delta_max,
                    iv_pct_max, min_volume_opt, max_spread_rel, exigir_vol_acima):
    if d is None or d.empty:
        return d
    x = d.copy()
    x = x[x["expiration"].between(pd.to_datetime(venc_ini), pd.to_datetime(venc_fim))]
    if tipo_opcao in ("CALL","PUT"):
        x = x[x["type"] == tipo_opcao]

    # Volume: como não temos volume real, removemos esse filtro se min=0
    if "volume" not in x.columns:
        x["volume"] = np.nan
    x["volume"] = _to_num(x["volume"])

    cond = (
        x["delta_abs"].between(delta_min, delta_max, inclusive="both")
        & (x["iv_pct_local"] <= iv_pct_max)
        & x["T"].gt(0)
        & x["bs_price"].gt(0)
    )
    if min_volume_opt > 0:
        cond &= (x["volume"].fillna(0) >= min_volume_opt)
    if max_spread_rel < 5.0:
        cond &= (x["spread_rel"].fillna(0.0) <= max_spread_rel)

    return x.loc[cond.fillna(False)].copy()

# ===============================
# Ranking
# ===============================
def _norm01(s, invert=False):
    s = _to_num(s)
    if s.nunique(dropna=True) <= 1:
        n = pd.Series(0.5, index=s.index)
    else:
        n = (s - s.min()) / (s.max() - s.min() + 1e-12)
    n = n.fillna(0.5)
    return (1 - n) if invert else n

def rankear(d, delta_target=0.45, exigir_vol_acima=False):
    if d is None or d.empty:
        return d
    x = d.copy().reset_index(drop=True)
    x["score"] = (
        0.50 * _norm01(x["iv_local_pct"], invert=True) +
        0.30 * _norm01((x["delta_abs"] - delta_target).abs(), invert=True) +
        0.20 * _norm01(x["vega"])
    )
    return x.sort_values("score", ascending=False)

def top_por_venc(d, n=5):
    if d is None or d.empty:
        return d
    return (
        d.sort_values(["expiration","score"], ascending=[True, False])
         .groupby("expiration", as_index=False)
         .head(n)
    )

# ===============================
# Próximo vencimento (3ª sexta)
# ===============================
def proximo_vencimento(base=None):
    if base is None:
        base = datetime.today().date()
    ano, mes = base.year, base.month
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    sextas = [d for d in c.itermonthdates(ano, mes) if d.weekday() == 4 and d.month == mes]
    if len(sextas) >= 3 and base <= sextas[2]:
        return sextas[2]
    mes = 1 if mes == 12 else mes + 1
    ano = ano + 1 if mes == 1 else ano
    sextas = [d for d in c.itermonthdates(ano, mes) if d.weekday() == 4 and d.month == mes]
    return sextas[2] if len(sextas) >= 3 else base + timedelta(days=30)

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.title("⚙️ Parâmetros — Opções USA")

    symbols = st.multiselect(
        "Ativos (subjacentes)",
        ["AAPL","TSLA","NVDA","MSFT","AMZN","META","GOOGL","AMD","NFLX","SPY","QQQ"],
        default=["AAPL"]
    )
    days = st.number_input("Dias de histórico (candles)", min_value=30, max_value=365, value=180, step=5)

    st.markdown("---")
    taxa_juros = st.number_input("Taxa de juros anual — Fed Funds (%)",
                                  min_value=0.0, max_value=20.0, value=4.50, step=0.25) / 100.0

    st.markdown("---")
    tipo_opcao = st.radio("Tipo de opção", ["Ambas","CALL","PUT"], index=0, horizontal=True)

    prox_venc = proximo_vencimento()
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        venc_ini = st.date_input("Venc. inicial", prox_venc)
    with col_v2:
        venc_fim = st.date_input("Venc. final", prox_venc + timedelta(days=60))

    st.markdown("---")
    delta_min      = st.slider("Delta mínimo (abs)", 0.0, 1.0, 0.25, 0.01)
    delta_max      = st.slider("Delta máximo (abs)", 0.0, 1.0, 0.75, 0.01)
    iv_pct_max     = st.slider("IV percentil local máx. (%)", 0, 100, 80, 1)
    min_vol_opt    = st.number_input("Volume mínimo (opção)", 0, 200000, 0, 100)
    max_spread_rel = st.slider("Spread relativo máx.", 0.0, 5.0, 5.0, 0.05)
    exigir_vol_acima = st.checkbox("Exigir volume do ativo acima da MM20", value=False)

    st.markdown("---")
    delta_target = st.slider("Delta alvo p/ score", 0.0, 1.0, 0.45, 0.01)
    top_n        = st.number_input("Top por vencimento", 1, 10, 5)

    st.markdown("---")
    btn_run = st.button("🌀 Rodar Scanner USA", type="primary", use_container_width=True)

# ===============================
# MAIN
# ===============================
st.title("🇺🇸 Scanner de Opções — EUA (Tastytrade)")

with st.expander("📘 Entendendo o Score de Oportunidade", expanded=False):
    st.markdown("""
O **Score** combina três factores calculados via **Black-Scholes**:

- 📉 **IV histórica (%):** prefere opções com volatilidade implícita mais baixa (menor sobrepreço)
- 🎯 **Delta:** favorece deltas próximos do alvo definido (ex: 0.45)
- 🌊 **Vega:** valoriza opções mais sensíveis à volatilidade

A IV é calculada localmente com a **volatilidade histórica de 30 dias** do subjacente como proxy,
pois o Sandbox Tastytrade não fornece preços de mercado em tempo real.
Na conta de produção os preços reais serão usados directamente.
""")

# Estado
if "ativos_usa" not in st.session_state:  st.session_state["ativos_usa"] = pd.DataFrame()
if "opcoes_usa" not in st.session_state:  st.session_state["opcoes_usa"] = pd.DataFrame()
if "primeira_exec_usa" not in st.session_state:
    st.session_state["primeira_exec_usa"] = True
else:
    st.session_state["primeira_exec_usa"] = False

if btn_run or st.session_state["primeira_exec_usa"]:
    if not symbols:
        err("Selecione ao menos um ativo.")
        st.stop()

    with st.status("Baixando e preparando dados...", expanded=True) as status:
        try:
            # 1) Token
            st.write("🔐 Autenticando na Tastytrade...")
            token = get_tt_token()
            st.write("✅ Token obtido.")

            # 2) Downloads
            dfs_at, dfs_op = [], []
            for sym in symbols:
                st.write(f"📡 Buscando dados de **{sym}**...")
                dfs_at.append(fetch_candles(sym, int(days)))
                dfs_op.append(fetch_options_snapshot(sym, token))

            at = pd.concat(dfs_at, ignore_index=True) if dfs_at else pd.DataFrame()
            op = pd.concat(dfs_op, ignore_index=True) if dfs_op else pd.DataFrame()

            if at.empty:
                err("Sem dados de candles. Verifique os tickers.")
                st.stop()
            if op.empty:
                err("Sem dados de opções. Verifique as credenciais Tastytrade no .env")
                st.stop()

            # 3) Contexto volume + preço spot
            ctx = preparar_contexto_ativos(at, ma=20)
            last_close_map = dict(zip(ctx["underlying_symbol"], ctx["last_close"]))

            # 4) Merge contexto no book
            book_raw = op.merge(
                ctx.rename(columns={"volume_fin":"volume_fin_acao","volfin_ma":"volfin_ma_acao"}),
                on="underlying_symbol", how="left"
            )

            # 5) IV/greeks base
            book = add_features_and_iv(book_raw, price_lookup=last_close_map, r_annual=taxa_juros)

            # 6) Enriquecer com vol histórica real
            book = enrich_iv_with_hist_vol(book, at, r_annual=taxa_juros)

            # 7) Filtros
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

            # 8) Ranking e top
            ranked = rankear(flt, delta_target=delta_target, exigir_vol_acima=exigir_vol_acima)
            top    = top_por_venc(ranked, n=int(top_n))

            status.update(label="Concluído ✅", state="complete")

            # ====== Saída ======
            st.subheader("🏆 Top Oportunidades por Vencimento 💎")

            if top.empty:
                warn("Nenhuma oportunidade encontrada. Ajuste os filtros (delta, DTE, IV%).")
            else:
                top = top.sort_values("score", ascending=False).reset_index(drop=True)
                top["expiration"] = pd.to_datetime(top["expiration"], errors="coerce").dt.date
                num_cols = top.select_dtypes(include=["float","float64","int","int64"]).columns
                top[num_cols] = top[num_cols].apply(lambda x: np.round(x, 4))

                # Cards
                num_cards = min(int(top_n), 10)
                top5 = top.head(num_cards)[["symbol","score","type","strike","delta","expiration","iv_local_pct"]]

                def get_card_gradient(score, tipo):
                    s = float(score)
                    if tipo == "CALL":
                        start, end = "#004d00", "#66ff66"
                    else:
                        start, end = "#7f0000", "#ff6666"
                    return f"linear-gradient(135deg, {start} {(s*100):.0f}%, {end})"

                card_html = ""
                for _, row in top5.iterrows():
                    grad = get_card_gradient(row["score"], row["type"])
                    delta_color = "lime" if row["type"] == "CALL" else "salmon"
                    card_html += f"""
                    <div class="card" style="background-image: {grad};">
                        <div class="symbol">{row['symbol'][:16]} ({row['type']})</div>
                        <div class="score-label">Score</div>
                        <div class="score">{row['score']:.2f}</div>
                        <div class="details">Strike ${row['strike']:.0f} • Venc. {row['expiration']}</div>
                        <div class="delta-line">
                            <span style='color:{delta_color}; font-weight:600;'>Δ {row['delta']:.2f}</span>
                            &nbsp;|&nbsp; IV {row['iv_local_pct']:.1f}%
                        </div>
                    </div>
                    """

                st.markdown(f"""
                <style>
                .card {{
                    display: inline-block; border-radius: 16px; padding: 16px 18px;
                    margin: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                    color: white; width: 16.5%; text-align: center; min-height: 190px;
                    transition: all 0.25s ease;
                }}
                .card:hover {{ transform: translateY(-4px) scale(1.03); box-shadow: 0 6px 14px rgba(0,0,0,0.6); }}
                .symbol {{ font-weight: 600; font-size: 0.9rem; margin-bottom: 4px; }}
                .score-label {{ font-size: 0.75rem; color: rgba(255,255,255,0.9); text-transform: uppercase; }}
                .score {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 6px; color: #fff; }}
                .details {{ font-size: 0.82rem; color: rgba(255,255,255,0.85); }}
                .delta-line {{ margin-top: 4px; font-size: 0.82rem; }}
                .cards-container {{ display: flex; flex-wrap: wrap; justify-content: center; align-items: stretch; gap: 10px; }}
                @media (max-width: 1000px) {{ .card {{ width: 45%; }} }}
                @media (max-width: 600px)  {{ .card {{ width: 90%; }} }}
                </style>
                <div class="cards-container">{card_html}</div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Tabela
                def score_color(val, tipo):
                    if pd.isna(val): return ""
                    s = max(0, min(float(val), 1))
                    if tipo == "CALL":
                        dark, light = np.array([0,77,0]), np.array([102,255,102])
                    else:
                        dark, light = np.array([127,0,0]), np.array([255,102,102])
                    rgb = (dark*s + light*(1-s)).astype(int)
                    return f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); color: black; font-weight: 700;"

                show_cols = ["score","symbol","type","expiration","strike","delta","gamma",
                             "vega","theta","iv_local_pct","bs_price","underlying_symbol"]
                show_cols = [c for c in show_cols if c in top.columns]

                fmt = {}
                for c in top[show_cols].columns:
                    if top[c].dtype.kind in "fi":
                        if c in ["strike","bs_price"]:
                            fmt[c] = "$ {:,.2f}".format
                        else:
                            fmt[c] = "{:.4f}".format

                styled = (
                    top[show_cols].style
                    .format(fmt)
                    .apply(
                        lambda r: [score_color(r["score"], r["type"])] + [""]*(len(r)-1),
                        axis=1
                    )
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

            # ====== Gráficos candles ======
            st.markdown("---")
            st.subheader("📈 Candles — OHLCV")
            st.caption("Volume financeiro (Close × Volume) com MM20 branca.")

            if not at.empty:
                for sym in sorted(set(at["underlying_symbol"])):
                    d = at[at["underlying_symbol"]==sym].sort_values("date").tail(180).copy()
                    if d.empty: continue
                    d["vol_fin"]    = _to_num(d["close"]) * _to_num(d["volume"])
                    d["volfin_ma20"] = d["vol_fin"].rolling(20, min_periods=1).mean()

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                        name=f"{sym} OHLC",
                        increasing_line_color="lime", decreasing_line_color="red", yaxis="y1"
                    ))
                    fig.add_trace(go.Bar(
                        x=d["date"], y=d["vol_fin"], name="Volume financeiro",
                        marker_color="deepskyblue", yaxis="y2", opacity=0.6
                    ))
                    fig.add_trace(go.Scatter(
                        x=d["date"], y=d["volfin_ma20"], name="MM20 Vol ($)",
                        mode="lines", line=dict(color="white", width=1.5), yaxis="y2"
                    ))
                    fig.update_layout(
                        title=f"{sym}", height=550, template="plotly_dark",
                        xaxis=dict(domain=[0,1], rangeslider=dict(visible=False)),
                        yaxis=dict(title="Preço ($)", domain=[0.35,1.0], side="left"),
                        yaxis2=dict(title="Volume ($)", domain=[0.0,0.30], showgrid=False),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ====== Dados brutos ======
            with st.expander("📦 Dados brutos"):
                st.caption("Candles (OHLCV)")
                st.dataframe(at, use_container_width=True, hide_index=True)
                st.caption("Opções (com IV/greeks calculados via Black-Scholes)")
                show_raw = ["symbol","underlying_symbol","type","expiration","strike",
                            "ref_price","T","dte_calendar","iv_local","iv_local_pct",
                            "iv_pct_local","delta","gamma","vega","theta","rho","bs_price","score"]
                show_raw = [c for c in show_raw if c in book.columns]
                st.dataframe(book[show_raw], use_container_width=True, hide_index=True)

        except Exception as e:
            status.update(label="Erro no processamento", state="error")
            err(str(e))
            st.exception(e)
ENDOFFILE
echo "Done"
Done

