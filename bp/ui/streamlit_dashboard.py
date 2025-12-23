# 09_⚙️_Scanner_Acoes_TOP5_DF.py
# -*- coding: utf-8 -*-

import streamlit as st
import time
import pandas as pd
import os

from bp.core.data_loader import get_ticker_data, validate_data
from bp.core.indicators import apply_all_indicators
from bp.core.criteria_engine import evaluate_all_criteria
from bp.core.scoring import calculate_score
from bp.core.selectors import select_top_assets
from bp.ui.visual_blocks import criteria_block
from bp.ui.radar_chart import plot_radar
from bp.core.trade_engine import generate_trade_setup


# ------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------
def setup_page():
    st.set_page_config(
        page_title="BP Fênix – Scanner Ações",
        layout="wide",
        page_icon="🦅"
    )
    st.title("🦅 Projeto Fênix — Scanner de Ações")


# ------------------------------------------------------------
# LOCALIZAÇÃO DO CSV
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "tickers_ibov.csv")


# ------------------------------------------------------------
# CICLO PRINCIPAL
# ------------------------------------------------------------
def run_full_cycle_with_logs(tickers):

    st.info(f"🔍 Total de ativos carregados: **{len(tickers)}**")

    progress = st.progress(0)
    status_box = st.status("🚀 Iniciando varredura do BP-Fênix...", expanded=False)

    results = {}
    total = len(tickers)

    with status_box:
        st.write("### 📡 LOG DA EXECUÇÃO")

        for i, ticker in enumerate(tickers):
            ticker_api = ticker + ".SA"

            st.write(f"🔵 **Processando {ticker}...**")

            df = get_ticker_data(ticker_api)
            if not validate_data(df):
                st.write(f"⚠️ Dados inválidos para {ticker}. Pulando...")
                continue

            df = apply_all_indicators(df)
            if df is None or df.empty:
                st.write(f"⚠️ Indicadores retornaram dataframe vazio para {ticker}.")
                continue

            criteria = evaluate_all_criteria(df)
            score_info = calculate_score(criteria)

            score_info["details"]["df"] = df
            results[ticker] = score_info

            st.write(f"✔️ Score de {ticker}: **{score_info['score']} / 5**")
            st.write("—")

            progress.progress((i + 1) / total)
            time.sleep(0.05)

        st.success("🟩 Varredura concluída!")

    return {
        "raw_results": results,
        "top_assets": select_top_assets(results)
    }


# ------------------------------------------------------------
# DASHBOARD PRINCIPAL
# ------------------------------------------------------------
def render_dashboard():

    setup_page()

    st.sidebar.header("📡 Filtros do Fênix")

    df_tickers = pd.read_csv(CSV_PATH, sep=";")

    indices = ["TODOS"] + df_tickers["indice"].unique().tolist()

    indice_escolhido = st.sidebar.selectbox(
        "Selecione o índice:",
        options=indices,
        index=0
    )

    if indice_escolhido == "TODOS":
        tickers_filtrados = df_tickers["ticker"].dropna().unique().tolist()
    else:
        tickers_filtrados = (
            df_tickers[df_tickers["indice"] == indice_escolhido]["ticker"]
            .dropna()
            .unique()
            .tolist()
        )

    st.sidebar.markdown(f"**Ativos carregados:** {len(tickers_filtrados)}")

    if st.button("🌀 Rodar Varredura Agora"):
        with st.spinner("Executando ciclo do BP-Fênix..."):
            output = run_full_cycle_with_logs(tickers_filtrados)

        st.session_state["fenix_output"] = output
        st.success("Ciclo concluído!")

    output = st.session_state.get("fenix_output")
    if not output:
        st.info("◀️ Selecione um índice e rode a varredura.")
        return

    # ------------------------------------------------------------
    # 🔥 TOP 5
    # ------------------------------------------------------------
    st.markdown("## 🔥 Top Selecionados pelo BP-Fênix")

    rows = []

    for asset in output["top_assets"]:

        ticker = asset["ticker"]
        fs_value = asset.get("fs", 0)

        try:
            indice_ticker = df_tickers[df_tickers["ticker"] == ticker]["indice"].values[0]
        except Exception:
            indice_ticker = "DESCONHECIDO"

        try:
            df_original = asset["details"]["df"]
            trade = generate_trade_setup(df_original, fs_value)
        except Exception:
            trade = None

        if not trade:
            continue

        rows.append({
            "Selecionar": False,
            "Ticker": ticker,
            "Índice": indice_ticker,
            "Score": round(asset["score"], 2),
            "FS": round(fs_value, 3),
            "Operação": "COMPRA" if trade["operacao"] == "LONG" else "VENDA",
            "Entrada": round(trade["entrada"], 2),
            "Stop": round(trade["stop"], 2),
            "Alvo": round(trade["alvo"], 2),
            "R/R": round(trade["rr"], 2),
            "ATR Stop": round(trade["stop_dist_atr"], 2),
            "ATR Alvo": round(trade["target_dist_atr"], 2),
        })

    if not rows:
        st.warning("Nenhum ativo válido no Top 5.")
        return

    df_top5 = pd.DataFrame(rows)

    st.markdown("## 📋 Seleção para Relatório")

    df_top5 = st.data_editor(
        df_top5,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Selecionar": st.column_config.CheckboxColumn("✔"),
        }
    )

    # ------------------------------------------------------------
    # 📄 GERAR RELATÓRIO (placeholder)
    # ------------------------------------------------------------
    if st.button("📄 Gerar Relatório (ativos selecionados)"):

        selecionados = df_top5[df_top5["Selecionar"]]

        if selecionados.empty:
            st.warning("Selecione ao menos um ativo.")
        else:
            st.success(f"{len(selecionados)} ativo(s) selecionado(s).")
            st.dataframe(selecionados, use_container_width=True)


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    render_dashboard()
