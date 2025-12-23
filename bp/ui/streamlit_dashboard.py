# 09_⚙️_Scanner_Acoes_TOP5_DF.py
# -*- coding: utf-8 -*-

import streamlit as st
import time
import pandas as pd
import os
import io
import datetime
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

from bp.core.data_loader import get_ticker_data, validate_data
from bp.core.indicators import apply_all_indicators
from bp.core.criteria_engine import evaluate_all_criteria
from bp.core.scoring import calculate_score
from bp.core.selectors import select_top_assets
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
        for i, ticker in enumerate(tickers):
            df = get_ticker_data(ticker + ".SA")
            if not validate_data(df):
                continue

            df = apply_all_indicators(df)
            if df is None or df.empty:
                continue

            criteria = evaluate_all_criteria(df)
            score_info = calculate_score(criteria)
            score_info["details"]["df"] = df
            results[ticker] = score_info

            progress.progress((i + 1) / total)
            time.sleep(0.05)

    return {
        "raw_results": results,
        "top_assets": select_top_assets(results)
    }


# ------------------------------------------------------------
# UTIL – NOME DA EMPRESA
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def nome_empresa_yf(ticker):
    try:
        info = yf.Ticker(f"{ticker}.SA").info
        return info.get("longName") or info.get("shortName") or ticker
    except:
        return ticker


# ------------------------------------------------------------
# UTIL – CANDLE
# ------------------------------------------------------------
def gerar_candle_png(ticker):
    df = yf.download(f"{ticker}.SA", period="6mo", interval="1d", progress=False)
    if df.empty:
        return None

    df = df.tail(70)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df.index, df["Close"], linewidth=1.6)
    ax.set_title(ticker)
    ax.grid(alpha=0.3)

    fname = f"_tmp_{ticker}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fname


# ------------------------------------------------------------
# 📄 RELATÓRIO APIMEC (DF LOCAL)
# ------------------------------------------------------------
def export_pdf_apimec_df(
    df_sel: pd.DataFrame,
    nome_analista: str,
    certificado_cnpi: str,
    cpf_analista: str
) -> bytes:

    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=landscape(A4))
    W, H = landscape(A4)
    hoje = datetime.date.today()

    for _, row in df_sel.iterrows():

        ticker = row["Ticker"]
        empresa = nome_empresa_yf(ticker)
        operacao = row["Operação"]
        entrada = row["Entrada"]
        stop = row["Stop"]
        alvo = row["Alvo"]

        # Cabeçalho
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, H-2*cm, "RELATÓRIO DE ANÁLISE — APIMEC")

        c.setFont("Helvetica", 10)
        c.drawString(2*cm, H-2.7*cm, f"Data: {hoje:%d/%m/%Y}")
        c.drawString(2*cm, H-3.2*cm, f"Analista: {nome_analista} — {certificado_cnpi}")

        # Ativo
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, H-4.2*cm, f"{ticker} — {empresa}")

        # Tabela simples
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, H-5.2*cm, f"Operação: {operacao}")
        c.drawString(2*cm, H-5.8*cm, f"Entrada: R$ {entrada:.2f}")
        c.drawString(2*cm, H-6.4*cm, f"Stop: R$ {stop:.2f}")
        c.drawString(2*cm, H-7.0*cm, f"Alvo: R$ {alvo:.2f}")

        # Candle
        img = gerar_candle_png(ticker)
        if img and os.path.exists(img):
            c.drawImage(ImageReader(img), 2*cm, H-13*cm, width=W-4*cm, height=6*cm)
            os.remove(img)

        # Rodapé
        c.setFont("Helvetica", 9)
        c.drawString(2*cm, 2.8*cm, f"{nome_analista} ({certificado_cnpi})")
        c.drawString(2*cm, 2.3*cm, f"CPF: {cpf_analista}")

        c.showPage()

    c.save()
    buff.seek(0)
    return buff.read()


# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------
def render_dashboard():

    setup_page()
    df_tickers = pd.read_csv(CSV_PATH, sep=";")

    indice = st.sidebar.selectbox(
        "Índice",
        ["TODOS"] + df_tickers["indice"].unique().tolist()
    )

    if indice == "TODOS":
        tickers = df_tickers["ticker"].dropna().unique().tolist()
    else:
        tickers = df_tickers[df_tickers["indice"] == indice]["ticker"].tolist()

    if st.button("🌀 Rodar Varredura"):
        st.session_state["out"] = run_full_cycle_with_logs(tickers)

    out = st.session_state.get("out")
    if not out:
        return

    rows = []
    for a in out["top_assets"]:
        df = a["details"]["df"]
        trade = generate_trade_setup(df, a.get("fs", 0))
        if not trade:
            continue

        rows.append({
            "Selecionar": False,
            "Ticker": a["ticker"],
            "Índice": indice,
            "Score": a["score"],
            "FS": a.get("fs", 0),
            "Operação": "COMPRA" if trade["operacao"] == "LONG" else "VENDA",
            "Entrada": trade["entrada"],
            "Stop": trade["stop"],
            "Alvo": trade["alvo"],
        })

    df_top = pd.DataFrame(rows)

    df_top = st.data_editor(
        df_top,
        hide_index=True,
        column_config={"Selecionar": st.column_config.CheckboxColumn("✔")}
    )

    st.markdown("### 🧑‍💼 Dados do Analista")
    nome = st.text_input("Nome do Analista")
    cnpi = st.text_input("CNPI")
    cpf = st.text_input("CPF")

    if st.button("📄 Gerar Relatório APIMEC"):
        sel = df_top[df_top["Selecionar"]]
        if sel.empty:
            st.warning("Selecione ao menos um ativo.")
            return

        if not (nome and cnpi and cpf):
            st.warning("Preencha os dados do analista.")
            return

        pdf = export_pdf_apimec_df(sel, nome, cnpi, cpf)

        st.download_button(
            "⬇️ Baixar Relatório",
            data=pdf,
            file_name="Relatorio_APIMEC_Fenix.pdf",
            mime="application/pdf"
        )


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    render_dashboard()
