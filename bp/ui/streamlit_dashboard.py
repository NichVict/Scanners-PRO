# ============================================================
# 🦅 PROJETO FÊNIX — SCANNER DE AÇÕES + RELATÓRIO APIMEC
# Arquivo ÚNICO • Sem Supabase • Scanner → Relatório
# ============================================================

import streamlit as st
import time, os, io, datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle, Paragraph, Frame
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# IMPORTS DO BP-FÊNIX
# ============================================================
from bp.core.data_loader import get_ticker_data, validate_data
from bp.core.indicators import apply_all_indicators
from bp.core.criteria_engine import evaluate_all_criteria
from bp.core.scoring import calculate_score
from bp.core.selectors import select_top_assets
from bp.core.trade_engine import generate_trade_setup
from bp.ui.visual_blocks import criteria_block
from bp.ui.radar_chart import plot_radar

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Projeto Fênix — Scanner de Ações",
    layout="wide",
    page_icon="🦅"
)

st.title("🦅 Projeto Fênix — Scanner de Ações")

# ============================================================
# CSV DE TICKERS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "tickers_ibov.csv")

# ============================================================
# SCANNER
# ============================================================
def run_full_cycle_with_logs(tickers):
    results = {}
    for ticker in tickers:
        df = get_ticker_data(ticker + ".SA")
        if not validate_data(df):
            continue
        df = apply_all_indicators(df)
        criteria = evaluate_all_criteria(df)
        score_info = calculate_score(criteria)
        score_info["details"]["df"] = df
        results[ticker] = score_info

    return {
        "raw_results": results,
        "top_assets": select_top_assets(results)
    }

# ============================================================
# UTIL — Nome empresa
# ============================================================
@st.cache_data(ttl=3600)
def nome_empresa_yf(ticker):
    try:
        tk = yf.Ticker(f"{ticker}.SA")
        return tk.info.get("longName") or ticker
    except:
        return ticker

# ============================================================
# CONVERTE DF → ATIVOS APIMEC
# ============================================================
def ler_ativos_from_dataframe(df):
    ativos = []
    for _, r in df.iterrows():
        ativos.append({
            "ticker": r["Ticker"],
            "operacao": "compra" if r["Operação"] == "COMPRA" else "venda",
            "preco": float(r["Entrada"]),
            "stop_loss": float(r["Stop"]),
            "stop_gain": float(r["Alvo"]),
            "indice": r["Índice"],
        })
    return ativos

# ============================================================
# CANDLE
# ============================================================
def _candles_png(ticker):
    raw = yf.download(f"{ticker}.SA", period="6mo", progress=False)
    if raw.empty:
        return None

    df = raw[["Open","High","Low","Close","Volume"]].dropna().tail(70)
    df["MM8"] = df["Close"].rolling(8).mean()
    df["MM50"] = df["Close"].rolling(50).mean()

    dates = mdates.date2num(df.index.to_pydatetime())
    imgfile = f"_apimec_{ticker}.png"

    fig, ax = plt.subplots(figsize=(6,3))
    for i,(o,h,l,c) in enumerate(zip(df.Open,df.High,df.Low,df.Close)):
        col = "green" if c>=o else "red"
        ax.plot([dates[i],dates[i]],[l,h],color=col)
        ax.add_patch(plt.Rectangle((dates[i]-0.3,min(o,c)),0.6,abs(o-c),color=col))
    ax.plot(df.index, df.MM8)
    ax.plot(df.index, df.MM50)
    fig.savefig(imgfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return imgfile

# ============================================================
# RELATÓRIO APIMEC (PDF)
# ============================================================
def export_pdf_apimec_from_scanner(df_sel, nome, cnpi, cpf):

    ativos = ler_ativos_from_dataframe(df_sel)
    if not ativos:
        raise RuntimeError("Nenhum ativo selecionado.")

    data_hoje = datetime.date.today()
    filename = f"Relatorio_APIMEC_{data_hoje}.pdf"

    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=landscape(A4))
    W, H = landscape(A4)

    for a in ativos:
        tk = a["ticker"]
        op = a["operacao"].upper()
        alvo = a["preco"]
        empresa = nome_empresa_yf(tk)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, H-2*cm, "RELATÓRIO DE ANÁLISE — APIMEC")

        c.setFont("Helvetica", 10)
        c.drawString(2*cm, H-3*cm, f"{tk} — {empresa}")
        c.drawString(2*cm, H-3.6*cm, f"Operação: {op} | Preço alvo: R$ {alvo:.2f}")

        img = _candles_png(tk)
        if img:
            c.drawImage(ImageReader(img), 2*cm, H-12*cm, width=W-4*cm, height=6*cm)
            os.remove(img)

        c.setFont("Helvetica", 9)
        c.drawString(2*cm, 2.8*cm, f"{nome} ({cnpi}) — CPF {cpf}")
        c.showPage()

    c.save()
    with open(filename, "wb") as f:
        f.write(buff.getvalue())
    buff.close()
    return filename

# ============================================================
# DASHBOARD
# ============================================================
df_tickers = pd.read_csv(CSV_PATH, sep=";")

indices = ["TODOS"] + df_tickers["indice"].unique().tolist()
indice = st.sidebar.selectbox("Índice", indices)

tickers = df_tickers["ticker"].tolist() if indice=="TODOS" else \
          df_tickers[df_tickers.indice==indice]["ticker"].tolist()

if st.button("🔍 Rodar Varredura"):
    out = run_full_cycle_with_logs(tickers)
    st.session_state["out"] = out

out = st.session_state.get("out")
if out:
    rows=[]
    for a in out["top_assets"]:
        trade = generate_trade_setup(a["details"]["df"], a["fs"])
        rows.append({
            "Selecionar": False,
            "Ticker": a["ticker"],
            "Índice": indice,
            "Score": a["score"],
            "FS": a["fs"],
            "Operação": "COMPRA" if trade["operacao"]=="LONG" else "VENDA",
            "Entrada": trade["entrada"],
            "Stop": trade["stop"],
            "Alvo": trade["alvo"],
            "R/R": trade["rr"]
        })

    df = st.data_editor(pd.DataFrame(rows), hide_index=True)

    st.markdown("### 🔐 Dados do Analista")
    nome = st.text_input("Nome")
    cnpi = st.text_input("CNPI")
    cpf = st.text_input("CPF")

    if st.button("📄 Gerar Relatório APIMEC"):
        sel = df[df["Selecionar"]]
        pdf = export_pdf_apimec_from_scanner(sel, nome, cnpi, cpf)
        with open(pdf,"rb") as f:
            st.download_button("⬇️ Baixar PDF", f, file_name=pdf)
