# 09_⚙️_Scanner_Acoes_TOP5_DF_APIMEC.py
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
from bp.core.trade_engine import generate_trade_setup

# ============================================================
# 📄 RELATÓRIO APIMEC (a partir do DataFrame selecionado)
# - Sem Supabase
# - 1 ativo por página
# - Candle + narrativa técnica
# ============================================================
import io
import datetime
import numpy as np
import yfinance as yf

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader


@st.cache_data(ttl=3600)
def nome_empresa_yf(ticker_b3: str) -> str:
    """Tenta obter o nome da empresa via Yahoo Finance; fallback = ticker."""
    try:
        tk = yf.Ticker(f"{ticker_b3}.SA")

        # Preferência: longName -> shortName
        try:
            long_name = tk.info.get("longName")
            if long_name and isinstance(long_name, str):
                return long_name
        except Exception:
            pass

        try:
            short_name = tk.info.get("shortName")
            if short_name and isinstance(short_name, str):
                return short_name
        except Exception:
            pass

        # fast_info (quando disponível)
        try:
            fast = getattr(tk, "fast_info", None)
            if fast and isinstance(fast, dict) and fast.get("longName"):
                return fast["longName"]
        except Exception:
            pass

    except Exception:
        pass

    return ticker_b3


# --- Indicadores utilitários (para narrativa)
def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _sma(s, w): return s.rolling(w).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(close, fast=12, slow=26, signal=9):
    ef, es = _ema(close, fast), _ema(close, slow)
    line = ef - es
    sig = _ema(line, signal)
    return line, sig, line - sig


def narrativa_cnpi(df: pd.DataFrame, operacao: str, preco_alvo: float) -> str:
    """Gera um texto técnico (estilo CNPI) para COMPRA/VENDA."""
    # remover candle anômalo
    try:
        if len(df) >= 2 and ("Volume" in df.columns) and ("Close" in df.columns):
            if df["Volume"].iloc[-1] == 0 or (df["Close"].iloc[-1] < df["Close"].iloc[-2] * 0.8):
                df = df.iloc[:-1]
    except Exception:
        pass

    if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
        return "Amostra insuficiente para leitura técnica completa. Ativo segue em monitoramento."

    # volume financeiro
    vol_fin = (df["Volume"] * df["Close"]).iloc[-1]
    vol_fin_med = (df["Volume"] * df["Close"]).rolling(14).mean().iloc[-1]
    vol_rel = (vol_fin / vol_fin_med - 1) * 100 if (vol_fin_med is not None and vol_fin_med > 0) else 0

    close = float(df["Close"].iloc[-1])
    mm20 = float(df["Close"].rolling(20).mean().iloc[-1])
    rsi = float(_rsi(df["Close"]).iloc[-1])

    macd_l, macd_s, _ = _macd(df["Close"])
    macd_val = float(macd_l.iloc[-1])
    macd_sig = float(macd_s.iloc[-1])

    if pd.isna(mm20) or pd.isna(rsi) or pd.isna(vol_fin_med):
        return "Amostra insuficiente para leitura técnica completa. Ativo segue em monitoramento."

    txt_vol = (
        f"{vol_rel:.1f}% acima da média financeira de 14 períodos"
        if vol_fin > vol_fin_med else
        f"{abs(vol_rel):.1f}% abaixo da média financeira de 14 períodos"
    )

    operacao = (operacao or "").lower().strip()

    if operacao == "compra":
        stop = float(preco_alvo) * 0.98
        objetivo = float(preco_alvo) * 1.03
        stop_txt = f"stop técnico ({stop:.2f})"
        objetivo_txt = f"objetivo ({objetivo:.2f})"

        return (
            f"Preço atual em R$ {close:.2f}, operando {((close/mm20)-1)*100:.1f}% acima da MM20 (R$ {mm20:.2f}). "
            f"RSI em {rsi:.1f}, indicando força compradora sustentada. "
            f"MACD em {macd_val:.4f} acima da signal ({macd_sig:.4f}), reforçando momentum positivo. "
            f"Volume financeiro do dia em {vol_fin/1e6:.2f}M, {txt_vol}. "
            f"Cenário favorece posições compradas acima da MM20 com {stop_txt} "
            f"e {objetivo_txt} na região de topo recente."
        )

    if operacao == "venda":
        stop = float(preco_alvo) * 1.02
        objetivo = float(preco_alvo) * 0.97
        stop_txt = f"stop técnico ({stop:.2f})"
        objetivo_txt = f"objetivo ({objetivo:.2f})"

        return (
            f"Preço atual em R$ {close:.2f}, operando {((close/mm20)-1)*100:.1f}% abaixo da MM20 (R$ {mm20:.2f}). "
            f"RSI em {rsi:.1f}, sugerindo perda de força compradora e pressão vendedora. "
            f"MACD em {macd_val:.4f} abaixo da signal ({macd_sig:.4f}), indicando aceleração negativa. "
            f"Volume financeiro do dia em {vol_fin/1e6:.2f}M, {txt_vol}. "
            f"Estratégia favorece operações defensivas na ponta vendedora com {stop_txt} "
            f"e {objetivo_txt} na região de suporte recente."
        )

    return (
        f"Preço próximo à MM20 (R$ {mm20:.2f}). RSI em {rsi:.1f} e volume financeiro {txt_vol}. "
        f"Aguardar definição de direção com confirmação de volume."
    )


def _candles_png(ticker_b3: str) -> str | None:
    """Gera PNG de candle e retorna o caminho."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        raw = yf.download(f"{ticker_b3}.SA", period="6mo", interval="1d", progress=False)
        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]

        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if df.empty:
            return None

        # remover candle bugado
        if len(df) > 2:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            cond_volume_zero = last["Volume"] == 0
            cond_queda_irreal = last["Close"] < prev["Close"] * 0.8
            cond_gap_irreal = abs(last["Close"] - prev["Close"]) > prev["Close"] * 0.15
            if cond_volume_zero or cond_queda_irreal or cond_gap_irreal:
                df = df.iloc[:-1]

        df["MM8"] = df["Close"].rolling(8).mean()
        df["MM50"] = df["Close"].rolling(50).mean()
        df["VolMM14"] = df["Volume"].rolling(14).mean()
        df = df.tail(70)

        dates = mdates.date2num(df.index.to_pydatetime())
        imgfile = f"_apimec_{ticker_b3}.png"

        fig = plt.figure(figsize=(6.4, 3))
        gs = fig.add_gridspec(5, 1, height_ratios=[4, 0.15, 1.8, 0.2, 0.1])

        ax_price = fig.add_subplot(gs[0, 0])
        ax_vol = fig.add_subplot(gs[2, 0], sharex=ax_price)

        navy = "#001A33"
        watermark = "#BFE4FF"

        fig.patch.set_facecolor(navy)
        ax_price.set_facecolor(navy)
        ax_vol.set_facecolor(navy)

        ax_price.text(
            0.5, 0.5, ticker_b3,
            fontsize=72, weight="bold", color=watermark, alpha=0.10,
            ha="center", va="center", transform=ax_price.transAxes
        )

        # candles
        for i, (o, h, l, c_) in enumerate(zip(df["Open"], df["High"], df["Low"], df["Close"])):
            color = "#06D6A0" if c_ >= o else "#EF476F"
            ax_price.plot([dates[i], dates[i]], [l, h], color=color, linewidth=1.7)
            ax_price.add_patch(
                plt.Rectangle((dates[i] - 0.35, min(o, c_)), 0.7, abs(o - c_), color=color, ec=color)
            )

        ax_price.plot(df.index, df["MM8"], color="#2E86FF", linewidth=1.2)
        mm50 = df["MM50"].dropna()
        if not mm50.empty:
            ax_price.plot(mm50.index, mm50, color="#FFD700", linewidth=1.5, alpha=0.9)

        for i, (v, o, c_) in enumerate(zip(df["Volume"], df["Open"], df["Close"])):
            color = "#06D6A0" if c_ >= o else "#EF476F"
            ax_vol.bar(dates[i], v, width=0.6, color=color)
        ax_vol.plot(df.index, df["VolMM14"], color="#C77DFF", linewidth=1.1)

        ax_price.grid(which="major", linestyle="--", linewidth=0.35, alpha=0.30, color="#8AA1C1")
        ax_vol.grid(which="major", linestyle="--", linewidth=0.25, alpha=0.25, color="#8AA1C1")

        ax_price.yaxis.tick_right()
        ax_price.yaxis.set_major_formatter(lambda x, _: f"{x:.2f}")
        ax_price.tick_params(axis="y", labelsize=7, colors="white")

        ax_vol.set_yticks([])

        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax_price.tick_params(axis="x", labelsize=6, rotation=0, colors="white")
        plt.setp(ax_vol.get_xticklabels(), fontsize=6, color="white")

        fig.subplots_adjust(left=0.04, right=0.97, top=0.97, bottom=0.22, hspace=0.05)
        fig.savefig(imgfile, dpi=350, bbox_inches="tight")
        plt.close(fig)
        return imgfile

    except Exception:
        return None


def export_pdf_apimec_from_df(
    df_sel: pd.DataFrame,
    nome_analista: str,
    certificado_cnpi: str,
    cpf_analista: str,
) -> str:
    """Gera o PDF APIMEC a partir do DataFrame selecionado."""
    if df_sel is None or df_sel.empty:
        raise RuntimeError("Nenhum ativo selecionado para gerar relatório.")

    data_hoje = datetime.date.today()
    filename = f"Relatorio_APIMEC_SCANNER_{data_hoje}.pdf"

    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=landscape(A4))
    W, H = landscape(A4)

    for _, row in df_sel.iterrows():
        tk = str(row.get("Ticker", "")).upper().strip()
        if not tk:
            continue

        op_ui = str(row.get("Operação", "COMPRA")).upper().strip()
        operacao = "compra" if "COMPRA" in op_ui else "venda"

        preco_alvo = row.get("Entrada", None)
        try:
            preco_alvo = float(preco_alvo)
        except Exception:
            preco_alvo = None

        if preco_alvo is None:
            try:
                preco_alvo = float(row.get("Preço alvo"))
            except Exception:
                preco_alvo = 0.0

        nome = nome_empresa_yf(tk)

        # Cabeçalho
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, H - 2 * cm, "RELATÓRIO DE ANÁLISE — APIMEC")

        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, H - 2.6 * cm, f"Data: {data_hoje:%d/%m/%Y}")
        c.drawString(2 * cm, H - 3.1 * cm, f"Analista: {nome_analista} — {certificado_cnpi}")

        # Título
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, H - 4 * cm, f"{tk} — {nome}")

        # Tabela
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors

        tbl = [
            ["Ticker", "Empresa", "Operação", "Preço alvo"],
            [tk, nome, op_ui, f"R$ {preco_alvo:,.2f}".replace(".", ",")],
        ]
        col_w = (W - 4 * cm) / 4
        t = Table(tbl, colWidths=[col_w] * 4)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        t.wrapOn(c, W, H)
        t.drawOn(c, 2 * cm, H - 6.5 * cm)

        # Candle
        imgfile = _candles_png(tk)
        if imgfile and os.path.exists(imgfile):
            try:
                c.drawImage(
                    ImageReader(imgfile),
                    2 * cm, H - 13 * cm,
                    width=W - 4 * cm,
                    height=6 * cm,
                )
                c.setFont("Helvetica-Oblique", 7)
                c.drawString(2 * cm, H - 13.4 * cm, 'Fonte: disponível no website "yahoofinance.com"')
            except Exception:
                c.setFont("Helvetica-Oblique", 10)
                c.drawString(2 * cm, H - 7.5 * cm, "Erro ao carregar gráfico.")
            finally:
                try:
                    os.remove(imgfile)
                except Exception:
                    pass
        else:
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(2 * cm, H - 7.5 * cm, "Gráfico indisponível.")

        # Narrativa
        try:
            hist = yf.download(f"{tk}.SA", period="1y", interval="1d", progress=False).dropna()
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if len(hist) < 60 or "Close" not in hist.columns:
                texto = "Amostra insuficiente para leitura técnica."
            else:
                texto = narrativa_cnpi(hist, operacao, preco_alvo)
        except Exception:
            texto = "Amostra insuficiente para leitura técnica."

        # Recomendação
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, 6.3 * cm, f"Recomendação: {op_ui}")

        # Texto técnico
        c.setFont("Helvetica", 9)
        tx = c.beginText()
        tx.setTextOrigin(2 * cm, 5.8 * cm)
        tx.setLeading(12)
        for line in str(texto).split(". "):
            line = line.strip()
            if not line:
                continue
            if not line.endswith("."):
                line += "."
            tx.textLine(line)
        c.drawText(tx)

        # Rodapé
        c.setFont("Helvetica", 9)
        c.drawString(2 * cm, 3.2 * cm, f"{data_hoje:%d/%m/%Y}")
        c.drawString(2 * cm, 2.8 * cm, f"{nome_analista} ({certificado_cnpi})")
        c.drawString(2 * cm, 2.4 * cm, f"CPF nº {cpf_analista}")

        c.setFont("Helvetica-Oblique", 7)
        c.drawString(
            2 * cm, 1.4 * cm,
            "As recomendações refletem análise independente e não garantem resultados futuros."
        )

        c.showPage()

    # Página final
    from reportlab.platypus import Paragraph, Frame
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    style_title = styles["Heading2"]
    style_title.fontSize = 12
    style_text = styles["Normal"]
    style_text.fontSize = 9
    style_text.leading = 14

    titulo = Paragraph("Declarações Importantes do Relatório", style_title)

    texto_final = (
        "Objetivo: compartilhar o melhor entendimento técnico sobre a ação na presente data.<br/><br/>"
        "As recomendações não constituem promessa de resultados futuros e não garantem rentabilidade.<br/><br/>"
        "As recomendações refletem exclusivamente a opinião independente do analista certificado.<br/><br/>"
        "Stop Loss: nível máximo de perda; pode não ser respeitado em gaps.<br/><br/>"
        "Stop Gain: alvo técnico estimado para realização parcial ou total da operação.<br/><br/>"
        f"Analista responsável: <b>{nome_analista}</b> — <b>{certificado_cnpi}</b><br/>"
        f"CPF: <b>{cpf_analista}</b><br/><br/>"
        f"Código interno: <b>{datetime.date.today():%d%m%Y}</b>"
    )

    paragrafo = Paragraph(texto_final, style_text)
    frame = Frame(2 * cm, 2 * cm, W - 4 * cm, H - 4 * cm, showBoundary=0)
    frame.addFromList([titulo, paragrafo], c)
    c.showPage()

    c.save()

    with open(filename, "wb") as f:
        f.write(buff.getvalue())
    buff.close()

    return filename


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

    # Top 5
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
            "Score": round(asset.get("score", 0), 2),
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
        column_config={"Selecionar": st.column_config.CheckboxColumn("✔")},
    )

    selecionados = df_top5[df_top5["Selecionar"]].copy()

    # Relatório APIMEC
    st.markdown("---")
    st.markdown("## 📑 Relatório APIMEC (PDF)")

    col1, col2, col3 = st.columns(3)
    with col1:
        nome_analista = st.text_input("Nome completo do analista", key="apimec_nome")
    with col2:
        certificado_cnpi = st.text_input("Número do certificado CNPI", placeholder="Ex: CNPI EM-12345", key="apimec_cnpi")
    with col3:
        cpf_analista = st.text_input("CPF do analista", placeholder="Ex: 123.456.789-00", key="apimec_cpf")

    if st.button("📄 Gerar Relatório APIMEC (ativos selecionados)", type="primary"):
        if selecionados.empty:
            st.warning("Selecione ao menos um ativo no Top 5.")
            st.stop()

        if not (nome_analista and certificado_cnpi and cpf_analista):
            st.error("Preencha Nome, CNPI e CPF antes de gerar o relatório.")
            st.stop()

        try:
            pdf_path = export_pdf_apimec_from_df(
                selecionados,
                nome_analista=nome_analista,
                certificado_cnpi=certificado_cnpi,
                cpf_analista=cpf_analista,
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Baixar Relatório APIMEC",
                    f,
                    file_name=pdf_path,
                    mime="application/pdf"
                )

            st.success("✅ Relatório APIMEC gerado com sucesso!")

        except Exception as e:
            st.error(f"Erro ao gerar PDF: {e}")


if __name__ == "__main__":
    render_dashboard()
