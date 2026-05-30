# 09_⚙️_Scanner_Acoes_INTEGRADO_APIMEC.py
# -*- coding: utf-8 -*-

"""
Projeto Fênix — Scanner de Ações

✅ Preserva o Scanner original (radar + critérios + card de setup)
✅ Adiciona (no SIDEBAR) a seção de seleção Top 5 (DataFrame com checkbox)
✅ Gera Relatório APIMEC (PDF) baseado nos ativos selecionados
❌ Remove botões de envio para robô (sem Supabase)
"""

import io
import os
import time
import datetime
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# PDF / gráficos
import matplotlib
matplotlib.use("Agg")  # Streamlit-safe
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
from bp.ui.visual_blocks import criteria_block
from bp.ui.radar_chart import plot_radar


# ------------------------------------------------------------
# CSS (preserva o comportamento de expander suave)
# ------------------------------------------------------------
st.markdown(
    """
<style>
div.streamlit-expanderContent {
    overflow: hidden;
    transition: max-height 0.35s ease-out;
}
details[open] > div.streamlit-expanderContent { max-height: 800px; }
details:not([open]) > div.streamlit-expanderContent { max-height: 0px; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LOCALIZAÇÃO DO CSV
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "tickers_ibov.csv")


# ------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------
def setup_page():
    st.set_page_config(
        page_title="BP Fênix – Scanner Ações",
        layout="wide",
        page_icon="🦅",
    )
    st.title("🦅 Projeto Fênix — Scanner de Ações")


# ------------------------------------------------------------
# CICLO PRINCIPAL
# ------------------------------------------------------------
# ------------------------------------------------------------
# CICLO PRINCIPAL
# ------------------------------------------------------------
def _build_ticker_api(ticker: str, indice: str) -> str:
    """
    USA: ticker direto, sem sufixo.
    Todos os outros (IBOV, SMALL CAPS, BDR, ETF): sufixo .SA
    """
    if indice == "USA":
        return ticker
    return ticker + ".SA"


def run_full_cycle_with_logs(tickers: list[str], df_tickers: pd.DataFrame):
    st.info(f"🔍 Total de ativos carregados: **{len(tickers)}**")

    progress = st.progress(0)
    status_box = st.status("🚀 Iniciando varredura do BP-Fênix...", expanded=False)

    results: dict = {}
    total = max(len(tickers), 1)

    ticker_indice_map = (
        df_tickers.drop_duplicates(subset="ticker")
        .set_index("ticker")["indice"]
        .to_dict()
    )

    with status_box:
        st.write("### 📡 LOG DA EXECUÇÃO")

        for i, ticker in enumerate(tickers):
            indice = ticker_indice_map.get(ticker, "IBOV")
            ticker_api = _build_ticker_api(ticker, indice)

            st.write(f"🔵 **Processando {ticker}** ({indice}) → `{ticker_api}`")

            df = get_ticker_data(ticker_api)
            if not validate_data(df):
                st.write(f"⚠️ Dados inválidos para {ticker}. Pulando...")
                progress.progress((i + 1) / total)
                continue

            df = apply_all_indicators(df)
            if df is None or df.empty:
                st.write(f"⚠️ Indicadores retornaram dataframe vazio para {ticker}.")
                progress.progress((i + 1) / total)
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
        "top_assets": select_top_assets(results),
    }


# ------------------------------------------------------------
# RELATÓRIO APIMEC (baseado no exemplo que você enviou)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def nome_empresa_yf(ticker_b3: str) -> str:
    """Tenta puxar nome longo/curto do Yahoo. Fallback = ticker."""
    try:
        tk = yf.Ticker(f"{ticker_b3}.SA")

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

        try:
            fast = tk.fast_info
            if fast and "longName" in fast and fast["longName"]:
                return fast["longName"]
        except Exception:
            pass
    except Exception:
        pass

    return ticker_b3


def _ema(s, span):  # noqa: ANN001
    return s.ewm(span=span, adjust=False).mean()


def _sma(s, w):  # noqa: ANN001
    return s.rolling(w).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef, es = _ema(close, fast), _ema(close, slow)
    line = ef - es
    sig = _ema(line, signal)
    return line, sig, line - sig


def narrativa_cnpi(df: pd.DataFrame, operacao: str, preco_alvo: float) -> str:
    # --- Remover último candle anômalo (volume zero ou variação extrema do Yahoo) ---
    if len(df) >= 2:
        if df["Volume"].iloc[-1] == 0 or (df["Close"].iloc[-1] < df["Close"].iloc[-2] * 0.8):
            df = df.iloc[:-1].copy()

    # --- Volume financeiro ---
    vol_fin = (df["Volume"] * df["Close"]).iloc[-1]
    vol_fin_med = (df["Volume"] * df["Close"]).rolling(14).mean().iloc[-1]
    vol_rel = (vol_fin / vol_fin_med - 1) * 100 if vol_fin_med and vol_fin_med > 0 else 0

    close = df["Close"].iloc[-1]
    mm20 = df["Close"].rolling(20).mean().iloc[-1]
    rsi = _rsi(df["Close"]).iloc[-1]

    macd_l, macd_s, _ = _macd(df["Close"])
    macd_val = macd_l.iloc[-1]
    macd_sig = macd_s.iloc[-1]

    if pd.isna(mm20) or pd.isna(rsi) or pd.isna(vol_fin_med):
        return "Amostra insuficiente para leitura técnica completa. Ativo segue em monitoramento."

    txt_vol = (
        f"{vol_rel:.1f}% acima da média financeira de 14 períodos"
        if vol_fin > vol_fin_med
        else f"{abs(vol_rel):.1f}% abaixo da média financeira de 14 períodos"
    )

    if operacao.lower() == "compra":
        stop = preco_alvo * 0.98
        objetivo = preco_alvo * 1.03
        return (
            f"Preço atual em R$ {close:.2f}, operando {((close/mm20)-1)*100:.1f}% acima da MM20 (R$ {mm20:.2f}). "
            f"RSI em {rsi:.1f}, indicando força compradora sustentada. "
            f"MACD em {macd_val:.4f} acima da signal ({macd_sig:.4f}), reforçando momentum positivo. "
            f"Volume financeiro do dia em {vol_fin/1e6:.2f}M, {txt_vol}. "
            f"Cenário favorece posições compradas acima da MM20 com stop técnico ({stop:.2f}) "
            f"e objetivo ({objetivo:.2f}) na região de topo recente."
        )

    if operacao.lower() == "venda":
        stop = preco_alvo * 1.02
        objetivo = preco_alvo * 0.97
        return (
            f"Preço atual em R$ {close:.2f}, operando {((close/mm20)-1)*100:.1f}% abaixo da MM20 (R$ {mm20:.2f}). "
            f"RSI em {rsi:.1f}, sugerindo perda de força compradora e pressão vendedora. "
            f"MACD em {macd_val:.4f} abaixo da signal ({macd_sig:.4f}), indicando aceleração negativa. "
            f"Volume financeiro do dia em {vol_fin/1e6:.2f}M, {txt_vol}. "
            f"Estratégia favorece operações defensivas na ponta vendedora com stop técnico ({stop:.2f}) "
            f"e objetivo ({objetivo:.2f}) na região de suporte recente."
        )

    return (
        f"Preço próximo à MM20 (R$ {mm20:.2f}). RSI em {rsi:.1f} e volume financeiro {txt_vol}. "
        f"Aguardar definição de direção com confirmação de volume."
    )


def _candles_png(ticker_b3: str) -> Optional[str]:
    """Gera PNG com candles + MMs + volume (baseado no modelo aprovado)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        raw = yf.download(f"{ticker_b3}.SA", period="6mo", interval="1d", progress=False)
        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]

        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

        # --- Remover último candle bugado do Yahoo ---
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
            0.5,
            0.5,
            ticker_b3,
            fontsize=72,
            weight="bold",
            color=watermark,
            alpha=0.10,
            ha="center",
            va="center",
            transform=ax_price.transAxes,
        )

        # Candles
        for i, (o, h, l, c) in enumerate(zip(df["Open"], df["High"], df["Low"], df["Close"])):
            color = "#06D6A0" if c >= o else "#EF476F"
            ax_price.plot([dates[i], dates[i]], [l, h], color=color, linewidth=1.7)
            ax_price.add_patch(
                plt.Rectangle(
                    (dates[i] - 0.35, min(o, c)),
                    0.7,
                    abs(o - c),
                    color=color,
                    ec=color,
                )
            )

        ax_price.plot(df.index, df["MM8"], color="#2E86FF", linewidth=1.2)

        mm50 = df["MM50"].dropna()
        if not mm50.empty:
            ax_price.plot(mm50.index, mm50, color="#FFD700", linewidth=1.5, alpha=0.9)

        # Volume
        for i, (v, o, c) in enumerate(zip(df["Volume"], df["Open"], df["Close"])):
            color = "#06D6A0" if c >= o else "#EF476F"
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


def export_pdf_apimec_selected(
    df_selected: pd.DataFrame,
    nome_analista: str,
    certificado_cnpi: str,
    cpf_analista: str,
) -> str:
    """Gera PDF APIMEC para os ativos selecionados no Top 5."""
    if df_selected is None or df_selected.empty:
        raise RuntimeError("Nenhum ativo selecionado para o relatório.")

    data_hoje = datetime.date.today()
    filename = f"Relatorio_APIMEC_Scanner_{data_hoje}.pdf"

    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=landscape(A4))
    W, H = landscape(A4)

    for _, row in df_selected.iterrows():
        tk = str(row["Ticker"]).upper().strip()
        op = str(row["Operação"]).upper().strip()  # COMPRA / VENDA
        alvo = float(row["Entrada"])  # preço alvo = entrada do setup
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

        # Tabela resumida
        tbl = [
            ["Ticker", "Empresa", "Operação", "Preço alvo", "Stop", "Alvo"],
            [
                tk,
                nome,
                op,
                f"R$ {alvo:,.2f}".replace(".", ","),
                f"R$ {float(row['Stop']):,.2f}".replace(".", ","),
                f"R$ {float(row['Alvo']):,.2f}".replace(".", ","),
            ],
        ]

        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors

        col_w = (W - 4 * cm) / len(tbl[0])
        t = Table(tbl, colWidths=[col_w] * len(tbl[0]))
        t.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        t.wrapOn(c, W, H)
        t.drawOn(c, 2 * cm, H - 6.6 * cm)

        # Candle
        imgfile = _candles_png(tk)
        if imgfile and os.path.exists(imgfile):
            try:
                c.drawImage(ImageReader(imgfile), 2 * cm, H - 13 * cm, width=W - 4 * cm, height=6 * cm)
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

        # Texto técnico
        try:
            hist = yf.download(f"{tk}.SA", period="1y", interval="1d", progress=False).dropna()
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if len(hist) < 60 or "Close" not in hist.columns:
                texto = "Amostra insuficiente para leitura técnica."
            else:
                # narrativa usa compra/venda em minúsculo
                texto = narrativa_cnpi(hist, "compra" if op == "COMPRA" else "venda", alvo)
        except Exception:
            texto = "Amostra insuficiente para leitura técnica."

        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, 6.3 * cm, f"Recomendação: {op}")

        c.setFont("Helvetica", 9)
        tx = c.beginText()
        tx.setTextOrigin(2 * cm, 5.8 * cm)
        tx.setLeading(12)
        for line in texto.split(". "):
            tx.textLine(line.strip() + ".")
        c.drawText(tx)

        # Rodapé
        c.setFont("Helvetica", 9)
        c.drawString(2 * cm, 3.2 * cm, f"{data_hoje:%d/%m/%Y}")
        c.drawString(2 * cm, 2.8 * cm, f"{nome_analista} ({certificado_cnpi})")
        c.drawString(2 * cm, 2.4 * cm, f"CPF nº {cpf_analista}")

        c.setFont("Helvetica-Oblique", 7)
        c.drawString(2 * cm, 1.4 * cm, "As recomendações refletem análise independente e não garantem resultados futuros.")

        c.showPage()

    # Página final (declarações)
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
# DETALHES POR ATIVO (preservado, sem botão de envio)
# ------------------------------------------------------------
def show_asset_details(
    ticker: str,
    details: dict,
    fs_components: Optional[dict] = None,
    trade: Optional[dict] = None,
    indice_ticker: Optional[str] = None,
):
    if indice_ticker is None:
        indice_ticker = "DESCONHECIDO"

    # FS explainer
    if fs_components:
        with st.expander("📊 Detalhamento do Fênix Strength (FS)"):
            st.markdown(
                """
O **Fênix Strength (FS)** é a média normalizada dos 5 pilares:

- **Tendência**
- **Momentum**
- **Volatilidade**
- **Sinal Técnico**
- **Volume**
"""
            )
            st.write(
                {
                    "Tendência (norm)": round(fs_components.get("tendencia_norm", 0), 3),
                    "Momentum (norm)": round(fs_components.get("momentum_norm", 0), 3),
                    "Volatilidade (norm)": round(fs_components.get("volatilidade_norm", 0), 3),
                    "Sinal Técnico (norm)": round(fs_components.get("sinal_norm", 0), 3),
                    "Volume (norm)": round(fs_components.get("volume_norm", 0), 3),
                    "FS Total": round(fs_components.get("fs", 0), 3),
                }
            )

    # Radar
    radar = plot_radar(details)
    st.plotly_chart(
        radar,
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"radar_{ticker}",
    )

    # Critérios
    st.subheader("Critérios Avaliados")

    with st.expander("📈 Tendência"):
        criteria_block("Tendência", details["tendencia"]["status"], details["tendencia"]["detail"])

    with st.expander("⚡ Momentum"):
        criteria_block("Momentum", details["momentum"]["status"], details["momentum"]["detail"])

    with st.expander("🌪️ Volatilidade"):
        criteria_block("Volatilidade", details["volatilidade"]["status"], details["volatilidade"]["detail"])

    with st.expander("🎯 Sinal Técnico"):
        criteria_block("Sinal Técnico", details["sinal_tecnico"]["status"], details["sinal_tecnico"]["detail"])

    with st.expander("📊 Volume"):
        criteria_block("Volume", details["volume"]["status"], details["volume"]["detail"])

    # Card do setup (preservado)
    if trade:
        bg_color = "#0E3D1D" if trade["operacao"] == "LONG" else "#3D0E0E"
        border_color = "#27E062" if trade["operacao"] == "LONG" else "#FF4D4D"
        label = "COMPRA (LONG)" if trade["operacao"] == "LONG" else "VENDA (SHORT)"

        card_html = f"""
<div style="background-color:{bg_color};
            border-left:6px solid {border_color};
            padding:18px;
            border-radius:10px;
            margin-top:22px;
            margin-bottom:10px;
            color:white;
            font-family:'Segoe UI',sans-serif;">
  <div style="font-size:20px;font-weight:700;margin-bottom:4px;">
    {label} — {ticker}
  </div>
  <div style="font-size:13px;opacity:0.8;margin-bottom:10px;">
    Modelo C — Setup Profissional Adaptativo Fênix
  </div>
  <div style="font-size:13px;opacity:0.8;margin-bottom:10px;">
    Índice: <b>{indice_ticker}</b>
  </div>
  <div style="font-size:16px;line-height:1.6;">
    <b>🎯 Entrada:</b> {trade["entrada"]:.2f}<br>
    <b>🛑 Stop Loss:</b> {trade["stop"]:.2f}<br>
    <b>🎯 Take Profit:</b> {trade["alvo"]:.2f}<br>
    <b>📊 Risco/Retorno:</b> {trade["rr"]:.2f}
  </div>
  <div style="font-size:12px;opacity:0.7;margin-top:8px;">
    Stops: {trade["stop_dist_atr"]:.2f} ATR — Alvo: {trade["target_dist_atr"]:.2f} ATR
  </div>
</div>
"""
        st.markdown(card_html, unsafe_allow_html=True)


# ------------------------------------------------------------
# TABELA COMPLETA (raw)
# ------------------------------------------------------------
def show_results_table(results: dict):
    rows = []
    for ticker, item in results.items():
        rows.append(
            {
                "Ticker": ticker,
                "Score": item.get("score"),
                "Passaram": ", ".join(item.get("passed", [])),
                "Falharam": ", ".join(item.get("failed", [])),
            }
        )
    st.dataframe(rows, use_container_width=True)


# ------------------------------------------------------------
# SIDEBAR: Top 5 + Relatório APIMEC
# ------------------------------------------------------------
def sidebar_report_section(df_top5: pd.DataFrame):
    st.sidebar.markdown("### 📄 Relatório (Top 5)")
    with st.sidebar.expander("📋 Seleção para Relatório APIMEC", expanded=True):
        st.caption("Marque os ativos e gere o PDF APIMEC (sem login).")

        nome_analista = st.text_input("Nome completo do analista", key="apimec_nome")
        certificado_cnpi = st.text_input("Número do certificado CNPI", key="apimec_cnpi", placeholder="Ex: CNPI EM-12345")
        cpf_analista = st.text_input("CPF do analista", key="apimec_cpf", placeholder="Ex: 123.456.789-00")

        df_edit = st.data_editor(
            df_top5,
            hide_index=True,
            use_container_width=True,
            column_config={"Selecionar": st.column_config.CheckboxColumn("✔")},
            key="top5_editor",
        )

        if st.button("📄 Gerar Relatório APIMEC (selecionados)", key="btn_pdf_apimec"):
            selecionados = df_edit[df_edit["Selecionar"]].copy()

            if selecionados.empty:
                st.warning("Selecione ao menos um ativo.")
                return

            if not (nome_analista and certificado_cnpi and cpf_analista):
                st.error("Preencha nome, CNPI e CPF antes de gerar o relatório.")
                return

            try:
                pdf = export_pdf_apimec_selected(
                    selecionados,
                    nome_analista=nome_analista,
                    certificado_cnpi=certificado_cnpi,
                    cpf_analista=cpf_analista,
                )

                with open(pdf, "rb") as f:
                    st.download_button(
                        "⬇️ Baixar Relatório APIMEC",
                        f,
                        file_name=pdf,
                        mime="application/pdf",
                        key="download_apimec_pdf",
                    )

                st.success("✅ Relatório APIMEC gerado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")


# ------------------------------------------------------------
# DASHBOARD PRINCIPAL
# ------------------------------------------------------------
def render_dashboard():
    setup_page()

    st.sidebar.header("📡 Filtros do Fênix")

    df_tickers = pd.read_csv(CSV_PATH, sep=";")
    
    indices = ["TODOS"] + df_tickers["indice"].dropna().unique().tolist()
    indice_escolhido = st.sidebar.selectbox("Selecione o índice:", options=indices, index=0)
    
    if indice_escolhido == "TODOS":
        tickers_filtrados = df_tickers[df_tickers["indice"] != "USA"]["ticker"].dropna().unique().tolist()
    elif indice_escolhido == "USA":
        tickers_filtrados = df_tickers[df_tickers["indice"] == "USA"]["ticker"].dropna().unique().tolist()
    else:
        tickers_filtrados = (
            df_tickers[df_tickers["indice"] == indice_escolhido]["ticker"].dropna().unique().tolist()
        )

    st.sidebar.markdown(f"**Ativos carregados:** {len(tickers_filtrados)}")

    # Botão rodar
    if st.button("🌀 Rodar Varredura Agora"):
        with st.spinner("Executando ciclo do BP-Fênix..."):
            output = run_full_cycle_with_logs(tickers_filtrados, df_tickers)
        st.session_state["fenix_output"] = output
        st.success("Ciclo concluído!")

    output = st.session_state.get("fenix_output", None)
    if not output:
        st.info("◀️ Escolha um índice no sidebar e clique no botão acima para iniciar a varredura.")
        return

    # ------------------------------------------------------------
    # 🔥 TOP ASSETS (painel principal preservado)
    # ------------------------------------------------------------
    st.markdown("## 🔥 Top Selecionados pelo BP-Fênix")

    top_assets = output.get("top_assets", []) or []
    if not top_assets:
        st.warning("Nenhum ativo atingiu o score mínimo.")
        return

    # ------------------------------------------------------------
    # 📋 DataFrame Top 5 (para sidebar)
    # ------------------------------------------------------------
    rows_df = []
    for asset in top_assets:
        ticker = asset["ticker"]
        fs_value = float(asset.get("fs", 0) or 0)

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

        rows_df.append(
            {
                "Selecionar": False,
                "Ticker": ticker,
                "Índice": indice_ticker,
                "Score": round(float(asset.get("score", 0) or 0), 2),
                "FS": round(fs_value, 3),
                "Operação": "COMPRA" if trade["operacao"] == "LONG" else "VENDA",
                "Entrada": round(float(trade["entrada"]), 2),
                "Stop": round(float(trade["stop"]), 2),
                "Alvo": round(float(trade["alvo"]), 2),
                "R/R": round(float(trade["rr"]), 2),
                "ATR Stop": round(float(trade["stop_dist_atr"]), 2),
                "ATR Alvo": round(float(trade["target_dist_atr"]), 2),
            }
        )

    if rows_df:
        df_top5 = pd.DataFrame(rows_df)
        sidebar_report_section(df_top5)

    # ------------------------------------------------------------
    # 🧠 Detalhes por ativo (preservado)
    # ------------------------------------------------------------
    for asset in top_assets:
        ticker = asset["ticker"]

        nome_empresa = None
        try:
            nome_empresa = df_tickers[df_tickers["ticker"] == ticker]["nome"].values[0]
        except Exception:
            nome_empresa = ticker

        fs_value = float(asset.get("fs", 0) or 0)

        try:
            indice_ticker = df_tickers[df_tickers["ticker"] == ticker]["indice"].values[0]
        except Exception:
            indice_ticker = "DESCONHECIDO"

        st.markdown(f"### ⭐ {ticker} ({nome_empresa}) — Score Fênix: **{fs_value:.2f} / 5.00**")

        trade = None
        try:
            df_original = asset["details"]["df"]
            trade = generate_trade_setup(df_original, fs_value)
        except Exception as e:
            st.error(f"Erro ao gerar setup de trade: {e}")

        show_asset_details(
            ticker=ticker,
            details=asset["details"],
            fs_components={
                "tendencia_norm": asset.get("tendencia_norm"),
                "momentum_norm": asset.get("momentum_norm"),
                "volatilidade_norm": asset.get("volatilidade_norm"),
                "volume_norm": asset.get("volume_norm"),
                "sinal_norm": asset.get("sinal_norm"),
                "fs": asset.get("fs"),
            },
            trade=trade,
            indice_ticker=indice_ticker,
        )

        st.markdown("---")

    # ------------------------------------------------------------
    # TABELA COMPLETA + DEBUG (preservado)
    # ------------------------------------------------------------
    with st.expander("📦 Ver Tabela Completa da Varredura"):
        show_results_table(output["raw_results"])

    with st.expander("🐞 Diagnóstico Técnico — Ver detalhes de cada ativo"):
        for ticker, data in output["raw_results"].items():
            st.markdown(f"### 🔍 {ticker}")
            st.write("Score:", data.get("score"))
            st.write("Passaram:", data.get("passed", []))
            st.write("Falharam:", data.get("failed", []))
            st.json(data.get("details", {}))


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    render_dashboard()
