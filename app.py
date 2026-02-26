"""
ui/pages/screening.py
Pagina screening universo: analisi parallela, filtro score, ranking.
"""

from __future__ import annotations

import concurrent.futures
import numpy as np
import pandas as pd
import streamlit as st

from logic.data_fetcher import DataFetcher
from logic.scoring_engine import ScoringEngine, signal_label
from logic.dcf_engine import DCFEngine
from logic.config import THEMES

from ui.components import section_header, sidebar_theme_picker, sidebar_compute_preset

_fetcher = DataFetcher()
_scorer  = ScoringEngine()
_dcf     = DCFEngine()

# Universi predefiniti (subset, non caricare 500 titoli live)
QUICK_UNIVERSES = {
    "US Mega Cap":  ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JPM", "V"],
    "US Value":     ["BRK-B", "JPM", "JNJ", "PG", "KO", "XOM", "CVX", "BAC", "WFC", "MRK"],
    "US Tech":      ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "CRM", "NOW", "ADBE", "ORCL"],
    "EU Blue Chip": ["ASML.AS", "SAP.DE", "LVMH.PA", "NESN.SW", "NOVO-B.CO", "TTE.PA", "SIE.DE", "AIR.PA", "MC.PA", "OR.PA"],
    "Custom":       [],
}


def _analyze_ticker(ticker: str, weights: dict, dcf_n: int, dcf_years: int) -> dict | None:
    try:
        data = _fetcher.fetch(ticker)
        if data is None:
            return None
        dcf_res = _dcf.simulate(data, n_sim=dcf_n, years=dcf_years)
        data["_dcf_result"] = dcf_res
        score = _scorer.calculate(data, weights=weights)
        f = data.get("fundamentals", {}) or {}
        return {
            "ticker":       ticker,
            "name":         data.get("name", ticker),
            "sector":       data.get("sector", "N/A"),
            "price":        data.get("current_price"),
            "market_cap_b": (data.get("market_cap") or 0) / 1e9,
            "composite":    score["composite"],
            "fundamental":  score["fundamental"],
            "momentum":     score["momentum"],
            "valuation":    score["valuation"],
            "quality":      score["quality"],
            "signal":       signal_label(score["composite"])[0],
            "pe":           f.get("pe_ratio"),
            "pb":           f.get("pb_ratio"),
            "roe":          f.get("roe"),
            "rev_growth":   f.get("revenue_growth"),
            "dcf_upside":   dcf_res.get("implied_upside_pct") if dcf_res.get("status") == "ok" else None,
            "prob_upside":  dcf_res.get("prob_upside")        if dcf_res.get("status") == "ok" else None,
            "beta":         f.get("beta"),
        }
    except Exception as e:
        return None


def render():
    st.title("🔍 Screening Universo")

    theme, weights = sidebar_theme_picker(THEMES)
    preset = sidebar_compute_preset()

    # ---- Selezione universo ----
    section_header("Configura Screening", "⚙️")
    universe_key = st.selectbox("Universo", list(QUICK_UNIVERSES.keys()))
    tickers = QUICK_UNIVERSES[universe_key].copy()

    if universe_key == "Custom":
        raw = st.text_area("Inserisci ticker (uno per riga o separati da virgola)", height=100)
        tickers = [t.strip().upper() for t in raw.replace(",", "\n").splitlines() if t.strip()]

    if not tickers:
        st.info("Seleziona un universo o inserisci ticker custom.")
        return

    min_score = st.slider("Score minimo per incluso nel risultato", 0, 90, 50, 5)
    max_workers = st.slider("Thread paralleli", 2, 8, 4)

    st.info(f"Universo: **{len(tickers)} titoli** | Tema: **{theme}** | "
            f"Preset: **{preset['dcf_n']} sim DCF**")

    run = st.button("🚀 Avvia Screening", type="primary", use_container_width=True)
    if not run:
        return

    # ---- Analisi parallela ----
    results = []
    progress_bar = st.progress(0, text="Analisi in corso...")
    status_text = st.empty()
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_analyze_ticker, t, weights, preset["dcf_n"], preset["dcf_years"]): t
            for t in tickers
        }
        for future in concurrent.futures.as_completed(futures):
            done += 1
            progress_bar.progress(done / len(tickers), text=f"{done}/{len(tickers)} analizzati")
            res = future.result()
            if res is not None:
                results.append(res)
            status_text.text(f"✓ {futures[future]}")

    progress_bar.empty()
    status_text.empty()

    if not results:
        st.error("Nessun dato recuperato. Controlla la connessione.")
        return

    # ---- Filtra e ordina ----
    df = pd.DataFrame(results)
    df_filtered = df[df["composite"] >= min_score].sort_values("composite", ascending=False).reset_index(drop=True)

    # ---- Risultati ----
    section_header(f"Risultati ({len(df_filtered)} titoli sopra score {min_score})", "📋")

    # Segnali di sintesi
    signal_counts = df_filtered["signal"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, sig, color in [
        (c1, "STRONG BUY", "#22c55e"), (c2, "BUY", "#86efac"),
        (c3, "HOLD", "#facc15"), (c4, "SELL", "#f97316"),
        (c5, "STRONG SELL", "#ef4444"),
    ]:
        with col:
            count = signal_counts.get(sig, 0)
            st.markdown(
                f'<div style="background:#161b22;border:1px solid {color};border-radius:6px;'
                f'padding:8px;text-align:center;color:{color};font-weight:700;">'
                f'{sig}<br><span style="font-size:24px">{count}</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")

    # Tabella interattiva
    display_cols = {
        "ticker": "Ticker", "name": "Nome", "sector": "Settore",
        "price": "Prezzo", "market_cap_b": "MCap ($B)",
        "composite": "Score", "signal": "Segnale",
        "pe": "P/E", "pb": "P/B", "roe": "ROE%",
        "rev_growth": "Rev Growth%", "dcf_upside": "DCF Upside%",
        "prob_upside": "Prob Up%", "beta": "Beta",
    }

    df_display = df_filtered[list(display_cols.keys())].rename(columns=display_cols)

    # Formattazione
    for col in ["Prezzo", "MCap ($B)", "P/E", "P/B", "ROE%", "Rev Growth%",
                "DCF Upside%", "Prob Up%", "Beta"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda v: f"{v:.2f}" if isinstance(v, float) and not pd.isna(v) else (str(v) if v else "N/A")
            )

    st.dataframe(
        df_display,
        use_container_width=True,
        height=min(600, 50 + 35 * len(df_display)),
        hide_index=True,
    )

    # Download
    st.download_button(
        "📥 Scarica CSV",
        df_filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"screening_{universe_key}_{theme}.csv",
        mime="text/csv",
    )

    # Top 5 chart
    if len(df_filtered) >= 2:
        section_header("Top 10 Score Breakdown", "🏆")
        import plotly.graph_objects as go
        top = df_filtered.head(10)
        fig = go.Figure()
        for dim, color in [
            ("fundamental", "#58a6ff"), ("momentum", "#3fb950"),
            ("valuation", "#d29922"), ("quality", "#f0883e"),
        ]:
            fig.add_trace(go.Bar(
                name=dim.capitalize(), x=top["ticker"],
                y=top[dim], marker_color=color,
            ))
        fig.update_layout(
            barmode="stack",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", range=[0, 100]),
            title="Score per dimensione (Top 10)",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
