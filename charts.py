"""
ui/pages/single_stock.py
Pagina analisi singolo titolo: scoring, DCF, MC, stress test.
"""

from __future__ import annotations

import streamlit as st
import numpy as np

from logic.data_fetcher import DataFetcher
from logic.scoring_engine import ScoringEngine
from logic.dcf_engine import DCFEngine
from logic.monte_carlo import MonteCarloEngine
from logic.stress_test import StressTestEngine
from logic.config import THEMES

from ui.components import (
    section_header, score_badge, metric_card, kv_table,
    assumptions_expander, mc_metrics_row, dcf_metrics_row, outlier_warning,
    sidebar_theme_picker, sidebar_compute_preset, sidebar_api_keys,
)
from ui.charts import (
    price_history_chart, dcf_histogram, mc_fan_chart,
    score_radar, stress_bar_chart,
)

_fetcher  = DataFetcher()
_scorer   = ScoringEngine()
_dcf      = DCFEngine()
_mc       = MonteCarloEngine()
_stress   = StressTestEngine()


def render():
    st.title("📊 Analisi Singolo Titolo")

    # ---- Sidebar ----
    theme, weights = sidebar_theme_picker(THEMES)
    preset = sidebar_compute_preset()
    api_keys = sidebar_api_keys()

    # ---- Input ticker ----
    ticker_raw = st.text_input(
        "Inserisci Ticker (es. AAPL, MSFT, ENI.MI)",
        value="AAPL",
        placeholder="Ticker Yahoo Finance",
    ).strip().upper()

    col_analyze, col_refresh = st.columns([3, 1])
    with col_analyze:
        run = st.button("🔍 Analizza", use_container_width=True, type="primary")
    with col_refresh:
        force = st.checkbox("Force refresh", value=False)

    if not run:
        st.info("Inserisci un ticker e premi **Analizza**.")
        return

    if not ticker_raw:
        st.error("Ticker non valido.")
        return

    # ---- Fetch ----
    with st.spinner(f"Caricamento dati {ticker_raw}..."):
        data = _fetcher.fetch(
            ticker_raw,
            force_refresh=force,
            fmp_key=api_keys.get("fmp_key", ""),
            av_key=api_keys.get("av_key", ""),
        )

    if data is None:
        st.error(f"❌ Impossibile recuperare dati per **{ticker_raw}**.")
        st.info("""
**Possibili cause:**
- Ticker non corretto (verifica su [finance.yahoo.com](https://finance.yahoo.com))
- Per titoli italiani usa il suffisso `.MI` → es. `ENI.MI`, `ENEL.MI`
- Per titoli tedeschi usa `.DE` → es. `SAP.DE`
- Yahoo Finance può essere temporaneamente non disponibile

**Soluzioni:**
1. Prova ad aggiungere una FMP API Key nel pannello 🔑 nella sidebar
2. Prova con un ticker diverso (es. `AAPL`, `MSFT`)
3. Riprova tra qualche secondo (problema di rete temporaneo)
        """)
        return

    cur_price = float(data.get("current_price") or 0)
    name      = data.get("name", ticker_raw)
    sector    = data.get("sector", "N/A")
    region    = data.get("region", "N/A")

    # ---- Header ----
    st.markdown(f"## {name} `{ticker_raw}`")
    st.caption(f"Settore: {sector} | Regione: {region} | "
               f"Market Cap: ${(data.get('market_cap') or 0)/1e9:.2f}B | "
               f"Prezzo: ${cur_price:.2f}")

    # ---- DCF (serve prima per valuation score) ----
    with st.spinner("DCF Monte Carlo..."):
        dcf_res = _dcf.simulate(data, n_sim=preset["dcf_n"], years=preset["dcf_years"])
    data["_dcf_result"] = dcf_res  # iniettato per scoring

    # ---- Scoring ----
    score_res = _scorer.calculate(data, weights=weights, theme=theme)
    outlier_warning(score_res.get("outlier_flags", []))

    # ---- MC ----
    with st.spinner("Monte Carlo prezzi..."):
        mc_res = _mc.simulate(data, n_sim=preset["mc_n"])

    # ================================================================
    # SEZIONE 1: Score + Metriche chiave
    # ================================================================
    section_header("Valutazione Composita", "🏆")
    col_score, col_fund = st.columns([1, 3])
    with col_score:
        score_badge(score_res["composite"])
    with col_fund:
        f = data.get("fundamentals", {}) or {}
        kv_table([
            ("P/E",             _fmt(f.get("pe_ratio"),      ".1f")),
            ("Forward P/E",     _fmt(f.get("forward_pe"),    ".1f")),
            ("P/B",             _fmt(f.get("pb_ratio"),      ".2f")),
            ("EV/EBITDA",       _fmt(f.get("ev_ebitda"),     ".1f")),
            ("ROE",             _fmt(f.get("roe"),           ".1f", "%")),
            ("Profit Margin",   _fmt(f.get("profit_margin"), ".1f", "%")),
            ("Debt/Equity",     _fmt(f.get("debt_equity"),   ".1f")),
            ("FCF Yield",       _fmt(f.get("fcf_yield"),     ".1f", "%")),
            ("Rev Growth",      _fmt(f.get("revenue_growth"),".1f", "%")),
            ("Beta",            _fmt(f.get("beta"),          ".2f")),
        ])

    st.plotly_chart(score_radar(score_res, ticker_raw), use_container_width=True)

    # ================================================================
    # SEZIONE 2: Storico prezzi
    # ================================================================
    section_header("Storico Prezzi", "📈")
    t = data.get("technicals", {}) or {}
    fig_hist = price_history_chart(
        data.get("price_dates", []),
        data.get("price_history", []),
        ticker_raw,
        ma50=t.get("ma50"),
        ma200=t.get("ma200"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Momentum row
    section_header("Momentum", "⚡")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, key in [
        (c1, "1M", "r1m"), (c2, "3M", "r3m"),
        (c3, "6M", "r6m"), (c4, "12M", "r12m"),
    ]:
        v = t.get(key)
        with col:
            metric_card(label, _fmt(v, ".1f", "%") if v else "N/A",
                        delta_positive=(v or 0) >= 0)
    with c5:
        metric_card("RSI 14", _fmt(t.get("rsi"), ".1f") if t.get("rsi") else "N/A")

    # ================================================================
    # SEZIONE 3: DCF
    # ================================================================
    section_header("DCF Monte Carlo", "🎰")
    if dcf_res.get("status") == "ok":
        dcf_metrics_row(dcf_res, cur_price)
        st.plotly_chart(dcf_histogram(dcf_res, cur_price, ticker_raw), use_container_width=True)
        assumptions_expander(dcf_res.get("assumptions", {}))
    else:
        st.warning(f"DCF non disponibile: {dcf_res.get('reason', 'dati mancanti')}")

    # ================================================================
    # SEZIONE 4: Monte Carlo prezzi
    # ================================================================
    section_header("Monte Carlo Price Target (1 anno)", "🎯")
    if mc_res.get("status") == "ok":
        mc_metrics_row(mc_res)
        st.plotly_chart(mc_fan_chart(mc_res, cur_price, ticker_raw), use_container_width=True)
        st.caption(f"Modello: {mc_res.get('model')} | "
                   f"σ giorn.: {mc_res.get('sigma_daily', 0)*100:.3f}% | "
                   f"Excess Kurtosis: {mc_res.get('excess_kurtosis', 0):.2f}")
    else:
        st.warning("Monte Carlo non disponibile (dati storici insufficienti).")

    # ================================================================
    # SEZIONE 5: Stress Test
    # ================================================================
    section_header("Stress Test Macro", "🔬")
    with st.spinner("Stress test..."):
        stress_res = _stress.run_all(data)
    st.plotly_chart(stress_bar_chart(stress_res, ticker_raw), use_container_width=True)

    with st.expander("📋 Dettaglio scenari", expanded=False):
        for key, sr in stress_res.items():
            if "error" in sr:
                continue
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.write(f"{sr['icon']} {sr['scenario']}")
            c2.write(f"Drawdown: **{sr['drawdown_pct']:.1f}%**")
            c3.write(f"Prezzo stress: ${sr['stressed_price']:.2f}")
            c4.write(f"Recovery: {sr['recovery_months']}m")


# ------------------------------------------------------------------ #
# HELPERS
# ------------------------------------------------------------------ #

def _fmt(val, fmt: str = ".2f", suffix: str = "", prefix: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        return f"{prefix}{float(val):{fmt}}{suffix}"
    except Exception:
        return "N/A"
