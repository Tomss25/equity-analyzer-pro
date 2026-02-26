"""
logic/scoring_engine.py
Scoring composito 4 dimensioni con sigmoid continua.
Blending 60% settore / 40% regione geografica (v8 feature).
"""

from __future__ import annotations

import numpy as np
from .config import (
    CONFIG, THEMES, SECTOR_BENCHMARKS, REGION_BENCHMARKS,
    sanitize_fundamentals
)

_FLOOR = CONFIG["SCORE_FLOOR"]
_CEIL  = CONFIG["SCORE_CEILING"]

# Pesi interni per dimensione
_FW = {"pe": 0.25, "fwd_pe": 0.10, "pb": 0.10, "ev_ebitda": 0.20,
       "peg": 0.15, "fcf_yield": 0.20}
_MW = {"r1m": 0.10, "r3m": 0.15, "r6m": 0.20, "r12m": 0.20,
       "rsi": 0.10, "ma": 0.10, "sharpe": 0.10, "vol": 0.05}
_QW = {"margin": 0.15, "op_margin": 0.15, "roe": 0.20, "leverage": 0.15,
       "liquidity": 0.10, "rev_growth": 0.15, "eps_growth": 0.10}


def _sigmoid(z: float) -> float:
    z = max(-6.0, min(6.0, z))
    return 1.0 / (1.0 + np.exp(-z))


def _score(value, benchmark, scale, lower_is_better: bool = True) -> float:
    if value is None or benchmark is None or scale <= 0:
        return 50.0
    try:
        v, b = float(value), float(benchmark)
        if not np.isfinite(v):
            return 50.0
        z = (b - v) / scale if lower_is_better else (v - b) / scale
        raw = _sigmoid(z)
        return _FLOOR + (_CEIL - _FLOOR) * raw
    except Exception:
        return 50.0


def _weighted_avg(scores: list, weights: list) -> float:
    if not scores:
        return 50.0
    s = sum(sc * w for sc, w in zip(scores, weights))
    w_tot = sum(weights)
    return s / w_tot if w_tot > 0 else 50.0


class ScoringEngine:

    def calculate(
        self,
        stock_data: dict,
        weights: dict | None = None,
        theme: str = "Balanced",
    ) -> dict:
        sector = stock_data.get("sector", "default") or "default"
        region = (stock_data.get("region") or "default").strip() or "default"

        # Blending 60% settore / 40% regione
        s_bm = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["default"])
        r_bm = REGION_BENCHMARKS.get(region, REGION_BENCHMARKS["default"])
        bm: dict = {}
        for k in ["pe", "pb", "ev_ebitda", "roe_min", "margin_min"]:
            sv = s_bm.get(k)
            rv = r_bm.get(k)
            if sv is not None and rv is not None:
                bm[k] = 0.6 * sv + 0.4 * rv
            elif sv is not None:
                bm[k] = sv
            else:
                bm[k] = rv or SECTOR_BENCHMARKS["default"][k]

        raw_f = stock_data.get("fundamentals", {}) or {}
        f = sanitize_fundamentals(raw_f)

        fund  = self._fundamental(f, bm)
        mom   = self._momentum(stock_data)
        val   = self._valuation(stock_data)
        qual  = self._quality(f, stock_data.get("technicals", {}))

        if not weights:
            t = THEMES.get(theme, THEMES["Balanced"])
            weights = t["weights"]

        composite = (
            fund * weights.get("fundamental", 0.30) +
            mom  * weights.get("momentum",    0.25) +
            val  * weights.get("dcf",         0.25) +
            qual * weights.get("quality",      0.20)
        )

        return {
            "fundamental": round(fund,      1),
            "momentum":    round(mom,       1),
            "valuation":   round(val,       1),
            "quality":     round(qual,      1),
            "composite":   round(composite, 1),
            "theme":       theme,
            "outlier_flags": f.get("_outlier_flags", []),
        }

    # ------------------------------------------------------------------ #
    # DIMENSIONI
    # ------------------------------------------------------------------ #

    def _fundamental(self, f: dict, bm: dict) -> float:
        scores, weights = [], []

        pe = f.get("pe_ratio")
        if pe is not None:
            s = 12.0 if pe < 0 else _score(pe, bm["pe"], bm["pe"] * 0.4)
            scores.append(s); weights.append(_FW["pe"])

        fwd = f.get("forward_pe")
        if fwd and fwd > 0:
            scores.append(_score(fwd, bm["pe"] * 0.9, bm["pe"] * 0.35))
            weights.append(_FW["fwd_pe"])

        pb = f.get("pb_ratio")
        if pb and pb > 0:
            scores.append(_score(pb, bm["pb"], bm["pb"] * 0.5))
            weights.append(_FW["pb"])

        ev = f.get("ev_ebitda")
        if ev and ev > 0:
            scores.append(_score(ev, bm["ev_ebitda"], bm["ev_ebitda"] * 0.4))
            weights.append(_FW["ev_ebitda"])

        peg = f.get("peg_ratio")
        if peg and peg > 0:
            scores.append(_score(peg, 1.0, 0.6))
            weights.append(_FW["peg"])

        fcf = f.get("fcf_yield")
        if fcf is not None:
            scores.append(_score(fcf, 5.0, 4.0, lower_is_better=False))
            weights.append(_FW["fcf_yield"])

        return _weighted_avg(scores, weights)

    def _momentum(self, stock_data: dict) -> float:
        t = stock_data.get("technicals", {}) or {}
        scores, weights = [], []

        for field, w, lo, hi in [
            ("r1m",  _MW["r1m"],  -5,  10),
            ("r3m",  _MW["r3m"],  -10, 20),
            ("r6m",  _MW["r6m"],  -15, 30),
            ("r12m", _MW["r12m"], -20, 50),
        ]:
            v = t.get(field)
            if v is not None:
                scale = (hi - lo) / 4.0
                scores.append(_score(v, 5.0, scale, lower_is_better=False))
                weights.append(w)

        rsi = t.get("rsi")
        if rsi is not None:
            rsi = float(rsi)
            if rsi < 30:
                scores.append(80.0)
            elif rsi > 70:
                scores.append(25.0)
            else:
                scores.append(50.0 + (rsi - 50) * 0.5)
            weights.append(_MW["rsi"])

        ma_cross = t.get("ma_cross")
        if ma_cross is not None:
            scores.append(70.0 if ma_cross else 30.0)
            weights.append(_MW["ma"])

        sharpe = t.get("sharpe_1y")
        if sharpe is not None:
            scores.append(_score(sharpe, 0.5, 0.5, lower_is_better=False))
            weights.append(_MW["sharpe"])

        vol = t.get("volatility")
        if vol is not None:
            scores.append(_score(vol, 25.0, 15.0, lower_is_better=True))
            weights.append(_MW["vol"])

        return _weighted_avg(scores, weights)

    def _valuation(self, stock_data: dict) -> float:
        """DCF implied upside come proxy del dimension 'dcf'."""
        dcf = stock_data.get("_dcf_result")
        if dcf and dcf.get("status") == "ok":
            upside = float(dcf.get("implied_upside_pct", 0))
            prob   = float(dcf.get("prob_upside", 50))
            s_up   = _score(upside, 15.0, 20.0, lower_is_better=False)
            s_prob = _score(prob,   60.0, 20.0, lower_is_better=False)
            return 0.6 * s_up + 0.4 * s_prob

        # Fallback: FCF yield + EV/EBITDA
        f   = sanitize_fundamentals(stock_data.get("fundamentals", {}) or {})
        fcf = f.get("fcf_yield")
        ev  = f.get("ev_ebitda")
        scores, weights = [], []
        if fcf is not None:
            scores.append(_score(fcf, 5.0, 4.0, lower_is_better=False)); weights.append(0.6)
        if ev and ev > 0:
            scores.append(_score(ev, 14.0, 6.0, lower_is_better=True)); weights.append(0.4)
        return _weighted_avg(scores, weights)

    def _quality(self, f: dict, t: dict) -> float:
        scores, weights = [], []

        pm = f.get("profit_margin")
        if pm is not None:
            scores.append(_score(pm, 10.0, 8.0, lower_is_better=False)); weights.append(_QW["margin"])

        om = f.get("operating_margin")
        if om is not None:
            scores.append(_score(om, 12.0, 8.0, lower_is_better=False)); weights.append(_QW["op_margin"])

        roe = f.get("roe")
        if roe is not None:
            scores.append(_score(roe, 12.0, 8.0, lower_is_better=False)); weights.append(_QW["roe"])

        de = f.get("debt_equity")
        if de is not None:
            scores.append(_score(de, 100.0, 80.0, lower_is_better=True)); weights.append(_QW["leverage"])

        cr = f.get("current_ratio")
        if cr is not None:
            scores.append(_score(cr, 1.5, 0.8, lower_is_better=False)); weights.append(_QW["liquidity"])

        rg = f.get("revenue_growth")
        if rg is not None:
            scores.append(_score(rg, 5.0, 8.0, lower_is_better=False)); weights.append(_QW["rev_growth"])

        eg = f.get("eps_growth")
        if eg is not None:
            scores.append(_score(eg, 8.0, 12.0, lower_is_better=False)); weights.append(_QW["eps_growth"])

        return _weighted_avg(scores, weights)


def signal_label(composite: float) -> tuple[str, str]:
    """Ritorna (label, colore_hex) in base al composite score."""
    if composite >= 75:
        return "STRONG BUY", "#22c55e"
    elif composite >= 62:
        return "BUY", "#86efac"
    elif composite >= 45:
        return "HOLD", "#facc15"
    elif composite >= 32:
        return "SELL", "#f97316"
    else:
        return "STRONG SELL", "#ef4444"
