"""
logic/dcf_engine.py
DCF Monte Carlo con shock Cholesky-correlati e margin mean-reversion.

CORREZIONI v8-st:
  1. wacc_vol / g_vol / margin_vol ora letti da CONFIG (non ridefiniti localmente).
  2. Matrice correlazione importata da CONFIG (non ridefinita nel loop).
  3. g_terminal_base letto da CONFIG['DCF_G_TERMINAL'].
"""

from __future__ import annotations

import numpy as np

from .config import (
    CONFIG, SECTOR_CAPEX, SECTOR_DA, SECTOR_GROWTH,
    SECTOR_MARGIN_CAP, SECTOR_MARGINS, clamp
)


class DCFEngine:
    """Valutazione intrinseca probabilistica via FCFF discounted by WACC."""

    def __init__(self, seed: int | None = None):
        self.seed = seed

    # ------------------------------------------------------------------ #
    # PUBLIC
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        stock_data: dict,
        n_sim: int | None = None,
        years: int | None = None,
    ) -> dict:
        rng = np.random.default_rng(self.seed)
        n = int(n_sim or CONFIG["DCF_N_SIM"])
        T = int(years or CONFIG["DCF_YEARS"])

        fundamentals      = stock_data.get("fundamentals", {}) or {}
        intrinsic_inputs  = stock_data.get("intrinsic_inputs", {}) or {}
        current_price     = float(stock_data.get("current_price") or 0)
        sector            = stock_data.get("sector", "default") or "default"

        # ---- Inputs di base ----
        revenue = intrinsic_inputs.get("revenue_ttm")
        ebit    = intrinsic_inputs.get("ebit_ttm")
        shares  = intrinsic_inputs.get("shares_outstanding")
        net_debt = intrinsic_inputs.get("net_debt")

        if not revenue or revenue <= 0 or not shares or shares <= 0:
            return self._empty("missing_revenue_or_shares")

        # EBIT fallback da operating_margin
        if ebit is None or not np.isfinite(float(ebit)):
            op_margin = fundamentals.get("operating_margin")
            if op_margin is not None and np.isfinite(float(op_margin)):
                ebit = float(revenue) * (float(op_margin) / 100.0)
            else:
                return self._empty("missing_ebit")

        revenue, ebit = float(revenue), float(ebit)

        # Tax rate
        tax_rate = intrinsic_inputs.get("tax_rate")
        tax_rate = float(clamp(float(tax_rate) if tax_rate is not None else 0.25, 0.05, 0.45))

        # CapEx / D&A / ΔWC come ratio su revenue
        cap_lo, cap_hi, cap_def = SECTOR_CAPEX.get(sector, SECTOR_CAPEX["default"])
        da_lo,  da_hi,  da_def  = SECTOR_DA.get(sector,   SECTOR_DA["default"])

        def ratio(val, rev, lo, hi, default):
            if val is not None and np.isfinite(float(val)):
                return float(clamp(abs(float(val)) / rev, lo, hi))
            return default

        capex_r  = ratio(intrinsic_inputs.get("capex"),    revenue, cap_lo, cap_hi, cap_def)
        da_r     = ratio(intrinsic_inputs.get("da"),       revenue, da_lo,  da_hi,  da_def)
        dwc_r    = float(clamp((intrinsic_inputs.get("delta_wc") or 0) / revenue, -0.02, 0.05))

        # Crescita base
        rev_growth_pct = fundamentals.get("revenue_growth")
        g_lo, g_hi = SECTOR_GROWTH.get(sector, SECTOR_GROWTH["default"])
        base_g = float(clamp(
            float(rev_growth_pct) / 100.0 if rev_growth_pct is not None else 0.04,
            g_lo, g_hi
        ))

        # Margine base
        m_cap = SECTOR_MARGIN_CAP.get(sector, 0.25)
        base_margin = float(clamp(ebit / revenue, -0.10, m_cap))

        # Margin mean-reversion target
        sector_avg_margin = SECTOR_MARGINS.get(sector, SECTOR_MARGINS["default"])
        if base_margin > sector_avg_margin:
            mr_target = sector_avg_margin + (base_margin - sector_avg_margin) * 0.4
        else:
            mr_target = sector_avg_margin + (base_margin - sector_avg_margin) * 0.6
        mr_target = float(clamp(mr_target, -0.05, 0.35))

        # WACC
        beta = float(clamp(
            float(stock_data.get("technicals", {}).get("beta") or 1.0),
            CONFIG["MIN_BETA"], CONFIG["MAX_BETA"]
        ))
        rf  = float(CONFIG["RISK_FREE_RATE"])
        erp = float(CONFIG["EQUITY_RISK_PREMIUM"])
        cost_equity = rf + beta * erp

        total_debt   = intrinsic_inputs.get("total_debt")
        cash         = intrinsic_inputs.get("cash")
        if net_debt is None and total_debt is not None and cash is not None:
            net_debt = float(total_debt) - float(cash)
        net_debt = float(net_debt or 0.0)

        market_cap = stock_data.get("market_cap")
        if not market_cap or market_cap <= 0:
            market_cap = current_price * shares if current_price > 0 else None
        if not market_cap or market_cap <= 0:
            return self._empty("missing_market_cap")

        ev = float(market_cap) + max(0.0, net_debt)
        wd = float(clamp(max(0.0, net_debt) / ev if ev > 0 else 0.0, 0.0, CONFIG["MAX_DEBT_WEIGHT"]))
        we = 1.0 - wd
        leverage = max(0.0, net_debt) / float(market_cap) if float(market_cap) > 0 else 0.0
        spread = 0.015 + 0.03 * float(clamp(leverage, 0.0, 2.0))
        cost_debt = float(clamp(rf + spread, 0.01, 0.18))
        wacc_base = float(clamp(we * cost_equity + wd * cost_debt * (1.0 - tax_rate),
                                CONFIG["MIN_WACC"], CONFIG["MAX_WACC"]))

        # CORREZIONE 1: vol da CONFIG
        wacc_vol   = float(CONFIG["WACC_VOL"])
        g_vol      = float(CONFIG["GROWTH_VOL"])
        margin_vol = float(CONFIG["MARGIN_VOL"])

        # CORREZIONE 2: matrice correlazione da CONFIG
        L = np.linalg.cholesky(CONFIG["CORRELATION_MATRIX"])

        # CORREZIONE 3: g_terminal da CONFIG
        g_terminal_base = float(CONFIG["DCF_G_TERMINAL"])

        # Shocks correlati
        Z = rng.standard_normal((3, n))
        X = L @ Z
        wacc_sh, g_sh, margin_sh = X[0], X[1], X[2]

        fv_per_share  = np.zeros(n, dtype=float)
        fv_enterprise = np.zeros(n, dtype=float)

        for i in range(n):
            wacc_i = float(clamp(wacc_base * (1.0 + wacc_vol * wacc_sh[i]),
                                 CONFIG["MIN_WACC"], CONFIG["MAX_WACC"]))
            g_i0 = float(clamp(base_g * (1.0 + g_vol * g_sh[i]),
                               g_lo - 0.05, g_hi + 0.05))
            m_i0 = float(clamp(base_margin * (1.0 + margin_vol * margin_sh[i]),
                               -0.15, min(0.45, m_cap + 0.08)))

            g_long = g_terminal_base
            m_long = float(clamp(mr_target, -0.05, 0.35))
            rev_t  = revenue
            pv     = 0.0
            fcff_t = 0.0

            for t in range(1, T + 1):
                decay = np.exp(-0.35 * (t - 1))
                g_t   = g_long + (g_i0 - g_long) * decay
                m_t   = m_long + (m_i0 - m_long) * decay
                rev_t *= (1.0 + g_t)
                nopat = rev_t * m_t * (1.0 - tax_rate)
                fcff_t = nopat + rev_t * da_r - rev_t * capex_r - rev_t * dwc_r
                pv    += fcff_t / ((1.0 + wacc_i) ** t)

            # Terminal value
            g_term = float(clamp(g_terminal_base + 0.008 * g_sh[i], -0.01, 0.035))
            g_term = min(g_term, wacc_i - CONFIG["MIN_TV_SPREAD"])
            denom  = max(CONFIG["MIN_TV_SPREAD"], wacc_i - g_term)
            tv     = (fcff_t * (1.0 + g_term)) / denom
            pv_tv  = tv / ((1.0 + wacc_i) ** T)

            ev_i = pv + pv_tv
            eq_i = ev_i - net_debt
            fv_enterprise[i] = ev_i
            fv_per_share[i]  = eq_i / float(shares)

        fv_per_share = fv_per_share[np.isfinite(fv_per_share)]
        if fv_per_share.size < max(200, int(0.2 * n)):
            return self._empty("insufficient_valid_sims")

        p5, p50, p95 = (float(np.percentile(fv_per_share, q)) for q in (5, 50, 95))
        prob_upside   = float(np.mean(fv_per_share > current_price) * 100) if current_price > 0 else 0.0
        implied_upside = ((p50 - current_price) / current_price * 100) if current_price > 0 else 0.0

        return {
            "status":           "ok",
            "n_simulations":    int(fv_per_share.size),
            "years":            T,
            "fv_p5":            round(p5,  2),
            "fv_median":        round(p50, 2),
            "fv_p95":           round(p95, 2),
            "prob_upside":      round(prob_upside,    1),
            "implied_upside_pct": round(implied_upside, 1),
            "assumptions": {
                "wacc_base_pct":           round(wacc_base * 100, 2),
                "beta":                    round(beta, 2),
                "rf_pct":                  round(rf  * 100, 2),
                "erp_pct":                 round(erp * 100, 2),
                "tax_rate_pct":            round(tax_rate * 100, 1),
                "base_growth_pct":         round(base_g * 100, 1),
                "base_margin_pct":         round(base_margin * 100, 1),
                "capex_ratio_pct":         round(capex_r * 100, 2),
                "da_ratio_pct":            round(da_r  * 100, 2),
                "dwc_ratio_pct":           round(dwc_r * 100, 2),
                "net_debt":                float(net_debt),
                "debt_weight_pct":         round(wd * 100, 1),
                "sector_avg_margin_pct":   round(sector_avg_margin * 100, 1),
                "margin_rev_target_pct":   round(mr_target * 100, 1),
                "g_terminal_pct":          round(g_terminal_base * 100, 2),
            },
            "hist": self._histogram(fv_per_share, bins=40),
        }

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _histogram(arr: np.ndarray, bins: int = 40) -> dict:
        counts, edges = np.histogram(arr, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return {"bins": centers.tolist(), "counts": counts.tolist()}

    @staticmethod
    def _empty(reason: str = "unavailable") -> dict:
        return {
            "status": "na", "reason": reason,
            "n_simulations": 0, "years": 0,
            "fv_p5": 0, "fv_median": 0, "fv_p95": 0,
            "prob_upside": 0, "implied_upside_pct": 0,
            "assumptions": {}, "hist": {"bins": [], "counts": []},
        }
