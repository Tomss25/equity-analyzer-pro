"""
logic/monte_carlo.py
Simulazione prezzi con Merton Jump-Diffusion (fallback GBM).
Drift prevalentemente risk-neutral, vol EWMA.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .config import CONFIG, clamp


class MonteCarloEngine:
    """Genera distribuzioni di prezzo forward (1 anno) per singolo titolo."""

    N_SIM     = 10_000
    HORIZON   = 252        # giorni trading
    # Merton JD: parametri conservativi
    JD_LAMBDA = 1.0        # jumps/anno
    JD_MU     = -0.05      # media jump log-size
    JD_SIGMA  = 0.10       # dispersione jump

    def simulate(
        self,
        stock_data: dict,
        n_sim: int | None = None,
        horizon_days: int | None = None,
        model: str = "jump_diffusion",
    ) -> dict:
        current_price = float(stock_data.get("current_price") or 0)
        if current_price <= 0:
            return self._empty()

        daily_returns = stock_data.get("daily_returns")
        if not daily_returns or len(daily_returns) < 60:
            return self._empty()

        n = int(n_sim or self.N_SIM)
        T = int(horizon_days or self.HORIZON)

        if CONFIG.get("USE_FIXED_SEED"):
            rng = np.random.default_rng(int(CONFIG.get("SEED", 42)))
        else:
            rng = np.random.default_rng()

        rets = np.asarray(daily_returns, dtype=float)
        rets = rets[np.isfinite(rets)]
        if rets.size < 60:
            return self._empty()

        # ---- Drift (desk-style) ----
        rf_ann = float(CONFIG.get("RISK_FREE_RATE", 0.045))
        q_ann  = self._div_yield(stock_data)
        mu_rn  = (rf_ann - q_ann) / 252.0
        mu_hist = float(clamp(np.mean(rets), *CONFIG.get("MU_DAILY_CLAMP", (-0.005, 0.005))))
        blend   = float(clamp(CONFIG.get("MC_DRIFT_BLEND", 0.15), 0.0, 0.40))
        mu      = (1.0 - blend) * mu_rn + blend * mu_hist

        # ---- Volatilità EWMA + stdev blend ----
        lam      = float(clamp(CONFIG.get("MC_EWMA_LAMBDA", 0.94), 0.80, 0.99))
        sig_ewma = self._ewma_vol(rets, lam)
        sig_std  = float(np.std(rets[-252:]))
        w_ewma   = float(CONFIG.get("MC_VOL_BLEND_EWMA", 0.70))
        sigma    = w_ewma * sig_ewma + (1.0 - w_ewma) * sig_std
        sigma    = float(clamp(sigma, *CONFIG.get("SIGMA_DAILY_CLAMP", (0.05/252**0.5, 2.0/252**0.5))))

        # ---- Simulazione ----
        log_prices = np.zeros((n, T + 1))
        log_p0 = np.log(current_price)
        log_prices[:, 0] = log_p0

        # GBM term
        Z = rng.standard_normal((n, T))
        gbm_step = (mu - 0.5 * sigma ** 2) + sigma * Z

        if model == "jump_diffusion":
            lam_d   = self.JD_LAMBDA / 252.0
            n_jumps = rng.poisson(lam_d, (n, T))
            jump_sz = rng.normal(self.JD_MU, self.JD_SIGMA, (n, T))
            jump_contrib = n_jumps * jump_sz
            steps = gbm_step + jump_contrib
        else:
            steps = gbm_step

        log_prices[:, 1:] = log_p0 + np.cumsum(steps, axis=1)
        final_prices = np.exp(log_prices[:, -1])
        final_prices = final_prices[np.isfinite(final_prices) & (final_prices > 0)]

        if final_prices.size < max(100, int(0.1 * n)):
            return self._empty()

        p5, p50, p95 = (float(np.percentile(final_prices, q)) for q in (5, 50, 95))
        prob_up       = float(np.mean(final_prices > current_price) * 100)
        implied_up    = (p50 - current_price) / current_price * 100 if current_price > 0 else 0.0

        # VaR e CVaR (rendimenti simulati)
        sim_rets   = (final_prices / current_price - 1.0) * 100
        var_95     = float(np.percentile(sim_rets, 5))
        cvar_95    = float(np.mean(sim_rets[sim_rets <= var_95]))
        ex_kurt    = float(self._excess_kurtosis(sim_rets))

        return {
            "status":            "ok",
            "model":             model,
            "n_simulations":     int(final_prices.size),
            "horizon_days":      T,
            "current_price":     round(current_price, 2),
            "p5":                round(p5,  2),
            "base":              round(p50, 2),
            "p95":               round(p95, 2),
            "prob_upside":       round(prob_up,     1),
            "implied_upside_pct":round(implied_up,  1),
            "var_95_pct":        round(var_95,      2),
            "cvar_95_pct":       round(cvar_95,     2),
            "excess_kurtosis":   round(ex_kurt,     2),
            "sigma_daily":       round(sigma,       5),
            "mu_daily":          round(mu,          6),
        }

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _div_yield(stock_data: dict) -> float:
        dy = stock_data.get("fundamentals", {}).get("dividend_yield")
        if dy is not None:
            try:
                return float(dy) / 100.0
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _ewma_vol(rets: np.ndarray, lam: float) -> float:
        var = float(rets[0] ** 2)
        for r in rets[1:]:
            var = lam * var + (1.0 - lam) * float(r) ** 2
        return float(var ** 0.5)

    @staticmethod
    def _excess_kurtosis(arr: np.ndarray) -> float:
        try:
            mu  = np.mean(arr)
            s   = np.std(arr)
            if s == 0:
                return 0.0
            return float(np.mean(((arr - mu) / s) ** 4) - 3.0)
        except Exception:
            return 0.0

    @staticmethod
    def _empty() -> dict:
        return {
            "status": "na", "model": "none",
            "n_simulations": 0, "horizon_days": 0,
            "current_price": 0, "p5": 0, "base": 0, "p95": 0,
            "prob_upside": 0, "implied_upside_pct": 0,
            "var_95_pct": 0, "cvar_95_pct": 0, "excess_kurtosis": 0,
        }
