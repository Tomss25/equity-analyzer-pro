"""
logic/portfolio_optimizer.py
Markowitz MVO + Black-Litterman + Fama-French Factor Decomposition.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import List, Optional

from .config import CONFIG


# ---------------------------------------------------------------------------
# LEDOIT-WOLF SHRINKAGE
# ---------------------------------------------------------------------------

def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Stima covarianza con shrinkage Ledoit-Wolf (target: scaled identity)."""
    n, p = returns.shape
    if p == 1:
        return np.cov(returns.T).reshape(1, 1)

    sample_cov = np.cov(returns.T)
    mu_trace   = np.trace(sample_cov) / p
    delta_sq   = np.linalg.norm(sample_cov - mu_trace * np.eye(p), "fro") ** 2

    beta_bar = 0.0
    for i in range(n):
        xi = returns[i].reshape(-1, 1)
        beta_bar += np.linalg.norm(xi @ xi.T - sample_cov, "fro") ** 2
    beta_bar /= n ** 2

    beta  = min(beta_bar / (delta_sq + 1e-12), 1.0)
    alpha = 1.0 - beta
    return alpha * sample_cov + beta * mu_trace * np.eye(p)


# ---------------------------------------------------------------------------
# MARKOWITZ
# ---------------------------------------------------------------------------

class MarkowitzOptimizer:

    def __init__(self, risk_free_rate: float = 0.045):
        self.rf = risk_free_rate

    def optimize(
        self,
        tickers: List[str],
        returns_matrix: np.ndarray,    # shape (T, N) daily returns
        objective: str = "max_sharpe",
        max_weight: float = 0.40,
        min_weight: float = 0.0,
    ) -> dict:
        n = len(tickers)
        if n < 2 or returns_matrix.shape[1] != n:
            return {"error": "Dati insufficienti per ottimizzazione"}

        mu  = np.mean(returns_matrix, axis=0) * 252
        cov = ledoit_wolf_cov(returns_matrix) * 252

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(min_weight, max_weight)] * n
        w0 = np.ones(n) / n

        if objective == "max_sharpe":
            def neg_sharpe(w):
                r   = w @ mu
                vol = np.sqrt(w @ cov @ w)
                return -(r - self.rf) / (vol + 1e-9)
            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-9, "maxiter": 500})
        elif objective == "min_vol":
            def portfolio_vol(w):
                return np.sqrt(w @ cov @ w)
            res = minimize(portfolio_vol, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-9, "maxiter": 500})
        else:
            return {"error": f"Obiettivo '{objective}' non supportato"}

        if not res.success:
            return {"error": f"Ottimizzazione non convergita: {res.message}"}

        w = np.maximum(res.x, 0)
        w /= w.sum()

        p_ret = float(w @ mu)
        p_vol = float(np.sqrt(w @ cov @ w))
        p_sharpe = (p_ret - self.rf) / (p_vol + 1e-9)

        return {
            "status":    "ok",
            "objective": objective,
            "weights":   {t: round(float(wi), 4) for t, wi in zip(tickers, w)},
            "portfolio_return_pct":  round(p_ret * 100, 2),
            "portfolio_vol_pct":     round(p_vol * 100, 2),
            "portfolio_sharpe":      round(p_sharpe, 3),
        }

    def efficient_frontier(
        self,
        tickers: List[str],
        returns_matrix: np.ndarray,
        n_points: int = 50,
        max_weight: float = 0.40,
    ) -> dict:
        """Genera N punti sulla frontiera efficiente."""
        n = len(tickers)
        if n < 2:
            return {"error": "Almeno 2 titoli richiesti"}

        mu  = np.mean(returns_matrix, axis=0) * 252
        cov = ledoit_wolf_cov(returns_matrix) * 252
        bounds = [(0.0, max_weight)] * n
        w0 = np.ones(n) / n

        # Range target return
        r_min = float(np.min(mu))
        r_max = float(np.max(mu))
        targets = np.linspace(r_min, r_max, n_points)

        frontier_vols, frontier_rets = [], []
        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
            ]
            res = minimize(
                lambda w: np.sqrt(w @ cov @ w), w0,
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 300},
            )
            if res.success:
                w = np.maximum(res.x, 0); w /= w.sum()
                frontier_vols.append(float(np.sqrt(w @ cov @ w)) * 100)
                frontier_rets.append(float(w @ mu) * 100)

        return {
            "frontier_vols": frontier_vols,
            "frontier_rets": frontier_rets,
        }


# ---------------------------------------------------------------------------
# BLACK-LITTERMAN
# ---------------------------------------------------------------------------

class BlackLittermanOptimizer:

    def __init__(self, tau: float = 0.05, risk_free_rate: float = 0.045):
        self.tau = tau
        self.rf  = risk_free_rate

    def optimize(
        self,
        tickers: List[str],
        returns_matrix: np.ndarray,
        views: Optional[List[dict]] = None,
        market_weights: Optional[np.ndarray] = None,
    ) -> dict:
        """
        views: lista di {"assets": ["AAPL", "MSFT"], "returns": [0.12, -0.05]}
               oppure {"assets": ["AAPL"], "returns": [0.10]}
        """
        n = len(tickers)
        mu_hist = np.mean(returns_matrix, axis=0) * 252
        cov     = ledoit_wolf_cov(returns_matrix) * 252

        # Prior: CAPM equilibrium
        mw = market_weights if market_weights is not None else np.ones(n) / n
        lam = (float(mu_hist @ mw) - self.rf) / float(mw @ cov @ mw + 1e-9)
        lam = max(1.0, min(lam, 5.0))
        pi  = lam * cov @ mw   # implied equilibrium returns

        if not views:
            # Nessuna view: ritorna prior
            posterior_mu = pi
        else:
            # Costruisci P, Q, Omega
            P_rows, Q_vals = [], []
            for v in views:
                assets  = v.get("assets", [])
                rets    = v.get("returns", [])
                if len(assets) != len(rets):
                    continue
                row = np.zeros(n)
                for asset, ret in zip(assets, rets):
                    if asset in tickers:
                        idx = tickers.index(asset)
                        row[idx] = ret / (abs(ret) + 1e-9)
                Q_vals.append(np.mean(rets))
                P_rows.append(row)

            if P_rows:
                P = np.array(P_rows)    # (k, n)
                Q = np.array(Q_vals)    # (k,)
                tau_cov = self.tau * cov
                Omega   = np.diag(np.diag(P @ tau_cov @ P.T))  # diagonal uncertainty

                M   = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P)
                posterior_mu = M @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
            else:
                posterior_mu = pi

        # Ottimizza con posterior
        bounds = [(0.0, 0.40)] * n
        w0 = mw.copy()

        def neg_sharpe(w):
            r   = w @ posterior_mu
            vol = np.sqrt(w @ cov @ w + 1e-12)
            return -(r - self.rf) / vol

        res = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                       options={"ftol": 1e-9, "maxiter": 500})

        if not res.success:
            return {"error": f"BL ottimizzazione non convergita: {res.message}"}

        w = np.maximum(res.x, 0); w /= w.sum()
        p_ret = float(w @ posterior_mu)
        p_vol = float(np.sqrt(w @ cov @ w))

        return {
            "status":    "ok",
            "weights":   {t: round(float(wi), 4) for t, wi in zip(tickers, w)},
            "posterior_returns": {t: round(float(r) * 100, 2) for t, r in zip(tickers, posterior_mu)},
            "portfolio_return_pct": round(p_ret * 100, 2),
            "portfolio_vol_pct":    round(p_vol * 100, 2),
            "portfolio_sharpe":     round((p_ret - self.rf) / (p_vol + 1e-9), 3),
        }


# ---------------------------------------------------------------------------
# FACTOR DECOMPOSITION (Fama-French proxy)
# ---------------------------------------------------------------------------

class FactorDecomposer:
    """
    Decomposizione semplificata su fattori proxy:
      MKT  = rendimento portafoglio equi-pesato
      SMB  = avg small cap - avg large cap  (proxy: bottom vs top tercile per market cap)
      HML  = avg value (alto P/B) - avg growth (basso P/B)
      MOM  = avg 12m winner - avg 12m loser
    """

    def decompose(
        self,
        tickers: List[str],
        returns_matrix: np.ndarray,      # (T, N) daily returns
        market_caps: Optional[List[float]] = None,
        pb_ratios:   Optional[List[float]] = None,
        r12m_pcts:   Optional[List[float]] = None,
    ) -> dict:
        T, N = returns_matrix.shape
        if T < 60 or N < 3:
            return {"error": "Dati insufficienti per decomposizione fattoriale"}

        mu = np.mean(returns_matrix, axis=0) * 252

        # MKT factor
        mkt = np.mean(returns_matrix, axis=1)

        # SMB proxy
        if market_caps and len(market_caps) == N:
            mc_arr = np.array(market_caps, dtype=float)
            mc_arr[~np.isfinite(mc_arr)] = np.nanmedian(mc_arr)
            tercile = np.percentile(mc_arr, 33), np.percentile(mc_arr, 67)
            small_idx = mc_arr <= tercile[0]
            large_idx = mc_arr >= tercile[1]
            smb_daily = (np.mean(returns_matrix[:, small_idx], axis=1)
                         - np.mean(returns_matrix[:, large_idx], axis=1)) if (small_idx.any() and large_idx.any()) else np.zeros(T)
        else:
            smb_daily = np.zeros(T)

        # HML proxy
        if pb_ratios and len(pb_ratios) == N:
            pb_arr = np.array(pb_ratios, dtype=float)
            pb_arr[~np.isfinite(pb_arr)] = np.nanmedian(pb_arr)
            med_pb = np.median(pb_arr)
            value_idx  = pb_arr <= med_pb
            growth_idx = pb_arr >  med_pb
            hml_daily = (np.mean(returns_matrix[:, value_idx], axis=1)
                         - np.mean(returns_matrix[:, growth_idx], axis=1)) if (value_idx.any() and growth_idx.any()) else np.zeros(T)
        else:
            hml_daily = np.zeros(T)

        # MOM proxy
        if r12m_pcts and len(r12m_pcts) == N:
            r12_arr = np.array(r12m_pcts, dtype=float)
            r12_arr[~np.isfinite(r12_arr)] = 0.0
            med_r12 = np.median(r12_arr)
            winner_idx = r12_arr >= med_r12
            loser_idx  = r12_arr <  med_r12
            mom_daily = (np.mean(returns_matrix[:, winner_idx], axis=1)
                         - np.mean(returns_matrix[:, loser_idx], axis=1)) if (winner_idx.any() and loser_idx.any()) else np.zeros(T)
        else:
            mom_daily = np.zeros(T)

        # Regressione OLS per ogni titolo
        factor_matrix = np.column_stack([mkt, smb_daily, hml_daily, mom_daily])
        factor_labels  = ["MKT", "SMB", "HML", "MOM"]

        exposures = {}
        for i, ticker in enumerate(tickers):
            y = returns_matrix[:, i]
            X = np.column_stack([np.ones(T), factor_matrix])
            try:
                coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                alpha = float(coeff[0]) * 252
                betas = {lbl: round(float(b), 3) for lbl, b in zip(factor_labels, coeff[1:])}
                exposures[ticker] = {"alpha_annual": round(alpha, 4), "betas": betas}
            except Exception:
                exposures[ticker] = {"alpha_annual": 0.0, "betas": {lbl: 0.0 for lbl in factor_labels}}

        # Factor annualized returns (for display)
        factor_returns = {
            "MKT": round(float(np.mean(mkt) * 252 * 100), 2),
            "SMB": round(float(np.mean(smb_daily) * 252 * 100), 2),
            "HML": round(float(np.mean(hml_daily) * 252 * 100), 2),
            "MOM": round(float(np.mean(mom_daily) * 252 * 100), 2),
        }

        return {
            "status":         "ok",
            "exposures":      exposures,
            "factor_returns": factor_returns,
        }
