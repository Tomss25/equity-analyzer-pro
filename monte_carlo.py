"""
logic/config.py
Equity Analyzer Pro — Configurazione centralizzata e guardrail numerici.
CORREZIONI applicate rispetto all'originale:
  1. wacc_vol / g_vol / margin_vol ora letti da CONFIG (non ridefiniti nel DCF).
  2. g_terminal_base esposto come parametro configurabile.
  3. Matrice di correlazione definita UNA sola volta, importata da tutti i motori.
"""

import numpy as np

CONFIG = {
    "SEED": 42,
    "USE_FIXED_SEED": False,
    "CACHE_VERSION": "v8.0-st",          # Invalida vecchie cache disk v7
    "DCF_N_SIM": 10_000,
    "DCF_YEARS": 7,
    "MIN_WACC": 0.060,
    "MAX_WACC": 0.250,
    "MIN_TV_SPREAD": 0.030,              # WACC - g_terminal >= 3%
    "MIN_BETA": 0.5,
    "MAX_BETA": 3.0,
    "MAX_DEBT_WEIGHT": 0.60,
    "GDP_CLAMP": (-0.03, 0.06),
    "MU_DAILY_CLAMP": (-0.005, 0.005),
    "SIGMA_DAILY_CLAMP": (0.05 / (252**0.5), 2.0 / (252**0.5)),
    "RISK_FREE_RATE": 0.045,
    "EQUITY_RISK_PREMIUM": 0.050,
    # Drift blend: 15% storico, 85% risk-neutral (evita overfitting su momentum)
    "MC_DRIFT_BLEND": 0.15,
    "MC_EWMA_LAMBDA": 0.94,
    "MC_VOL_BLEND_EWMA": 0.70,
    # CORREZIONE 1: vol stochastiche DCF centralizzate qui
    "WACC_VOL": 0.15,
    "GROWTH_VOL": 0.25,
    "MARGIN_VOL": 0.10,
    # CORREZIONE 2: g_terminal parametrizzato
    "DCF_G_TERMINAL": 0.018,             # ~1.8% crescita nominale perpetua
    # Score bounds
    "SCORE_FLOOR": 5,
    "SCORE_CEILING": 95,
    # CORREZIONE 3: matrice correlazione unica (WACC, growth, margin)
    "CORRELATION_MATRIX": np.array([
        [ 1.0, -0.4, -0.3],
        [-0.4,  1.0,  0.6],
        [-0.3,  0.6,  1.0],
    ]),
}

# ---------------------------------------------------------------------------
# BENCHMARKS PER SETTORE (scoring)
# ---------------------------------------------------------------------------
SECTOR_BENCHMARKS = {
    "Technology":             {"pe": 28, "pb": 8,   "ev_ebitda": 20, "roe_min": 15, "margin_min": 15},
    "Financial Services":     {"pe": 15, "pb": 1.5, "ev_ebitda": 12, "roe_min": 10, "margin_min": 20},
    "Healthcare":             {"pe": 25, "pb": 5,   "ev_ebitda": 18, "roe_min": 12, "margin_min": 12},
    "Consumer Cyclical":      {"pe": 22, "pb": 4,   "ev_ebitda": 14, "roe_min": 12, "margin_min":  8},
    "Consumer Defensive":     {"pe": 20, "pb": 4,   "ev_ebitda": 14, "roe_min": 15, "margin_min": 10},
    "Industrials":            {"pe": 20, "pb": 3,   "ev_ebitda": 13, "roe_min": 12, "margin_min":  8},
    "Energy":                 {"pe": 15, "pb": 2,   "ev_ebitda":  8, "roe_min": 10, "margin_min":  8},
    "Utilities":              {"pe": 18, "pb": 2,   "ev_ebitda": 10, "roe_min":  8, "margin_min": 12},
    "Real Estate":            {"pe": 30, "pb": 2,   "ev_ebitda": 18, "roe_min":  8, "margin_min": 20},
    "Basic Materials":        {"pe": 16, "pb": 2.5, "ev_ebitda": 10, "roe_min": 10, "margin_min":  8},
    "Communication Services": {"pe": 22, "pb": 4,   "ev_ebitda": 14, "roe_min": 12, "margin_min": 12},
    "default":                {"pe": 20, "pb": 3,   "ev_ebitda": 14, "roe_min": 12, "margin_min": 10},
}

# Benchmarks per regione geografica (secondo layer di calibrazione scoring)
REGION_BENCHMARKS = {
    "USA":    {"pe": 24.0, "pb": 4.0, "ev_ebitda": 14.0, "roe_min": 12.0, "margin_min": 10.0},
    "Europe": {"pe": 20.0, "pb": 2.5, "ev_ebitda": 11.0, "roe_min": 10.0, "margin_min":  8.0},
    "Asia":   {"pe": 18.0, "pb": 2.0, "ev_ebitda": 10.0, "roe_min":  9.0, "margin_min":  8.0},
    "default":{"pe": 21.0, "pb": 3.0, "ev_ebitda": 12.0, "roe_min": 10.0, "margin_min":  9.0},
}

# Margini medi settoriali per mean-reversion DCF
SECTOR_MARGINS = {
    "Technology": 0.20, "Financial Services": 0.25, "Healthcare": 0.15,
    "Consumer Cyclical": 0.10, "Consumer Defensive": 0.12, "Industrials": 0.10,
    "Energy": 0.12, "Utilities": 0.15, "Real Estate": 0.25,
    "Basic Materials": 0.10, "Communication Services": 0.15, "default": 0.12,
}

# Temi di investimento con pesi dimensionali
THEMES = {
    "Balanced":   {"icon": "⚖️",  "weights": {"fundamental": 0.30, "momentum": 0.25, "dcf": 0.25, "quality": 0.20}},
    "Quality":    {"icon": "💎",  "weights": {"fundamental": 0.20, "momentum": 0.15, "dcf": 0.25, "quality": 0.40}},
    "Growth":     {"icon": "🚀",  "weights": {"fundamental": 0.15, "momentum": 0.35, "dcf": 0.35, "quality": 0.15}},
    "Value":      {"icon": "🏷️", "weights": {"fundamental": 0.40, "momentum": 0.10, "dcf": 0.35, "quality": 0.15}},
    "Defensive":  {"icon": "🛡️", "weights": {"fundamental": 0.25, "momentum": 0.10, "dcf": 0.25, "quality": 0.40}},
    "Tech Alpha": {"icon": "⚡",  "weights": {"fundamental": 0.10, "momentum": 0.40, "dcf": 0.30, "quality": 0.20}},
}

# Sane ranges per winsorizzazione
SANE_RANGES = {
    "pe_ratio":       (-500, 2000), "forward_pe":     (-500, 2000),
    "pb_ratio":       (-50,   200), "ev_ebitda":      (-100,  500),
    "ps_ratio":       (0,     200), "roe":            (-200,  300),
    "roa":            (-100,  100), "profit_margin":  (-200,  100),
    "operating_margin":(-200, 100), "debt_equity":    (0,    5000),
    "current_ratio":  (0,      50), "fcf_yield":      (-100,  200),
    "dividend_yield": (0,      50), "revenue_growth": (-100,  500),
    "eps_growth":     (-500, 5000), "peg_ratio":      (-10,   100),
    "beta":           (0,       5), "volatility":     (0,     500),
    "rsi":            (0,     100),
}

# CapEx e D&A come % revenue per settore: (min, max, default)
SECTOR_CAPEX = {
    "Technology": (0.02, 0.10, 0.04), "Communication Services": (0.02, 0.10, 0.05),
    "Healthcare": (0.03, 0.12, 0.05), "Consumer Defensive": (0.03, 0.10, 0.04),
    "Consumer Cyclical": (0.03, 0.12, 0.05), "Industrials": (0.03, 0.14, 0.06),
    "Basic Materials": (0.04, 0.18, 0.08), "Energy": (0.06, 0.22, 0.10),
    "Utilities": (0.05, 0.18, 0.08), "Real Estate": (0.03, 0.16, 0.06),
    "Financial Services": (0.02, 0.12, 0.04), "default": (0.03, 0.14, 0.06),
}

SECTOR_DA = {
    "Technology": (0.01, 0.08, 0.03), "Communication Services": (0.02, 0.10, 0.05),
    "Healthcare": (0.02, 0.10, 0.04), "Consumer Defensive": (0.02, 0.08, 0.03),
    "Consumer Cyclical": (0.02, 0.10, 0.04), "Industrials": (0.03, 0.12, 0.05),
    "Basic Materials": (0.04, 0.16, 0.07), "Energy": (0.05, 0.20, 0.10),
    "Utilities": (0.04, 0.16, 0.07), "Real Estate": (0.03, 0.14, 0.06),
    "Financial Services": (0.01, 0.06, 0.03), "default": (0.02, 0.12, 0.05),
}

SECTOR_GROWTH = {
    "Technology": (-0.05, 0.12), "Communication Services": (-0.05, 0.10),
    "Healthcare": (-0.04, 0.10), "Consumer Cyclical": (-0.06, 0.10),
    "Consumer Defensive": (-0.04, 0.08), "Industrials": (-0.05, 0.08),
    "Basic Materials": (-0.06, 0.08), "Energy": (-0.08, 0.07),
    "Utilities": (-0.04, 0.06), "Real Estate": (-0.05, 0.07),
    "Financial Services": (-0.05, 0.08), "default": (-0.05, 0.09),
}

SECTOR_MARGIN_CAP = {
    "Technology": 0.35, "Communication Services": 0.30, "Healthcare": 0.30,
    "Consumer Cyclical": 0.22, "Consumer Defensive": 0.22, "Industrials": 0.20,
    "Basic Materials": 0.22, "Energy": 0.25, "Utilities": 0.22,
    "Real Estate": 0.30, "Financial Services": 0.35, "default": 0.25,
}


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------
def clamp(x, lo, hi):
    if x is None:
        return None
    return max(lo, min(hi, x))


def sanitize_fundamentals(data: dict) -> dict:
    """Winsorizza i fondamentali entro SANE_RANGES, traccia outlier."""
    clean, flags = {}, []
    for key, val in data.items():
        if key in SANE_RANGES and val is not None:
            lo, hi = SANE_RANGES[key]
            try:
                v = float(val)
                if np.isfinite(v):
                    if v < lo or v > hi:
                        flags.append(key)
                    clean[key] = float(clamp(v, lo, hi))
                else:
                    clean[key] = None
            except (TypeError, ValueError):
                clean[key] = val
        else:
            clean[key] = val
    clean["_outlier_flags"] = flags
    return clean
