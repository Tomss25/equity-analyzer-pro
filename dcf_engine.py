"""
logic/stress_test.py
Stress testing macro: impatto scenari su singoli titoli e portafogli.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

STRESS_SCENARIOS = {
    "recession_severe": {
        "name": "Recessione Severa",
        "icon": "📉",
        "color": "#f85149",
        "market_shock": -0.30,
        "rate_change_bps": -200,
        "duration_months": 18,
        "recovery_months": 36,
        "sector_multipliers": {
            "Technology": 1.3, "Consumer Cyclical": 1.5, "Financial Services": 1.4,
            "Industrials": 1.3, "Energy": 0.8, "Consumer Defensive": 0.5,
            "Healthcare": 0.6, "Utilities": 0.5, "Real Estate": 1.2,
            "Basic Materials": 1.1, "Communication Services": 1.1, "default": 1.0,
        },
    },
    "rate_shock_200bps": {
        "name": "Shock Tassi +200bps",
        "icon": "📈",
        "color": "#ff6d00",
        "market_shock": -0.15,
        "rate_change_bps": +200,
        "duration_months": 9,
        "recovery_months": 18,
        "sector_multipliers": {
            "Technology": 1.6, "Real Estate": 1.8, "Utilities": 1.4,
            "Consumer Cyclical": 1.2, "Financial Services": 0.6, "Energy": 0.9,
            "Consumer Defensive": 0.8, "Healthcare": 0.9, "Industrials": 1.1,
            "Basic Materials": 1.0, "Communication Services": 1.3, "default": 1.0,
        },
    },
    "inflation_shock": {
        "name": "Shock Inflazione +5%",
        "icon": "🔥",
        "color": "#fbbf24",
        "market_shock": -0.12,
        "rate_change_bps": +150,
        "duration_months": 12,
        "recovery_months": 24,
        "sector_multipliers": {
            "Energy": 0.4, "Basic Materials": 0.5, "Consumer Defensive": 0.7,
            "Technology": 1.2, "Real Estate": 1.5, "Utilities": 1.3,
            "Consumer Cyclical": 1.3, "Financial Services": 0.8, "Healthcare": 0.9,
            "Industrials": 0.9, "Communication Services": 1.1, "default": 1.0,
        },
    },
    "tech_crash": {
        "name": "Crash Tecnologico",
        "icon": "💥",
        "color": "#c084fc",
        "market_shock": -0.25,
        "rate_change_bps": 0,
        "duration_months": 24,
        "recovery_months": 48,
        "sector_multipliers": {
            "Technology": 2.2, "Communication Services": 1.8, "Consumer Cyclical": 1.3,
            "Financial Services": 1.1, "Healthcare": 0.7, "Utilities": 0.6,
            "Consumer Defensive": 0.5, "Energy": 0.8, "Industrials": 1.0,
            "Basic Materials": 0.9, "Real Estate": 1.0, "default": 1.0,
        },
    },
    "stagflation": {
        "name": "Stagflazione",
        "icon": "🌀",
        "color": "#94a3b8",
        "market_shock": -0.20,
        "rate_change_bps": +100,
        "duration_months": 24,
        "recovery_months": 36,
        "sector_multipliers": {
            "Energy": 0.3, "Basic Materials": 0.5, "Consumer Defensive": 0.6,
            "Technology": 1.5, "Real Estate": 1.6, "Consumer Cyclical": 1.7,
            "Financial Services": 1.2, "Healthcare": 0.8, "Utilities": 1.2,
            "Industrials": 1.1, "Communication Services": 1.2, "default": 1.0,
        },
    },
}


class StressTestEngine:

    def run_single(self, stock_data: dict, scenario_key: str) -> dict:
        """Calcola impatto di uno scenario su un singolo titolo."""
        if scenario_key not in STRESS_SCENARIOS:
            return {"error": f"Scenario '{scenario_key}' non trovato"}

        sc    = STRESS_SCENARIOS[scenario_key]
        sector = stock_data.get("sector", "default") or "default"
        beta   = float(stock_data.get("technicals", {}).get("beta") or 1.0)
        beta   = max(0.1, min(3.0, beta))

        mkt_shock   = float(sc["market_shock"])
        sector_mult = sc["sector_multipliers"].get(sector, sc["sector_multipliers"].get("default", 1.0))

        # Drawdown stimato: beta × shock mercato × moltiplicatore settore
        drawdown  = mkt_shock * beta * sector_mult
        drawdown  = max(-0.95, min(0.20, drawdown))   # guardrail: max -95% / +20%

        cur_price = float(stock_data.get("current_price") or 0)
        stressed_price = cur_price * (1.0 + drawdown) if cur_price > 0 else 0.0

        # Recovery: tempo e prezzo stimato
        wacc_approx   = 0.09
        recovery_yrs  = sc["recovery_months"] / 12.0
        recovery_price = stressed_price * ((1.0 + wacc_approx) ** recovery_yrs)

        return {
            "scenario":          sc["name"],
            "icon":              sc["icon"],
            "color":             sc["color"],
            "drawdown_pct":      round(drawdown * 100, 1),
            "stressed_price":    round(stressed_price, 2),
            "recovery_price":    round(recovery_price, 2),
            "duration_months":   sc["duration_months"],
            "recovery_months":   sc["recovery_months"],
            "beta_used":         round(beta, 2),
            "sector_multiplier": round(sector_mult, 2),
        }

    def run_all(self, stock_data: dict) -> dict:
        """Esegue tutti gli scenari e ritorna dict {scenario_key: result}."""
        results = {}
        for key in STRESS_SCENARIOS:
            results[key] = self.run_single(stock_data, key)
        return results
