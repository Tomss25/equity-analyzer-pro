"""
logic/data_fetcher.py
Equity Analyzer Pro — Fetch robusto con fallback multipli.

Gerarchia provider:
  1. Yahoo Finance (yfinance) — primario, gratuito
  2. Financial Modeling Prep (FMP) — opzionale, richiede API key
  3. Alpha Vantage — opzionale, richiede API key

Se Yahoo restituisce dati vuoti o parziali, prova automaticamente gli altri.
"""

from __future__ import annotations

import gzip
import os
import pickle
import threading
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .config import CONFIG

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _env(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return None


# ---------------------------------------------------------------------------
# HELPERS CONDIVISI
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    if v is None: return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _safe_pct(v) -> Optional[float]:
    f = _safe_float(v)
    return f * 100 if f is not None else None

def _safe_pct_str(v) -> Optional[float]:
    if v is None: return None
    try:
        s = str(v).replace("%", "").strip()
        f = float(s)
        return f * 100 if abs(f) < 5 else f
    except Exception:
        return None

def _fcf_yield(info: dict) -> Optional[float]:
    try:
        fcf = info.get("freeCashflow")
        mc  = info.get("marketCap")
        if fcf and mc and mc > 0:
            return float(fcf) / float(mc) * 100
    except Exception:
        pass
    return None

def _calc_rsi(closes: list, period: int = 14) -> Optional[float]:
    try:
        if len(closes) < period + 1: return None
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains  = [max(d, 0)  for d in deltas[-period:]]
        losses = [abs(min(d, 0)) for d in deltas[-period:]]
        avg_g  = sum(gains)  / period
        avg_l  = sum(losses) / period
        if avg_l == 0: return 100.0
        return float(100 - 100 / (1 + avg_g / avg_l))
    except Exception:
        return None

def _country_to_region(country: str) -> str:
    c = (country or "").strip().upper()
    if c in {"UNITED STATES", "USA", "US", "UNITED STATES OF AMERICA"}: return "USA"
    if c in {"GERMANY","FRANCE","ITALY","SPAIN","NETHERLANDS","BELGIUM","SWEDEN",
             "SWITZERLAND","DENMARK","NORWAY","FINLAND","AUSTRIA","PORTUGAL",
             "IRELAND","LUXEMBOURG","UNITED KINGDOM","UK"}: return "Europe"
    if c in {"JAPAN","CHINA","SOUTH KOREA","TAIWAN","HONG KONG","SINGAPORE",
             "INDIA","AUSTRALIA","NEW ZEALAND"}: return "Asia"
    return "default"

def _pret(closes: list, n: int) -> Optional[float]:
    if len(closes) < n + 1: return None
    return (closes[-1] / closes[-n-1] - 1) * 100


# ---------------------------------------------------------------------------
# PROVIDER: Yahoo Finance
# ---------------------------------------------------------------------------

def _yahoo_fetch(ticker: str) -> dict:
    try:
        import yfinance as yf

        t_obj = yf.Ticker(ticker)

        hist = t_obj.history(period="2y", auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 30:
            hist = t_obj.history(period="1y", auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 30:
            return {}

        hist      = hist[["Close", "Volume"]].dropna()
        closes    = hist["Close"].tolist()
        dates     = [str(d.date()) for d in hist.index]
        d_returns = list(hist["Close"].pct_change().dropna())

        try:
            info = t_obj.info or {}
        except Exception:
            info = {}

        cur = (info.get("currentPrice") or info.get("regularMarketPrice")
               or info.get("previousClose") or closes[-1])
        try:
            cur = float(cur)
        except Exception:
            cur = float(closes[-1])
        if cur <= 0: return {}

        vol_d = float(np.std(d_returns[-252:])) if len(d_returns) >= 30 else 0.0
        vol_a = vol_d * (252**0.5) * 100
        ma50  = float(np.mean(closes[-50:]))  if len(closes) >= 50  else None
        ma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else None
        rsi   = _calc_rsi(closes)

        sharpe = None
        if len(d_returns) >= 60:
            arr = np.array(d_returns[-252:])
            exc = arr - (0.045/252)
            s   = float(np.std(exc))
            if s > 0: sharpe = float(np.mean(exc)/s*(252**0.5))

        def p(v):
            if v is None: return None
            try: return float(v)*100
            except: return None

        capex = da = delta_wc = None
        try:
            cf = t_obj.cashflow
            if cf is not None and not cf.empty:
                for n in ["Capital Expenditure","capitalExpenditures"]:
                    if n in cf.index:
                        v = cf.loc[n].iloc[0]
                        if pd.notna(v): capex = abs(float(v)); break
                for n in ["Depreciation & Amortization","depreciationAndAmortization"]:
                    if n in cf.index:
                        v = cf.loc[n].iloc[0]
                        if pd.notna(v): da = abs(float(v)); break
        except Exception: pass
        try:
            bs = t_obj.balance_sheet
            if bs is not None and not bs.empty and bs.shape[1] >= 2:
                cak = next((k for k in ["Current Assets","TotalCurrentAssets"] if k in bs.index), None)
                clk = next((k for k in ["Current Liabilities","TotalCurrentLiabilities"] if k in bs.index), None)
                if cak and clk:
                    wc0 = float(bs.loc[cak].iloc[0]) - float(bs.loc[clk].iloc[0])
                    wc1 = float(bs.loc[cak].iloc[1]) - float(bs.loc[clk].iloc[1])
                    delta_wc = wc0 - wc1
        except Exception: pass

        td  = info.get("totalDebt")
        cas = info.get("totalCash")
        nd  = None
        if td is not None and cas is not None:
            try: nd = float(td) - float(cas)
            except: pass

        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        mc     = info.get("marketCap")
        country = info.get("country", "")

        return {
            "ticker": ticker,
            "name":   info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "default",
            "industry": info.get("industry"),
            "region":   _country_to_region(country),
            "country":  country,
            "currency": info.get("currency", "USD"),
            "current_price": cur,
            "market_cap":    mc,
            "fundamentals": {
                "pe_ratio":         info.get("trailingPE"),
                "forward_pe":       info.get("forwardPE"),
                "pb_ratio":         info.get("priceToBook"),
                "ev_ebitda":        info.get("enterpriseToEbitda"),
                "ps_ratio":         info.get("priceToSalesTrailing12Months"),
                "roe":              p(info.get("returnOnEquity")),
                "roa":              p(info.get("returnOnAssets")),
                "profit_margin":    p(info.get("profitMargins")),
                "operating_margin": p(info.get("operatingMargins")),
                "debt_equity":      info.get("debtToEquity"),
                "current_ratio":    info.get("currentRatio"),
                "fcf_yield":        _fcf_yield(info),
                "dividend_yield":   p(info.get("dividendYield")),
                "revenue_growth":   p(info.get("revenueGrowth")),
                "eps_growth":       p(info.get("earningsGrowth")),
                "peg_ratio":        info.get("pegRatio"),
                "beta":             info.get("beta"),
                "volatility":       vol_a,
                "rsi":              rsi,
            },
            "technicals": {
                "beta": info.get("beta"), "volatility": vol_a, "rsi": rsi,
                "r1m":  _pret(closes,21), "r3m": _pret(closes,63),
                "r6m":  _pret(closes,126),"r12m": _pret(closes,252),
                "ma_cross": (cur > ma50 and ma50 > ma200) if (ma50 and ma200) else None,
                "ma50": ma50, "ma200": ma200, "sharpe_1y": sharpe,
            },
            "intrinsic_inputs": {
                "revenue_ttm": info.get("totalRevenue"),
                "ebit_ttm":    info.get("ebit"),
                "tax_rate":    info.get("effectiveTaxRate"),
                "capex": capex, "da": da, "delta_wc": delta_wc,
                "shares_outstanding": shares,
                "net_debt": nd, "total_debt": td, "cash": cas,
            },
            "daily_returns": d_returns,
            "price_history": closes,
            "price_dates":   dates,
            "_source": "yahoo",
            "_fetched_at": time.time(),
        }
    except Exception as e:
        print(f"[Yahoo] Errore {ticker}: {e}")
        return {}


# ---------------------------------------------------------------------------
# PROVIDER: FMP
# ---------------------------------------------------------------------------

def _fmp_fetch(ticker: str, api_key: str) -> dict:
    try:
        import requests
        base = "https://financialmodelingprep.com/api/v3"
        h = {"User-Agent": "EquityAnalyzerPro/8.0"}
        kw = {"apikey": api_key}

        r_p = requests.get(f"{base}/profile/{ticker}", params=kw, headers=h, timeout=10)
        prof = r_p.json() if r_p.ok else []
        if not prof or not isinstance(prof, list): return {}
        p = prof[0]

        r_q = requests.get(f"{base}/quote/{ticker}", params=kw, headers=h, timeout=10)
        quot = r_q.json() if r_q.ok else []
        q = quot[0] if quot and isinstance(quot, list) else {}

        r_km = requests.get(f"{base}/key-metrics-ttm/{ticker}", params=kw, headers=h, timeout=10)
        km_d = r_km.json() if r_km.ok else []
        km = km_d[0] if km_d and isinstance(km_d, list) else {}

        r_inc = requests.get(f"{base}/income-statement/{ticker}", params={**kw,"limit":1}, headers=h, timeout=10)
        inc_d = r_inc.json() if r_inc.ok else []
        inc = inc_d[0] if inc_d and isinstance(inc_d, list) else {}

        r_bs = requests.get(f"{base}/balance-sheet-statement/{ticker}", params={**kw,"limit":2}, headers=h, timeout=10)
        bs_d = r_bs.json() if r_bs.ok else []
        bs0  = bs_d[0] if len(bs_d) > 0 else {}
        bs1  = bs_d[1] if len(bs_d) > 1 else {}

        r_cf = requests.get(f"{base}/cash-flow-statement/{ticker}", params={**kw,"limit":1}, headers=h, timeout=10)
        cf_d = r_cf.json() if r_cf.ok else []
        cf = cf_d[0] if cf_d and isinstance(cf_d, list) else {}

        r_hist = requests.get(f"{base}/historical-price-full/{ticker}", params={**kw,"timeseries":504}, headers=h, timeout=15)
        hist_j = r_hist.json() if r_hist.ok else {}
        hist_l = hist_j.get("historical", []) if isinstance(hist_j, dict) else []

        cur = float(q.get("price") or p.get("price") or 0)
        if cur <= 0: return {}

        closes = [float(h2["close"]) for h2 in reversed(hist_l) if h2.get("close")]
        dates  = [h2["date"] for h2 in reversed(hist_l) if h2.get("close")]
        if len(closes) < 30: return {}

        d_returns = [(closes[i]/closes[i-1]-1) for i in range(1, len(closes))]
        vol_d  = float(np.std(d_returns[-252:])) if d_returns else 0.0
        vol_a  = vol_d * (252**0.5) * 100
        ma50   = float(np.mean(closes[-50:]))  if len(closes)>=50  else None
        ma200  = float(np.mean(closes[-200:])) if len(closes)>=200 else None
        rsi    = _calc_rsi(closes)

        mc = float(p.get("mktCap") or 0)
        country = p.get("country","")

        td  = _safe_float(bs0.get("totalDebt"))
        cas = _safe_float(bs0.get("cashAndCashEquivalents"))
        nd  = (td - cas) if (td is not None and cas is not None) else None

        wc0 = (_safe_float(bs0.get("totalCurrentAssets")) or 0) - (_safe_float(bs0.get("totalCurrentLiabilities")) or 0)
        wc1 = (_safe_float(bs1.get("totalCurrentAssets")) or 0) - (_safe_float(bs1.get("totalCurrentLiabilities")) or 0) if bs1 else None
        dwc = (wc0 - wc1) if wc1 is not None else None

        return {
            "ticker": ticker,
            "name":   p.get("companyName") or ticker,
            "sector": p.get("sector") or "default",
            "industry": p.get("industry"),
            "region":   _country_to_region(country),
            "country":  country,
            "currency": p.get("currency","USD"),
            "current_price": cur,
            "market_cap": mc,
            "fundamentals": {
                "pe_ratio":         _safe_float(q.get("pe")),
                "forward_pe":       _safe_float(km.get("peRatioTTM")),
                "pb_ratio":         _safe_float(km.get("pbRatioTTM")),
                "ev_ebitda":        _safe_float(km.get("evToEbitdaTTM")),
                "ps_ratio":         _safe_float(km.get("priceToSalesRatioTTM")),
                "roe":              _safe_pct(km.get("roeTTM")),
                "roa":              _safe_pct(km.get("returnOnTangibleAssetsTTM")),
                "profit_margin":    _safe_pct(km.get("netProfitMarginTTM")),
                "operating_margin": _safe_pct(inc.get("operatingIncomeRatio")),
                "debt_equity":      _safe_float(km.get("debtToEquityTTM")),
                "current_ratio":    _safe_float(km.get("currentRatioTTM")),
                "fcf_yield":        _safe_pct(km.get("freeCashFlowYieldTTM")),
                "dividend_yield":   _safe_float(p.get("lastDiv")),
                "revenue_growth":   _safe_pct(inc.get("revenueGrowth")),
                "eps_growth":       _safe_pct(inc.get("epsgrowth")),
                "peg_ratio":        _safe_float(km.get("priceEarningsToGrowthRatioTTM")),
                "beta":             _safe_float(p.get("beta")),
                "volatility":       vol_a,
                "rsi":              rsi,
            },
            "technicals": {
                "beta": _safe_float(p.get("beta")), "volatility": vol_a, "rsi": rsi,
                "r1m":  _pret(closes,21), "r3m": _pret(closes,63),
                "r6m":  _pret(closes,126),"r12m": _pret(closes,252),
                "ma_cross": (cur > ma50 and ma50 > ma200) if (ma50 and ma200) else None,
                "ma50": ma50, "ma200": ma200,
            },
            "intrinsic_inputs": {
                "revenue_ttm": _safe_float(inc.get("revenue")),
                "ebit_ttm":    _safe_float(inc.get("operatingIncome")),
                "tax_rate":    None,
                "capex":       abs(_safe_float(cf.get("capitalExpenditure")) or 0) or None,
                "da":          _safe_float(cf.get("depreciationAndAmortization")),
                "delta_wc":    dwc,
                "shares_outstanding": _safe_float(p.get("sharesOutstanding") or (mc/cur if cur>0 else None)),
                "net_debt": nd, "total_debt": td, "cash": cas,
            },
            "daily_returns": d_returns,
            "price_history": closes,
            "price_dates":   dates,
            "_source": "fmp",
            "_fetched_at": time.time(),
        }
    except Exception as e:
        print(f"[FMP] Errore {ticker}: {e}")
        return {}


# ---------------------------------------------------------------------------
# PROVIDER: Alpha Vantage
# ---------------------------------------------------------------------------

def _av_fetch(ticker: str, api_key: str) -> dict:
    try:
        import requests
        base = "https://www.alphavantage.co/query"
        h = {"User-Agent": "EquityAnalyzerPro/8.0"}

        r_ov = requests.get(base, params={"function":"OVERVIEW","symbol":ticker,"apikey":api_key}, headers=h, timeout=10)
        ov = r_ov.json() if r_ov.ok else {}
        if not ov or ov.get("Information") or ov.get("Note"): return {}

        r_d = requests.get(base, params={"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":ticker,"outputsize":"full","apikey":api_key}, headers=h, timeout=15)
        ts = (r_d.json() if r_d.ok else {}).get("Time Series (Daily)", {})
        if not ts: return {}

        sorted_dates = sorted(ts.keys())[-504:]
        closes = [float(ts[d]["5. adjusted close"]) for d in sorted_dates]
        dates  = sorted_dates
        if len(closes) < 30: return {}

        cur = closes[-1]
        d_returns = [(closes[i]/closes[i-1]-1) for i in range(1, len(closes))]
        vol_d  = float(np.std(d_returns[-252:])) if d_returns else 0.0
        vol_a  = vol_d * (252**0.5) * 100
        ma50   = float(np.mean(closes[-50:]))  if len(closes)>=50  else None
        ma200  = float(np.mean(closes[-200:])) if len(closes)>=200 else None
        rsi    = _calc_rsi(closes)

        country = ov.get("Country","")
        mc = _safe_float(ov.get("MarketCapitalization"))

        return {
            "ticker": ticker,
            "name":   ov.get("Name") or ticker,
            "sector": ov.get("Sector") or "default",
            "industry": ov.get("Industry"),
            "region":   _country_to_region(country),
            "country":  country,
            "currency": ov.get("Currency","USD"),
            "current_price": cur,
            "market_cap": mc,
            "fundamentals": {
                "pe_ratio":         _safe_float(ov.get("TrailingPE")),
                "forward_pe":       _safe_float(ov.get("ForwardPE")),
                "pb_ratio":         _safe_float(ov.get("PriceToBookRatio")),
                "ev_ebitda":        _safe_float(ov.get("EVToEBITDA")),
                "ps_ratio":         _safe_float(ov.get("PriceToSalesRatioTTM")),
                "roe":              _safe_pct_str(ov.get("ReturnOnEquityTTM")),
                "roa":              _safe_pct_str(ov.get("ReturnOnAssetsTTM")),
                "profit_margin":    _safe_pct_str(ov.get("ProfitMargin")),
                "operating_margin": _safe_pct_str(ov.get("OperatingMarginTTM")),
                "debt_equity":      _safe_float(ov.get("DebtToEquityRatio")),
                "dividend_yield":   _safe_pct_str(ov.get("DividendYield")),
                "revenue_growth":   _safe_pct_str(ov.get("QuarterlyRevenueGrowthYOY")),
                "eps_growth":       _safe_pct_str(ov.get("QuarterlyEarningsGrowthYOY")),
                "peg_ratio":        _safe_float(ov.get("PEGRatio")),
                "beta":             _safe_float(ov.get("Beta")),
                "volatility":       vol_a,
                "rsi":              rsi,
            },
            "technicals": {
                "beta": _safe_float(ov.get("Beta")), "volatility": vol_a, "rsi": rsi,
                "r1m":  _pret(closes,21), "r3m": _pret(closes,63),
                "r6m":  _pret(closes,126),"r12m": _pret(closes,252),
                "ma_cross": (cur > ma50 and ma50 > ma200) if (ma50 and ma200) else None,
                "ma50": ma50, "ma200": ma200,
            },
            "intrinsic_inputs": {
                "revenue_ttm": _safe_float(ov.get("RevenueTTM")),
                "ebit_ttm":    _safe_float(ov.get("EBITDA")),
                "shares_outstanding": _safe_float(ov.get("SharesOutstanding")),
                "tax_rate": None, "capex": None, "da": None, "delta_wc": None,
                "net_debt": None, "total_debt": None, "cash": None,
            },
            "daily_returns": d_returns,
            "price_history": closes,
            "price_dates":   dates,
            "_source": "alphavantage",
            "_fetched_at": time.time(),
        }
    except Exception as e:
        print(f"[AlphaVantage] Errore {ticker}: {e}")
        return {}


def _merge(base: dict, enr: dict) -> dict:
    r = dict(base)
    for k in ["name","sector","industry","region","country","currency","market_cap"]:
        if not r.get(k) and enr.get(k):
            r[k] = enr[k]
    for section in ["fundamentals","intrinsic_inputs"]:
        bd = r.get(section, {}); ed = enr.get(section, {})
        for k, v in ed.items():
            if bd.get(k) is None and v is not None:
                bd[k] = v
        r[section] = bd
    return r


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class DataFetcher:
    _cache: dict = {}
    _cache_lock   = threading.Lock()
    CACHE_TTL      = 300
    DISK_CACHE_TTL = 3600
    DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")

    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = risk_free_rate
        os.makedirs(self.DISK_CACHE_DIR, exist_ok=True)

    def fetch(self, ticker: str, force_refresh: bool = False,
              fmp_key: str = "", av_key: str = "") -> Optional[dict]:
        ticker = (ticker or "").strip().upper()
        if not ticker: return None
        now = time.time()

        if not force_refresh:
            with self._cache_lock:
                if ticker in self._cache:
                    data, ts = self._cache[ticker]
                    if now - ts < self.CACHE_TTL:
                        return data
            disk = self._load_disk(ticker)
            if disk is not None:
                with self._cache_lock:
                    self._cache[ticker] = (disk, now)
                return disk

        result = self._live(ticker, fmp_key, av_key)
        if result:
            self._save_disk(ticker, result)
            with self._cache_lock:
                self._cache[ticker] = (result, now)
        return result

    def _live(self, ticker: str, fmp_key: str, av_key: str) -> Optional[dict]:
        fmp_key = fmp_key or _env("FMP_API_KEY","FINANCIAL_MODELING_PREP_API_KEY") or ""
        av_key  = av_key  or _env("ALPHAVANTAGE_API_KEY","ALPHA_VANTAGE_API_KEY") or ""

        # 1. Yahoo
        result = _yahoo_fetch(ticker)

        # 2. FMP se Yahoo fallisce
        if not result and fmp_key:
            result = _fmp_fetch(ticker, fmp_key)

        # 3. Alpha Vantage come ultimo tentativo
        if not result and av_key:
            result = _av_fetch(ticker, av_key)

        if not result:
            return None

        # Arricchimento: se Yahoo ha dati ma fondamentali mancanti, integra con FMP
        if result.get("_source") == "yahoo" and fmp_key:
            f = result.get("fundamentals", {})
            missing = sum(1 for k in ["pe_ratio","pb_ratio","roe","operating_margin"] if f.get(k) is None)
            if missing >= 2:
                enr = _fmp_fetch(ticker, fmp_key)
                if enr: result = _merge(result, enr)

        return result

    def _cache_path(self, ticker: str) -> str:
        ver = CONFIG.get("CACHE_VERSION", "v8")
        return os.path.join(self.DISK_CACHE_DIR, f"{ticker}_{ver}.pkl.gz")

    def _load_disk(self, ticker: str) -> Optional[dict]:
        path = self._cache_path(ticker)
        if not os.path.exists(path): return None
        try:
            if time.time() - os.path.getmtime(path) > self.DISK_CACHE_TTL: return None
            with gzip.open(path, "rb") as f: return pickle.load(f)
        except Exception: return None

    def _save_disk(self, ticker: str, data: dict) -> None:
        try:
            with gzip.open(self._cache_path(ticker), "wb") as f:
                pickle.dump(data, f, protocol=4)
        except Exception as e:
            print(f"[Fetcher] Disk write error {ticker}: {e}")
