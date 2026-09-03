#!/usr/bin/env python3
"""
collect_technicals.py — pull OHLC from the Upstox public historical-candle API
and compute positional technical parameters for a list of NSE stocks.

Why Upstox: the /v3/historical-candle endpoint needs NO authentication, serves
native weekly and monthly candles back to Jan 2000, and its instrument key is
NSE_EQ|<ISIN> — which the Nifty 500 constituent list already carries.

    pip install requests pandas
    python collect_technicals.py                      # the 20 sector leaders
    python collect_technicals.py --csv nifty500.csv   # any list with an ISIN column
    python collect_technicals.py --out technicals.json

Output: one JSON file, schema matched to the Sector Leaders Technical Board.

Rate limits are 25/sec, 250/min, 1000/30min. Two calls per symbol, so 20 symbols
is 40 calls — nowhere near any limit. The full 499 is ~1000 calls; the script
paces itself and will take a few minutes.

NOTE: this hits an undocumented-as-public endpoint. It works today (verified
24 Aug 2026) but is not contractually guaranteed. If it ever 404s or starts
demanding a token, Angel One SmartAPI is the free fallback — it needs a TOTP
login and daily candles you resample to weekly yourself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import date, timedelta
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("pip install requests pandas")

BASE = "https://api.upstox.com/v3/historical-candle"
HEADERS = {"Accept": "application/json"}
PAUSE = 0.15          # seconds between calls — well inside 25/sec
RETRIES = 3

# The 20 NSE sector leaders by market cap. symbol, name, sector, ISIN.
LEADERS = [
    ("RELIANCE",   "Reliance Industries",      "Oil Gas & Consumable Fuels",        "INE002A01018"),
    ("BHARTIARTL", "Bharti Airtel",            "Telecommunication",                 "INE397D01024"),
    ("HDFCBANK",   "HDFC Bank",                "Financial Services",                "INE040A01034"),
    ("TCS",        "Tata Consultancy Services","Information Technology",            "INE467B01029"),
    ("LT",         "Larsen & Toubro",          "Construction",                      "INE018A01030"),
    ("HINDUNILVR", "Hindustan Unilever",       "Fast Moving Consumer Goods",        "INE030A01027"),
    ("SUNPHARMA",  "Sun Pharmaceutical",       "Healthcare",                        "INE044A01036"),
    ("TITAN",      "Titan Company",            "Consumer Durables",                 "INE280A01028"),
    ("MARUTI",     "Maruti Suzuki India",      "Automobile and Auto Components",    "INE585B01010"),
    ("ADANIENT",   "Adani Enterprises",        "Metals & Mining",                   "INE423A01024"),
    ("ADANIPOWER", "Adani Power",              "Power",                             "INE814H01029"),
    ("ADANIPORTS", "Adani Ports & SEZ",        "Services",                          "INE742F01042"),
    ("ULTRACEMCO", "UltraTech Cement",         "Construction Materials",            "INE481G01011"),
    ("HAL",        "Hindustan Aeronautics",    "Capital Goods",                     "INE066F01020"),
    ("ETERNAL",    "Eternal",                  "Consumer Services",                 "INE758T01015"),
    ("SOLARINDS",  "Solar Industries India",   "Chemicals",                         "INE343H01029"),
    ("DLF",        "DLF",                      "Realty",                            "INE271C01023"),
    ("GODREJIND",  "Godrej Industries",        "Diversified",                       "INE233A01035"),
    ("PAGEIND",    "Page Industries",          "Textiles",                          "INE761H01022"),
    ("PFOCUS",     "Prime Focus",              "Media Entertainment & Publication", "INE367G01038"),
]


# ---------------------------------------------------------------- indicators
# Written out rather than pulled from pandas-ta so the definitions are visible
# and pinned. RSI and MACD both use Wilder/EMA smoothing, matching what charting
# platforms show.

def sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values: list[float], period: int) -> list[float]:
    """Standard EMA, seeded with an SMA of the first `period` values."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(values: list[float], period: int = 14) -> float | None:
    """Wilder's RSI — the one TradingView, Investing.com and brokers display."""
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for prev, cur in zip(values, values[1:]):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) at the latest bar."""
    if len(values) < slow + signal:
        return None, None, None
    fast_e = ema_series(values, fast)
    slow_e = ema_series(values, slow)
    # align: the fast EMA starts (slow - fast) bars earlier
    offset = slow - fast
    line = [f - s for f, s in zip(fast_e[offset:], slow_e)]
    if len(line) < signal:
        return None, None, None
    sig = ema_series(line, signal)
    if not sig:
        return None, None, None
    return line[-1], sig[-1], line[-1] - sig[-1]


# ---------------------------------------------------------------- fetching

def fetch_candles(isin: str, unit: str, to_date: str, from_date: str) -> list[list[Any]]:
    """
    Upstox returns candles NEWEST FIRST as
    [timestamp, open, high, low, close, volume, open_interest].
    This flips them to oldest-first, which every indicator here assumes.
    """
    url = f"{BASE}/NSE_EQ%7C{isin}/{unit}/1/{to_date}/{from_date}"
    last_err = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            payload = r.json()
            if payload.get("status") != "success":
                raise RuntimeError(f"api status={payload.get('status')} {payload}")
            return list(reversed(payload["data"]["candles"]))
        except Exception as exc:                      # noqa: BLE001
            last_err = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"{isin} {unit}: {last_err}")


def closes(candles: list[list[Any]]) -> list[float]:
    return [float(c[4]) for c in candles]


# ---------------------------------------------------------------- per symbol

def analyse(symbol: str, name: str, sector: str, isin: str, today: date) -> dict:
    d_from = (today - timedelta(days=800)).isoformat()     # ~550 trading days
    w_from = (today - timedelta(days=365 * 6)).isoformat() # ~310 weekly bars
    to = today.isoformat()

    daily = fetch_candles(isin, "days", to, d_from)
    time.sleep(PAUSE)
    weekly = fetch_candles(isin, "weeks", to, w_from)
    time.sleep(PAUSE)

    dc, wc = closes(daily), closes(weekly)
    if len(dc) < 60:
        raise RuntimeError(f"{symbol}: only {len(dc)} daily bars")

    cmp_ = dc[-1]
    last_252 = daily[-252:] if len(daily) >= 252 else daily
    h52 = max(float(c[2]) for c in last_252)
    l52 = min(float(c[3]) for c in last_252)

    s50, s100, s200 = sma(dc, 50), sma(dc, 100), sma(dc, 200)
    md_line, md_sig, md_hist = macd(dc)
    mw_line, mw_sig, mw_hist = macd(wc)

    def ret(bars: list[float], back: int) -> float | None:
        if len(bars) <= back:
            return None
        past = bars[-1 - back]
        return None if past == 0 else (bars[-1] / past - 1) * 100

    if s50 and s100 and s200:
        stack = ("bullish" if s50 > s100 > s200
                 else "bearish" if s50 < s100 < s200 else "mixed")
    else:
        stack = "n/a"

    vols = [float(c[5]) for c in daily]
    v20 = sum(vols[-20:]) / 20 if len(vols) >= 20 else None
    v60 = sum(vols[-60:]) / 60 if len(vols) >= 60 else None
    if v20 and v60:
        participation = "expanding" if v20 > v60 * 1.1 else "contracting" if v20 < v60 * 0.9 else "steady"
    else:
        participation = "n/a"

    return {
        "sym": symbol, "name": name, "sector": sector, "isin": isin,
        "asof": daily[-1][0][:10],
        "cmp": round(cmp_, 2),
        "h52": round(h52, 2), "l52": round(l52, 2),
        "pct_from_52w_high": round((cmp_ / h52 - 1) * 100, 2),
        "sma50": round(s50, 2) if s50 else None,
        "sma100": round(s100, 2) if s100 else None,
        "sma200": round(s200, 2) if s200 else None,
        "pct_vs_sma200": round((cmp_ / s200 - 1) * 100, 2) if s200 else None,
        "stack": stack,
        "rsiD": round(rsi(dc), 1) if rsi(dc) is not None else None,
        "rsiW": round(rsi(wc), 1) if rsi(wc) is not None else None,
        "macdD": None if md_hist is None else ("bullish" if md_hist > 0 else "bearish"),
        "macdD_hist": round(md_hist, 3) if md_hist is not None else None,
        "macdW": None if mw_hist is None else ("bullish" if mw_hist > 0 else "bearish"),
        "macdW_hist": round(mw_hist, 3) if mw_hist is not None else None,
        "r1": round(ret(dc, 252), 1) if ret(dc, 252) is not None else None,
        "r3": round(ret(wc, 156), 1) if ret(wc, 156) is not None else None,
        "r5": round(ret(wc, 260), 1) if ret(wc, 260) is not None else None,
        "vol": participation,
        "daily_bars": len(dc), "weekly_bars": len(wc),
    }


# ---------------------------------------------------------------- scoring
# Same model as the published board. Chart structure is the one component a
# script cannot read off the chart, so it is inferred here from the 52-week
# position and MA alignment — a rougher proxy than a human eye on the weekly.

def score(rec: dict) -> tuple[int, str, str]:
    pts = 0
    cmp_ = rec["cmp"]
    if rec["sma200"] and cmp_ > rec["sma200"]:
        pts += 25
    if rec["sma100"] and cmp_ > rec["sma100"]:
        pts += 10
    if rec["sma50"] and cmp_ > rec["sma50"]:
        pts += 10
    pts += {"bullish": 15, "mixed": 7}.get(rec["stack"], 0)

    span = rec["h52"] - rec["l52"]
    pos = (cmp_ - rec["l52"]) / span if span else 0.5
    above200 = bool(rec["sma200"] and cmp_ > rec["sma200"])
    if above200 and pos >= 0.75:
        structure, s_pts = "uptrend", 25
    elif above200 and pos >= 0.45:
        structure, s_pts = "uptrend-consolidating", 18
    elif pos >= 0.35:
        structure, s_pts = "range", 12
    else:
        structure, s_pts = "downtrend", 0
    pts += s_pts

    if rec["macdW"] == "bullish":
        pts += 10
    elif rec["macdW"] is None and rec["macdD"] == "bullish":
        pts += 5
    if rec["r1"] and rec["r1"] > 0:
        pts += 5

    grade = ("Strong uptrend" if pts >= 75 else "Constructive" if pts >= 55
             else "Neutral / range" if pts >= 35 else "Weak" if pts >= 20 else "Downtrend")
    return pts, grade, structure


# ---------------------------------------------------------------- main

def load_csv(path: str) -> list[tuple[str, str, str, str]]:
    """Accepts the NSE constituent file: Company Name,Industry,Symbol,Series,ISIN."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 5 or not row[-1].startswith("INE"):
                continue
            out.append((row[-3].strip(), row[0].strip(), row[-4].strip(), row[-1].strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="constituent CSV (Company,Industry,Symbol,Series,ISIN)")
    ap.add_argument("--out", default="technicals.json")
    ap.add_argument("--asof", help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()

    today = date.fromisoformat(args.asof) if args.asof else date.today()
    universe = load_csv(args.csv) if args.csv else LEADERS
    print(f"{len(universe)} symbols, as of {today}", file=sys.stderr)

    results, failures = [], []
    for i, (symbol, name, sector, isin) in enumerate(universe, 1):
        try:
            rec = analyse(symbol, name, sector, isin, today)
            rec["score"], rec["grade"], rec["struct"] = score(rec)
            results.append(rec)
            print(f"  [{i:>3}/{len(universe)}] {symbol:<12} "
                  f"₹{rec['cmp']:>10,.2f}  RSI-W {rec['rsiW'] or 0:>5}  "
                  f"{rec['score']:>3} {rec['grade']}", file=sys.stderr)
        except Exception as exc:                      # noqa: BLE001
            failures.append({"symbol": symbol, "isin": isin, "error": str(exc)})
            print(f"  [{i:>3}/{len(universe)}] {symbol:<12} FAILED — {exc}", file=sys.stderr)

    results.sort(key=lambda r: -r["score"])
    payload = {
        "generated": today.isoformat(),
        "source": "Upstox /v3/historical-candle (public, unauthenticated)",
        "count": len(results),
        "failures": failures,
        "stocks": results,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\nwrote {args.out} — {len(results)} ok, {len(failures)} failed", file=sys.stderr)
    return 1 if failures and not results else 0


if __name__ == "__main__":
    sys.exit(main())
