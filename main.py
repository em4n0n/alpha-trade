from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from math import erf, exp, log, sqrt
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

NAV = 100_000  # fund net asset value
CAPITAL = 4_000  # deployable capital for this batch
MAX_LOSS_PER_TRADE = NAV * 0.005  # $500
POP_MIN = 0.65
CREDIT_LOSS_RATIO_MIN = 0.33
BASKET_DELTA_RANGE = (-0.30 * NAV / 100_000, 0.30 * NAV / 100_000)
BASKET_VEGA_MIN = -0.05 * NAV / 100_000
SECTOR_LIMIT = 2  # max trades per GICS sector
RISK_FREE = 0.03  # flat 3 % for Black‑Scholes


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Cumulative standard normal using math.erf (no SciPy needed)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_delta(option_type: str, S: float, K: float, T: float, sigma: float, r: float = RISK_FREE) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if option_type == "C":
        return _norm_cdf(d1)
    else:  # put
        return _norm_cdf(d1) - 1.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptionLeg:
    symbol: str
    expiry: datetime
    strike: float
    option_type: str  # "C" or "P"
    direction: str  # "SELL" or "BUY"
    price: float
    delta: float
    iv: float

    def contract_id(self) -> str:
        exp = self.expiry.strftime("%y-%m-%d")
        return f"{self.direction} {exp} {self.strike:.1f}{self.option_type}"


@dataclass
class TradeIdea:
    ticker: str
    strategy: str
    legs: List[OptionLeg]
    pop: float
    credit: float
    max_loss: float
    delta: float
    vega: float
    model_score: float
    sector: str
    thesis: str

    def meets_hard_filters(self) -> bool:
        return (
            self.pop >= POP_MIN
            and self.credit / self.max_loss >= CREDIT_LOSS_RATIO_MIN
            and self.max_loss <= MAX_LOSS_PER_TRADE
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _yf_ticker(ticker: str):
    return yf.Ticker(ticker)


def fetch_sector(ticker: str) -> str:
    info = _yf_ticker(ticker).info
    return info.get("sector", "Unknown")


def fetch_option_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    chain = _yf_ticker(ticker).option_chain(expiry)
    return chain.calls, chain.puts


def calc_pop(credit: float, max_loss: float) -> float:
    """Very rough POP proxy. Replace with full model if available."""
    edge = credit / max_loss
    return max(0.50, min(0.99, 0.50 + edge / 2))


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

def build_iron_condor(
    ticker: str,
    expiry: str,
    call_strikes: Tuple[float, float],
    put_strikes: Tuple[float, float],
) -> TradeIdea | None:
    calls_df, puts_df = fetch_option_chain(ticker, expiry)

    today_utc = datetime.now(timezone.utc)
    exp_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    T = (exp_dt - today_utc).days / 365.0

    spot = _yf_ticker(ticker).history(period="1d")["Close"].iloc[-1]

    def _make_leg(df: pd.DataFrame, strike: float, opt_type: str, sell_strike: float) -> OptionLeg | None:
        row = df.loc[(df["strike"] - strike).abs().idxmin()] if not df.empty else None
        if row is None or pd.isna(row["impliedVolatility"]):
            return None
        iv = row["impliedVolatility"]
        delta_in = row.get("delta")
        if pd.isna(delta_in):
            delta_in = bs_delta(opt_type, spot, strike, T, iv, RISK_FREE)
        mid_price = row["lastPrice"]
        if not pd.isna(row.get("bid")) and not pd.isna(row.get("ask")) and row["bid"] > 0 and row["ask"] > 0:
            mid_price = (row["bid"] + row["ask"]) / 2
        direction = "SELL" if strike == sell_strike else "BUY"
        return OptionLeg(
            symbol=row["contractSymbol"],
            expiry=exp_dt,
            strike=strike,
            option_type=opt_type,
            direction=direction,
            price=float(mid_price),
            delta=float(delta_in),
            iv=float(iv),
        )

    legs = [
        _make_leg(calls_df, call_strikes[0], "C", call_strikes[0]),
        _make_leg(calls_df, call_strikes[1], "C", call_strikes[0]),
        _make_leg(puts_df, put_strikes[0], "P", put_strikes[0]),
        _make_leg(puts_df, put_strikes[1], "P", put_strikes[0]),
    ]
    if any(l is None for l in legs):
        return None

    credit = sum(l.price if l.direction == "SELL" else -l.price for l in legs)
    width_call = abs(call_strikes[1] - call_strikes[0])
    width_put = abs(put_strikes[0] - put_strikes[1])
    max_loss = max(width_call, width_put) - credit
    pop = calc_pop(credit, max_loss)
    delta = sum(l.delta * (1 if l.direction == "SELL" else -1) for l in legs)
    vega = sum(l.iv * (1 if l.direction == "SELL" else -1) for l in legs) / 100

    return TradeIdea(
        ticker=ticker,
        strategy="30‑day iron condor",
        legs=legs,
        pop=pop,
        credit=credit,
        max_loss=max_loss,
        delta=delta,
        vega=vega,
        model_score=pop + credit / max_loss,
        sector=fetch_sector(ticker),
        thesis="Auto‑generated thesis placeholder",
    )


# ---------------------------------------------------------------------------
# Screening engine
# ---------------------------------------------------------------------------

def generate_candidates(tickers: List[str]) -> List[TradeIdea]:
    ideas: List[TradeIdea] = []
    today = datetime.now(timezone.utc).date()
    for tkr in tickers:
        tk = _yf_ticker(tkr)
        expiries = tk.options
        if not expiries:
            continue
        expiry = next(
            (e for e in expiries if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 25),
            None,
        )
        if not expiry:
            continue
        # 1‑y hist vol estimate
        hist = tk.history(period="1y")["Close"]
        if hist.empty:
            continue
        hv = np.std(np.log(hist / hist.shift(1)).dropna()) * np.sqrt(252)
        spot = hist.iloc[-1]
        stdev = spot * hv * sqrt(30 / 365)
        short_call = round(spot + 1.25 * stdev, 0)
        long_call = short_call + 5
        short_put = round(spot - 1.25 * stdev, 0)
        long_put = short_put - 5

        idea = build_iron_condor(tkr, expiry, (short_call, long_call), (short_put, long_put))
        if idea and idea.meets_hard_filters():
            ideas.append(idea)
    return ideas


def basket_ok(trades: List[TradeIdea]) -> bool:
    delta_sum = sum(t.delta for t in trades)
    vega_sum = sum(t.vega for t in trades)
    if not (BASKET_DELTA_RANGE[0] <= delta_sum <= BASKET_DELTA_RANGE[1]):
        return False
    if vega_sum < BASKET_VEGA_MIN:
        return False
    sector_counts: dict[str, int] = {}
    for t in trades:
        sector_counts[t.sector] = sector_counts.get(t.sector, 0) + 1
        if sector_counts[t.sector] > SECTOR_LIMIT:
            return False
    return True


def select_trades(candidates: List[TradeIdea]) -> List[TradeIdea]:
    sorted_cand = sorted(candidates, key=lambda x: x.model_score, reverse=True)
    portfolio: List[TradeIdea] = []
    for trade in sorted_cand:
        tentative = portfolio + [trade]
        if len(tentative) > 5:
            continue
        if basket_ok(tentative):
            portfolio = tentative
        if len(portfolio) == 5:
            break
    return portfolio


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_portfolio(trades: List[TradeIdea]):
    if len(trades) < 5:
        print("Fewer than 5 trades meet criteria, do not execute.")
        return
    rows = []
    for t in trades:
        legs_desc = "; ".join(l.contract_id() for l in t.legs)
        rows.append([t.ticker, t.strategy, legs_desc, t.thesis[:30], f"{t.pop:.2f}"])
    print(tabulate(rows, headers=["Ticker", "Strategy", "Legs", "Thesis (≤30)", "POP"], tablefmt="github"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    universe = [
        "AAPL",
        "XOM",
        "JNJ",
        "BAC",
        "NVDA",
        "MSFT",
        "KO",
        "AMZN",
        "TSLA",
        "META",
    ]
    candidates = generate_candidates(universe)
    portfolio = select_trades(candidates)
    print_portfolio(portfolio)


if __name__ == "__main__":
    main()