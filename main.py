# options_trade_suggester.py
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
        conditions = [
            self.pop >= POP_MIN,
            self.credit / self.max_loss >= CREDIT_LOSS_RATIO_MIN,
            self.max_loss <= MAX_LOSS_PER_TRADE,
        ]
        return all(conditions)


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
    """Crude probability‑of‑profit proxy using risk‑reward; replace with model if available."""
    edge = credit / max_loss
    return max(0.50, min(0.99, 0.50 + edge / 2))


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

def build_iron_condor(ticker: str, expiry: str, call_strikes: Tuple[float, float], put_strikes: Tuple[float, float]) -> TradeIdea | None:
    calls_df, puts_df = fetch_option_chain(ticker, expiry)

    def _row(df: pd.DataFrame, strike: float, opt_type: str):
        row = df.loc[np.isclose(df["strike"], strike)]
        if row.empty:
            return None
        r = row.iloc[0]
        return OptionLeg(
            symbol=f"{ticker}{expiry}{strike}{opt_type}",
            expiry=datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc),
            strike=strike,
            option_type=opt_type,
            direction="SELL" if opt_type == "C" and strike == call_strikes[0] or opt_type == "P" and strike == put_strikes[0] else "BUY",
            price=(r["bid"] + r["ask"]) / 2,
            delta=r["delta"],
            iv=r["impliedVolatility"],
        )

    legs = [
        _row(calls_df, call_strikes[0], "C"),
        _row(calls_df, call_strikes[1], "C"),
        _row(puts_df, put_strikes[0], "P"),
        _row(puts_df, put_strikes[1], "P"),
    ]

    if any(l is None for l in legs):
        return None

    credit = sum(l.price * (1 if l.direction == "SELL" else -1) for l in legs)
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
        model_score=pop + credit / max_loss,  # simple proxy
        sector=fetch_sector(ticker),
        thesis="Auto‑generated thesis placeholder",
    )


# ---------------------------------------------------------------------------
# Screening engine
# ---------------------------------------------------------------------------

def generate_candidates(tickers: List[str]) -> List[TradeIdea]:
    ideas: List[TradeIdea] = []
    today = datetime.utcnow().date()
    # choose nearest monthly expiry ≥30 d
    for tkr in tickers:
        tk = _yf_ticker(tkr)
        expiries = tk.options
        if not expiries:
            continue
        expiry = next((e for e in expiries if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 25), None)
        if not expiry:
            continue

        # crude strike selection: ± 1.25 std‑dev for shorts, ± 1.5 for longs
        hist = tk.history(period="1y")["Close"]
        hv = np.std(np.log(hist / hist.shift(1)).dropna()) * np.sqrt(252)
        spot = hist.iloc[-1]
        stdev = spot * hv * np.sqrt(30 / 365)
        short_call = round(spot + 1.25 * stdev, 0)
        long_call = short_call + 5
        short_put = round(spot - 1.25 * stdev, 0)
        long_put = short_put - 5

        idea = build_iron_condor(tkr, expiry, (short_call, long_call), (short_put, long_put))
        if idea and idea.meets_hard_filters():
            ideas.append(idea)
    return ideas


def basket_ok(trades: List[TradeIdea]) -> bool:
    delta = sum(t.delta for t in trades)
    vega = sum(t.vega for t in trades)
    if not (BASKET_DELTA_RANGE[0] <= delta <= BASKET_DELTA_RANGE[1]):
        return False
    if vega < BASKET_VEGA_MIN:
        return False
    # sector diversification
    sector_counts: dict[str, int] = {}
    for t in trades:
        sector_counts[t.sector] = sector_counts.get(t.sector, 0) + 1
        if sector_counts[t.sector] > SECTOR_LIMIT:
            return False
    return True


def select_trades(candidates: List[TradeIdea]) -> List[TradeIdea]:
    # sort by model_score descending
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
    print(tabulate(rows, headers=["Ticker", "Strategy", "Legs", "Thesis (≤30 words)", "POP"], tablefmt="github"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    universe = ["AAPL", "XOM", "JNJ", "BAC", "NVDA", "MSFT", "KO", "AMZN", "TSLA", "META"]
    candidates = generate_candidates(universe)
    portfolio = select_trades(candidates)
    print_portfolio(portfolio)


if __name__ == "__main__":
    main()
