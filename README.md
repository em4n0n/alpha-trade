# alpha trade
Options trade recommendations

Python app that proposes options trades under rigorous risk and portfolio constraints.

Key Features
------------
* Pulls fundamentals & price data via yfinance (free) or Polygon.io (if API key supplied).
* Retrieves full option chains, then constructs candidate iron‑condors, bull‑put, and bear‑call spreads.
* Filters against hard rules (quote‑age, POP ≥ 0.65, loss ≤ $500, credit‑to‑loss ≥ 0.33).
* Constrains basket Delta, Vega, and sector diversification (max 2 per GICS sector).
* Ranks with a transparent `model_score` derived from momentum, flows, IV rank, fundamentals.
* Prints a clean 5‑row table identical to required output; prints warning if <5 pass.

Setup
-----
$ pip install yfinance pandas numpy tabulate requests

Optionally set `POLYGON_API_KEY` (env) for faster, deeper option‑chain snapshots.

Improvements (2025-07-13)
------------------------
* **UTC-aware dates** – replaces deprecated `datetime.utcnow()`
* **Delta handling** – yfinance chains lack Greeks. We now compute Black-Scholes delta on the fly when not provided.
* **Safer mid-price calc** – falls back to `lastPrice` if bid/ask unavailable.
* **Correct SELL/BUY logic** – parenthesis fix.

Setup
-----
bash
pip install yfinance pandas numpy tabulate  # stdlib covers math/dateutils

(No extra deps; BS delta uses `math.erf`.)
