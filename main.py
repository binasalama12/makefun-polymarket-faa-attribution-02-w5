#!/usr/bin/env python3
"""Owl forecaster for Polymarket's FAA outage attribution family
(make.fun polymarket-faa-attribution-02).

Owl is the patient entry: it reads the market's entire history in
the case (up to the full 72 hours) and refuses to let a handful of
monster prints set the spread. Every h-minute change that feeds the
empirical quantiles is winsorised — clipped at the 99th percentile
of absolute changes for that horizon — so the tails stay honest
without being hostage to the single wildest minute in three days.

Per market:

- Empirical quantiles of winsorised h-minute changes over the full
  history at anchors 1, 5, 15, 30, 60, 120, 240, interpolated in
  sqrt(h); weight 45%.
- A Gaussian random walk at 55%, sigma from the mean absolute
  one-minute change times 1.2533, floored 0.001, capped 0.005.
- Centre: the last price. No regime factor — with three days of
  history the empirical side has already seen hot and quiet spells.

Deterministic, standard library only; quantiles clamped to [0, 1],
kept non-decreasing, rounded to 6 decimals; comfortably inside the
8 second per-case limit.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

ANCHORS = [1, 5, 15, 30, 60, 120, 240]
EMP_W = 0.45
WINSOR_Q = 0.99
SIG_LO, SIG_HI = 0.001, 0.005
MIN_HIST = 300
HOUSE_SIGMA = 0.0025


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def r6(v):
    return round(clamp(v, 0.0, 1.0) * 1e6) / 1e6 + 0.0


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    pos = tau * (n - 1)
    i = int(math.floor(pos))
    f = pos - i
    j = min(i + 1, n - 1)
    return sorted_vals[i] * (1.0 - f) + sorted_vals[j] * f


def mean_abs_step(prices):
    if len(prices) < 2:
        return 0.0
    return sum(abs(b - a) for a, b in zip(prices, prices[1:])) / (len(prices) - 1)


def sigma_of(prices):
    return clamp(1.2533 * mean_abs_step(prices), SIG_LO, SIG_HI)


def winsorised_quantiles(hist, h):
    """Change quantiles at horizon h with the extremes clipped."""
    n = len(hist)
    diffs = [hist[i + h] - hist[i] for i in range(n - h)]
    abs_sorted = sorted(abs(d) for d in diffs)
    cap = quantile(abs_sorted, WINSOR_Q)
    clipped = sorted(clamp(d, -cap, cap) for d in diffs)
    return [quantile(clipped, tau) for tau in TAUS]


def anchors_of(hist):
    n = len(hist)
    tab = {}
    for h in ANCHORS:
        if h < n:
            tab[h] = winsorised_quantiles(hist, h)
    return tab


def emp_at(tab, h):
    keys = sorted(tab)
    if not keys:
        return None
    if h <= keys[0]:
        return tab[keys[0]]
    if h >= keys[-1]:
        return tab[keys[-1]]
    for a, b in zip(keys, keys[1:]):
        if a <= h <= b:
            f = (math.sqrt(h) - math.sqrt(a)) / (math.sqrt(b) - math.sqrt(a))
            return [tab[a][j] * (1 - f) + tab[b][j] * f for j in range(5)]
    return tab[keys[-1]]


def forecast_market(prices, horizon, sigma):
    x = prices[-1]
    tab = anchors_of(prices) if len(prices) >= MIN_HIST else {}
    rows = []
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        gauss = [x + z * sigma * sq for z in Z]
        emp = emp_at(tab, h)
        if emp is None:
            row = gauss
        else:
            row = [EMP_W * (x + emp[j]) + (1 - EMP_W) * gauss[j]
                   for j in range(5)]
        rows.append(sorted(r6(v) for v in row))
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.iterdir()):
        if not path.name.endswith(".in"):
            continue
        case = json.loads(path.read_text())
        if list(case["quantiles"]) != TAUS:
            raise ValueError(f"{path.name}: unexpected quantiles")
        horizon = int(case["horizon_minutes"])
        markets = [(m["id"], [float(p) for p in m["prices"]])
                   for m in case["markets"]]
        for mid, prices in markets:
            if not prices:
                raise ValueError(f"market {mid} has no prices")
        own = {mid: sigma_of(p) for mid, p in markets if len(p) >= MIN_HIST}
        if own:
            vals = sorted(own.values())
            borrow = vals[len(vals) // 2]
        else:
            borrow = HOUSE_SIGMA
        out = {"forecasts": [
            {"id": mid, "q": forecast_market(p, horizon, own.get(mid, borrow))}
            for mid, p in markets]}
        (OUTPUT_DIR / (path.name[:-3] + ".out")).write_text(
            json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
