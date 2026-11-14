# makefun-polymarket-faa-attribution-02-w5

A forecaster for Polymarket's FAA outage attribution markets (Iran /
foreign actor / hack / AI sandbox escape). `main.py` (Python 3.13,
standard library only) reads one case from the input directory —
every market of the family with its Yes price at each minute up to a
checkpoint — and writes, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of that
minute's price to the output directory.

## Model — Owl full-history winsorised forecaster

emp 0.45, anchors 1/5/15/30/60/120/240, window full 4320, changes winsorised at 99th pct of |change|, mean-abs sigma floor 0.001 cap 0.005, no regime

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. The program is deterministic and runs in well under the
8 second per-case limit. A market with too little history of its own
borrows the median sigma of the case's established markets.

Backtested on the kit's practice inputs (the NVIDIA CEO practice
cases; later cases used as ground truth for earlier checkpoints),
this variant scores a mean pinball loss of about 0.3895 against the
house baseline's 0.3985 on the same folds. The practice is a
different event, so the number is a sanity check, not a prediction.

Entered on make.fun's polymarket-faa-attribution-02 bounty.
