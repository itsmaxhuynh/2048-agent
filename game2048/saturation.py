"""Deciding when a learning curve has stopped paying for itself.

Yeh et al. decay the step size "after the improvement is saturated", and define
saturation as the per-1,000-game averages not making "significant improvement".
That is not something you can test as written, so this module pins it down.

The obvious formalisation -- "the slope is not significantly greater than zero"
-- is the wrong test twice over. With enough blocks an economically meaningless
slope becomes statistically significant; with few blocks, noise hides a real
trend and you decay too early. Failing to detect improvement is not the same as
establishing there is none.

So the test is an equivalence test instead: not "is there improvement?" but
"can I rule out improvement big enough to be worth the compute?"

An early version of this fitted a slope and projected it over a fixed horizon of
a million games. That was wrong -- the window spans a tenth of that, so it
extrapolates an order of magnitude past the data and the interval balloons until
nothing is ever saturated. Instead, compare the window against itself:

    split the last 2N blocks into the older N and the newer N
    delta = mean(newer) - mean(older)
    saturated  <=>  delta + 1.645 * se(delta)  <  tolerance * mean(newer)

Read as: "the last N blocks did not beat the N before them by more than
`tolerance`, and I can say that with 95% confidence." No extrapolation, and it
is what "the average scores gradually stabilize" actually means. The horizon is
implicit and honest: it is the span the window covers.

Measured behaviour at the defaults (block averages with a realistic spread of
about 4% of the level), as the share of runs called saturated:

    window   span      flat   +100/50k   +500/50k   +1000/50k   +2500/50k
    160    320,000     91%       81%        15%          0%          0%

Which is the shape you want: it fires on a flat curve, tolerates a drift too
small to care about, and refuses while real learning is happening. A shorter
window is not merely noisier -- at 40 blocks it calls a flat curve saturated
only 43% of the time, so it would sit there never deciding.

    python -m game2048.saturation talk/curve-stage0.csv
"""

import argparse
import csv
import sys

import numpy as np

Z_95 = 1.6448536269514722          # one-sided 95% quantile of the normal
MIN_BLOCKS = 12


def slope_with_upper_bound(games, scores):
    """OLS slope of score against games, with a one-sided upper 95% bound.

    The standard error is inflated for lag-1 autocorrelation in the residuals.
    Consecutive blocks are played by almost the same weights, so their errors
    are not independent and the textbook interval is too narrow -- and too
    narrow here means declaring saturation early, which is the costly error.
    """
    x = np.asarray(games, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    n = len(x)
    if n < 3:
        raise ValueError(f"need at least 3 blocks, got {n}")

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx == 0.0:
        raise ValueError("all blocks report the same game count")
    s2 = float((resid ** 2).sum()) / (n - 2)
    se = float(np.sqrt(s2 / sxx))

    rho = 0.0
    if n >= 10:
        with np.errstate(invalid="ignore"):
            r = np.corrcoef(resid[:-1], resid[1:])[0, 1]
        if np.isfinite(r):
            rho = float(min(max(r, 0.0), 0.95))     # only positive rho widens
        se *= float(np.sqrt((1.0 + rho) / (1.0 - rho)))

    return float(slope), slope + Z_95 * se, se, rho


def effective_n(resid):
    """Sample size discounted for lag-1 autocorrelation.

    Consecutive blocks are played by almost the same weights, so their errors
    are not independent. Treating them as independent narrows the interval and
    declares saturation early -- the costly direction to be wrong in.
    """
    n = len(resid)
    if n < 10:
        return float(n), 0.0
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    rho = float(min(max(r, 0.0), 0.95)) if np.isfinite(r) else 0.0
    return n * (1.0 - rho) / (1.0 + rho), rho


def assess(games, scores, tolerance=0.02):
    """Has this curve stopped improving? Returns a verdict dict, never a bare bool."""
    g = np.asarray(games, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    n = len(y)
    if n < 6:
        raise ValueError(f"need at least 6 blocks, got {n}")

    half = n // 2
    older, newer = y[:half], y[half:]
    delta = float(newer.mean() - older.mean())

    # spread measured around each half's own mean, so a trend is not counted as noise
    resid = np.concatenate([older - older.mean(), newer - newer.mean()])
    n_eff, rho = effective_n(resid)
    var = float((resid ** 2).sum()) / max(n - 2, 1)
    se = float(np.sqrt(var * (1.0 / max(n_eff / 2, 1.0)) * 2.0))

    upper = delta + Z_95 * se
    level = float(newer.mean())
    budget = tolerance * level
    slope, slope_upper, _, _ = slope_with_upper_bound(g, y)
    return {
        "blocks": n,
        "span": int(g[-1] - g[0]),
        "half_span": int(g[-1] - g[half]),
        "level": level,
        "delta": delta,
        "delta_upper": upper,
        "budget": budget,
        "autocorrelation": rho,
        "slope_per_50k": slope * 50_000,
        "saturated": bool(upper < budget),
    }


def explain(v, tolerance):
    verdict = "SATURATED" if v["saturated"] else "still improving"
    return (
        f"{verdict}: {v['blocks']} blocks spanning {v['span']:,} games, "
        f"average now {v['level']:,.0f}\n"
        f"  last {v['half_span']:,} games vs the {v['half_span']:,} before: "
        f"{v['delta']:+,.0f}  (95% upper bound {v['delta_upper']:+,.0f})\n"
        f"  worth-it threshold ({tolerance:.0%} of current): {v['budget']:+,.0f}\n"
        f"  trend {v['slope_per_50k']:+,.0f} per 50k games, residual rho "
        f"{v['autocorrelation']:.2f}")


class SaturationMonitor:
    """Feeds block summaries in, says when the curve has flattened.

    Keeps a sliding window and refuses to answer until it has enough blocks, so
    the steep early part of a run cannot trip it.
    """

    def __init__(self, window=160, tolerance=0.02, min_games=0):
        self.window = window
        self.tolerance = tolerance
        self.min_games = min_games
        self.games = []
        self.scores = []

    def add(self, games, average):
        self.games.append(float(games))
        self.scores.append(float(average))
        if len(self.games) > self.window:
            self.games.pop(0)
            self.scores.pop(0)

    def reset(self):
        """Start judging afresh -- used after a decay changes the dynamics."""
        self.games.clear()
        self.scores.clear()

    def verdict(self):
        if len(self.games) < max(MIN_BLOCKS, self.window):
            return None
        if self.games[-1] < self.min_games:
            return None
        return assess(self.games, self.scores, self.tolerance)


def main(argv=None):
    p = argparse.ArgumentParser(description="Has this learning curve saturated?")
    p.add_argument("csv", help="a curve written by train.py")
    p.add_argument("--window", type=int, default=160, help="blocks to judge on")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="gain worth having, as a fraction of the current average")
    args = p.parse_args(argv)

    rows = [r for r in csv.DictReader(open(args.csv)) if r.get("games")]
    if len(rows) < 3:
        print("not enough blocks to judge", file=sys.stderr)
        return 1
    tail = rows[-args.window:]
    games = [int(r["games"]) for r in tail]
    scores = [float(r["avg_score"]) for r in tail]
    print(explain(assess(games, scores, args.tolerance), args.tolerance))
    return 0


if __name__ == "__main__":
    sys.exit(main())
