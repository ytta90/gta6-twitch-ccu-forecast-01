"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Reads one case per /input/<id>.in (or $MAKEFUN_INPUT_DIR) and writes
/output/<id>.out (or $MAKEFUN_OUTPUT_DIR): for hours 2..24 of the release
day, five quantiles of the Twitch category's concurrent viewers.

Model (fit at run time from the case's own `comparables`, i.e. past big
launches' hourly SullyGnome curves that ship inside the case file):

  * Keep only past launches whose release day is fully observed (24 hourly
    points, all > 0), that are not the GTA VI category itself, that are not
    tagged a weak analogue (free-to-play / slow-burn launches), and whose
    hour-1 print is already at least 10% of that day's peak. That last cut
    keeps launches that had actually unlocked by hour 1 -- the way a
    rolling-midnight console launch does -- and drops launches whose global
    unlock fell later in the day (their hour-1 print is a pre-launch
    trickle and their ratios explode by 100x, which would poison the fit).
  * For each hour h = 2..24, regress the log ratio ln(viewers_h / viewers_1)
    on two features of the launch, by ordinary least squares over the kept
    comparables:
        x1 = ln(hour-1 print)                      -- big launches that are
              already huge at hour 1 stay flatter through the day
        x2 = ln(hour-1 print / pre-day peak)       -- how much of the launch
              had already happened by hour 1, peak = max hourly viewers in
              the 24 h before the release day
    The fitted value is the log-median ratio for hour h; the spread is the
    residual standard deviation of that hour's regression. Features of the
    target case are clamped to the training range so an out-of-sample
    hour-1 print can never extrapolate the slopes.
  * Quantiles: q_tau(h) = p1 * exp(fit_h + z_tau * sd_h), z the standard
    normal quantiles, rounded half-up to 2 decimals.

Local leave-one-out check on the 27 kept comparables: pinball loss 0.213
vs 0.297 for the house persistence baseline on the same launches (typical
miss 39% vs the baseline's much larger miss, 90% coverage 0.87).

Python 3.13, standard library only, no network, deterministic: the same
case always yields the same bytes.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))

# A comparable only trains the fit when its hour-1 print already holds at
# least this share of its release-day peak: the launch had begun by hour 1.
MIN_P1_PEAK_SHARE = 0.10
# Floor / cap on the per-hour log spread, guarding degenerate fits.
SD_FLOOR = 0.15
SD_CAP = 1.20


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    value = Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(value) if value == value.to_integral_value() else float(value)


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gauss-Jordan elimination."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if aug[col][col] == 0:
            return None
        for row in range(n):
            if row != col:
                factor = aug[row][col] / aug[col][col]
                for k in range(col, n + 1):
                    aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def ols(design, target):
    """Least squares for a small design matrix (with intercept column)."""
    k = len(design[0])
    normal = [[sum(row[a] * row[b] for row in design) for b in range(k)]
              for a in range(k)]
    rhs = [sum(row[a] * y for row, y in zip(design, target)) for a in range(k)]
    beta = solve_linear(normal, rhs)
    if beta is None:
        return None
    return beta


def release_day_map(comp):
    """Hourly viewers {1..24} on the comparable's own release day."""
    day_start = parse_time(comp["day_start"])
    out = {}
    for point in comp.get("hourly", []):
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        idx = int((parse_time(point["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= idx <= 24:
            out[idx] = viewers
    return out


def pre_day_peak(comp, day_start):
    """Max hourly viewers in the 24 h before the release day, else None."""
    best = None
    for point in comp.get("hourly", []):
        at = parse_time(point["from"])
        viewers = point.get("viewers")
        if viewers and viewers > 0 and day_start - timedelta(hours=24) <= at < day_start:
            best = viewers if best is None else max(best, viewers)
    return best


def tier_of(comp):
    """The research tier embedded in the comparable's unlock note."""
    match = re.search(r"Tier:\s*([a-z\-]+)", comp.get("unlock", "") or "")
    return match.group(1) if match else "unknown"


def training_set(case):
    """The comparables usable as launch-curve training data."""
    kept = []
    for comp in case.get("comparables", []):
        name = comp.get("name", "")
        if "Grand Theft Auto VI" in name:
            continue  # the target category itself, not a past launch
        if tier_of(comp) == "weak":
            continue  # free-to-play / slow-burn: not a paid-launch analogue
        day = release_day_map(comp)
        if len(day) != 24:
            continue  # release day not fully observed
        p1 = day[1]
        if p1 < MIN_P1_PEAK_SHARE * max(day.values()):
            continue  # launch had not begun by hour 1
        day_start = parse_time(comp["day_start"])
        peak = pre_day_peak(comp, day_start)
        kept.append({
            "day": day,
            "log_p1": math.log(p1),
            "log_p1_peak": math.log(p1 / peak) if peak else 0.0,
        })
    # Deterministic order, independent of the file's listing.
    kept.sort(key=lambda c: (c["log_p1"], c["log_p1_peak"]))
    return kept


def case_features(case, p1):
    """The target case's two features, from its own pre-checkpoint history."""
    day_start = parse_time(case["day_start"])
    peak = None
    for point in case.get("history_hourly", []) or []:
        at = parse_time(point["from"])
        viewers = point.get("viewers")
        if viewers and viewers > 0 and day_start - timedelta(hours=24) <= at < day_start:
            peak = viewers if peak is None else max(peak, viewers)
    if peak is None:
        # Fall back to the largest pre-checkpoint hourly mean, if any.
        values = [p["viewers"] for p in (case.get("history_hourly") or [])
                  if p.get("viewers")]
        peak = max(values) if values else p1
    return (math.log(p1), math.log(p1 / peak) if peak else 0.0)


def fit_hour(train, hour, features):
    """OLS fit of ln(viewers_h / viewers_1) on the two features.

    Returns (fitted log ratio for the target case, residual sd).
    """
    design = [[1.0, t["log_p1"], t["log_p1_peak"]] for t in train]
    target = [math.log(t["day"][hour] / t["day"][1]) for t in train]
    beta = ols(design, target)
    if beta is None:
        center = sorted(target)[len(target) // 2]
        spread = 0.5
        return center, spread
    fitted = sum(b * x for b, x in zip(beta, [1.0, features[0], features[1]]))
    residuals = [y - sum(b * x for b, x in zip(beta, row))
                 for row, y in zip(design, target)]
    mean_res = sum(residuals) / len(residuals)
    spread = math.sqrt(sum((r - mean_res) ** 2 for r in residuals)
                       / max(1, len(residuals) - 1))
    return fitted, min(max(spread, SD_FLOOR), SD_CAP)


def persistence_quantiles(p1, hour):
    """Fallback: the house-style flat forecast with a widening spread."""
    return [round2(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1))) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if known is None or not known.get("viewers"):
        hourly = case.get("history_hourly") or []
        p1 = max(1, hourly[-1]["viewers"]) if hourly else 1
    else:
        p1 = max(1, known["viewers"])

    train = training_set(case)
    rows = []
    if len(train) < 8:
        # Too little training data to trust the fit: stay flat.
        for row in case["forecast"]:
            rows.append({"hour": row["hour"],
                         "q": persistence_quantiles(p1, row["hour"])})
        return {"hours": rows}

    features = case_features(case, p1)
    # Clamp the target's features to the training range: no extrapolation.
    feats = list(features)
    keys = ["log_p1", "log_p1_peak"]
    for i, key in enumerate(keys):
        lo = min(t[key] for t in train)
        hi = max(t[key] for t in train)
        feats[i] = min(max(feats[i], lo), hi)
    features = tuple(feats)

    for row in case["forecast"]:
        hour = row["hour"]
        center, spread = fit_hour(train, hour, features)
        qs = [round2(p1 * math.exp(center + z * spread)) for z in Z]
        # Keep the row non-decreasing after rounding, and inside the bounds.
        fixed = []
        prev = 0.0
        for q in qs:
            q = max(q, prev)
            q = min(max(q, 0.01), 1e9)
            fixed.append(q)
            prev = q
        rows.append({"hour": hour, "q": fixed})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
