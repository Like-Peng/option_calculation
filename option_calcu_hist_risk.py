"""Practical convertible-bond relative valuation.

This module implements the method disclosed in the CICC article
"Convertible-bond 'undervaluation' strategy optimization and Python
implementation" (2021-04-19):

1. Standardize bond/underlying-stock characteristics.
2. Find each bond's nearest comparable bonds (KNN-like Euclidean distance).
3. Fit the local parity-premium curve ``p = a + b*x + c/x``.
4. Convert the fitted premium into a fair bond price and rank the price gaps.
5. Rank today's residual inside the bond's own last 60 / 120 trading days.
6. Screen out ST / delisting / below-floor bonds and print risk warnings from
   each bond's own price, bond-floor and stock-price history.

The implementation deliberately uses only the Python standard library.  It is
therefore directly runnable without pandas, NumPy, scikit-learn, BSM, or a
binomial-tree model.

Rates named ``*_pct`` are percentage points: ``12.5`` means 12.5%.

Usage (two input files: today's quotes and the history of daily quotes)::

    python3 option_calcu_hist_risk.py --input today_withST.csv --date 2026-10-23 \\
        --history-data history_withST.csv --exclude-risky

    python3 option_calcu_hist_risk.py --self-check     # built-in checks
    python3 option_calcu_hist_risk.py                  # one-day synthetic example
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import os
import random
import tempfile
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Number = int | float
Record = Mapping[str, object]


def conversion_value(
    stock_price: Number,
    conversion_price: Number,
    face_value: Number = 100.0,
) -> float:
    """Return parity/conversion value: face value / conversion price * stock price."""
    stock_price = float(stock_price)
    conversion_price = float(conversion_price)
    face_value = float(face_value)
    if stock_price < 0:
        raise ValueError("stock_price must be non-negative")
    if conversion_price <= 0 or face_value <= 0:
        raise ValueError("conversion_price and face_value must be positive")
    return face_value * stock_price / conversion_price


def conversion_premium_pct(bond_price: Number, parity: Number) -> float:
    """Return the conversion premium in percentage points."""
    bond_price = float(bond_price)
    parity = float(parity)
    if bond_price < 0:
        raise ValueError("bond_price must be non-negative")
    if parity <= 0:
        raise ValueError("parity must be positive")
    return (bond_price / parity - 1.0) * 100.0


def price_from_premium(parity: Number, premium_pct: Number) -> float:
    """Convert parity and a premium percentage into a bond price."""
    parity = float(parity)
    if parity <= 0:
        raise ValueError("parity must be positive")
    return parity * (1.0 + float(premium_pct) / 100.0)


def _number(record: Record, key: str) -> float:
    """Read a finite numeric field and raise an actionable error if invalid."""
    if key not in record:
        raise ValueError(f"record {record.get('code', '<unknown>')!r} lacks {key!r}")
    try:
        value = float(record[key])  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"record {record.get('code', '<unknown>')!r}: {key!r} must be numeric"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"record {record.get('code', '<unknown>')!r}: {key!r} must be finite"
        )
    return value


def _code(record: Record) -> str:
    code = str(record.get("code", "")).strip()
    if not code:
        raise ValueError("every record needs a non-empty 'code'")
    return code


def enrich_market_fields(records: Iterable[Record]) -> List[Dict[str, object]]:
    """Fill ``conv`` and/or ``prem`` from raw market fields when necessary.

    Accepted inputs:
    - ``conv``: parity, or ``stock_price`` plus ``conversion_price``;
    - ``prem``: conversion premium in percentage points, or ``close`` plus parity.

    Existing values are retained, so callers can also pass vendor-calculated data.
    """
    enriched: List[Dict[str, object]] = []
    seen: set[str] = set()
    for source in records:
        row = dict(source)
        code = _code(row)
        if code in seen:
            raise ValueError(f"duplicate code: {code}")
        seen.add(code)

        if "conv" not in row:
            row["conv"] = conversion_value(
                _number(row, "stock_price"), _number(row, "conversion_price")
            )
        conv = _number(row, "conv")
        if conv <= 0:
            raise ValueError(f"record {code!r}: 'conv' must be positive")

        if "prem" not in row:
            row["prem"] = conversion_premium_pct(_number(row, "close"), conv)
        _number(row, "prem")

        if "close" not in row:
            row["close"] = price_from_premium(conv, _number(row, "prem"))
        if _number(row, "close") < 0:
            raise ValueError(f"record {code!r}: 'close' must be non-negative")
        enriched.append(row)
    return enriched


def _prepared_feature_matrix(
    records: Sequence[Record],
    feature_names: Sequence[str],
    log_features: Sequence[str],
) -> Tuple[List[str], List[List[float]]]:
    if not records:
        raise ValueError("records must not be empty")
    if not feature_names:
        raise ValueError("feature_names must not be empty")

    log_set = set(log_features)
    codes = [_code(record) for record in records]
    if len(codes) != len(set(codes)):
        raise ValueError("record codes must be unique")

    raw_columns: List[List[float]] = []
    for feature in feature_names:
        column = [_number(record, feature) for record in records]
        if feature in log_set:
            if any(value <= 0 for value in column):
                raise ValueError(f"log feature {feature!r} must be positive")
            column = [math.log(value) for value in column]
        raw_columns.append(column)

    standardized_columns: List[List[float]] = []
    for feature, column in zip(feature_names, raw_columns):
        mean = fmean(column)
        std = pstdev(column)
        if std == 0:
            raise ValueError(f"feature {feature!r} has zero variance")
        standardized_columns.append([(value - mean) / std for value in column])

    matrix = [list(row) for row in zip(*standardized_columns)]
    return codes, matrix


def article_neighbor_count(sample_size: int) -> int:
    """CICC's disclosed rule: min(max(6, floor(n/10)), floor(n/2), 18).

    The return value is capped at ``n - 1`` because the target itself is not a
    comparable.  At least three comparables are needed for the 3-parameter fit.
    """
    if sample_size < 4:
        raise ValueError("at least four bonds are required")
    proposed = min(max(6, math.floor(sample_size / 10)), math.floor(sample_size / 2), 18)
    return min(max(3, proposed), sample_size - 1)


def neighbors_cb_groups(
    records: Sequence[Record],
    feature_names: Sequence[str],
    *,
    log_features: Sequence[str] = ("issue_size",),
    neighbor_count: int | None = None,
) -> Dict[str, List[str]]:
    """Find each bond's nearest peers using squared standardized distance.

    ``feature_names`` should describe bond/stock attributes, such as issue size,
    analyst coverage, growth, and momentum.  Do not include ``conv`` or ``prem``:
    those are used later to fit the local valuation curve.
    """
    codes, matrix = _prepared_feature_matrix(records, feature_names, log_features)
    count = article_neighbor_count(len(records)) if neighbor_count is None else neighbor_count
    if not 3 <= count < len(records):
        raise ValueError("neighbor_count must be at least 3 and smaller than sample size")

    result: Dict[str, List[str]] = {}
    for i, code in enumerate(codes):
        distances = []
        for j, peer_code in enumerate(codes):
            if i == j:
                continue
            distance = sum((left - right) ** 2 for left, right in zip(matrix[i], matrix[j]))
            distances.append((distance, peer_code))
        distances.sort(key=lambda item: (item[0], item[1]))
        result[code] = [peer_code for _, peer_code in distances[:count]]
    return result


@dataclass(frozen=True)
class PremiumCurve:
    """Coefficients for ``premium_pct = a + b*conv + c/conv``."""

    a: float
    b: float
    c: float

    def predict_premium_pct(self, conv: Number) -> float:
        conv = float(conv)
        if conv <= 0:
            raise ValueError("conv must be positive")
        return self.a + self.b * conv + self.c / conv

    def predict_price(self, conv: Number) -> float:
        return price_from_premium(conv, self.predict_premium_pct(conv))


def _solve_3x3(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Solve a 3x3 linear system with partial pivoting."""
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("peer parity values do not contain enough variation for a curve fit")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def fit_premium_curve(peer_records: Sequence[Record]) -> PremiumCurve:
    """OLS fit of the article's disclosed ``a + b*conv + c/conv`` curve.

    Internally, parity is scaled by 100 to improve numerical conditioning.  The
    returned coefficients are converted back to the article's original basis.
    """
    if len(peer_records) < 3:
        raise ValueError("at least three peer bonds are required")

    design: List[List[float]] = []
    target: List[float] = []
    for record in peer_records:
        conv = _number(record, "conv")
        if conv <= 0:
            raise ValueError("peer 'conv' values must be positive")
        scaled = conv / 100.0
        design.append([1.0, scaled, 1.0 / scaled])
        target.append(_number(record, "prem"))

    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for row, y in zip(design, target):
        for i in range(3):
            xty[i] += row[i] * y
            for j in range(3):
                xtx[i][j] += row[i] * row[j]

    a, scaled_b, scaled_c = _solve_3x3(xtx, xty)
    return PremiumCurve(a=a, b=scaled_b / 100.0, c=scaled_c * 100.0)


@dataclass(frozen=True)
class ValuationResult:
    code: str
    parity: float
    actual_premium_pct: float
    fitted_premium_pct: float
    fair_value: float
    close: float
    diff: float
    diff_pct: float
    z_score: float = 0.0
    # actual premium - fitted premium, in percentage points.  Unlike ``diff``
    # (yuan) it does not grow with the price level, so it is the quantity
    # stored in the residual history.
    residual_pct: float = 0.0


def group_valuation(
    records: Sequence[Record],
    feature_names: Sequence[str],
    *,
    log_features: Sequence[str] = ("issue_size",),
    neighbor_count: int | None = None,
) -> List[ValuationResult]:
    """Value every bond against its nearest comparable bonds.

    ``diff = close - fair_value`` follows the article.  More negative ``diff``
    and ``z_score`` values indicate greater relative undervaluation.
    """
    rows = enrich_market_fields(records)
    groups = neighbors_cb_groups(
        rows,
        feature_names,
        log_features=log_features,
        neighbor_count=neighbor_count,
    )
    by_code = {_code(row): row for row in rows}
    provisional: List[ValuationResult] = []
    for row in rows:
        code = _code(row)
        curve = fit_premium_curve([by_code[peer] for peer in groups[code]])
        parity = _number(row, "conv")
        fair_value = curve.predict_price(parity)
        close = _number(row, "close")
        diff = close - fair_value
        actual_premium = _number(row, "prem")
        fitted_premium = curve.predict_premium_pct(parity)
        provisional.append(
            ValuationResult(
                code=code,
                parity=parity,
                actual_premium_pct=actual_premium,
                fitted_premium_pct=fitted_premium,
                fair_value=fair_value,
                close=close,
                diff=diff,
                diff_pct=diff / fair_value * 100.0,
                residual_pct=actual_premium - fitted_premium,
            )
        )

    diffs = [result.diff for result in provisional]
    mean_diff = fmean(diffs)
    std_diff = pstdev(diffs)
    results = [
        ValuationResult(
            **{
                **result.__dict__,
                "z_score": 0.0 if std_diff == 0 else (result.diff - mean_diff) / std_diff,
            }
        )
        for result in provisional
    ]
    return sorted(results, key=lambda result: (result.z_score, result.diff, result.code))


# Compatibility names corresponding to the article's published snippets.
def neighborsCbGroups(  # noqa: N802
    records: Sequence[Record],
    feature_names: Sequence[str],
    *,
    log_features: Sequence[str] = ("issue_size",),
    neighbor_count: int | None = None,
) -> Dict[str, List[str]]:
    return neighbors_cb_groups(
        records,
        feature_names,
        log_features=log_features,
        neighbor_count=neighbor_count,
    )


def olsForGroup(  # noqa: N802
    peer_records: Sequence[Record], conv: Number
) -> Tuple[float, Tuple[float, float, float]]:
    curve = fit_premium_curve(peer_records)
    return curve.predict_price(conv), (curve.a, curve.b, curve.c)


class GroupValuation:
    """Small object-oriented wrapper analogous to CICC's ``groupValuation``."""

    def __init__(
        self,
        records: Sequence[Record],
        feature_names: Sequence[str],
        *,
        log_features: Sequence[str] = ("issue_size",),
        neighbor_count: int | None = None,
    ) -> None:
        self.records = list(records)
        self.feature_names = tuple(feature_names)
        self.log_features = tuple(log_features)
        self.neighbor_count = neighbor_count
        self.ret = group_valuation(
            self.records,
            self.feature_names,
            log_features=self.log_features,
            neighbor_count=self.neighbor_count,
        )


groupValuation = GroupValuation  # article-compatible class alias


# ---------------------------------------------------------------------------
# Residual history
#
# Cross-section first, history second: each day the bond is valued against its
# peers (which removes the market-wide valuation level and the parity effect),
# and only that residual is compared with the bond's own past residuals (which
# removes the bond's persistent premium or discount).
# ---------------------------------------------------------------------------

HISTORY_FIELDS = (
    "date",
    "code",
    "parity",
    "actual_premium_pct",
    "fitted_premium_pct",
    "residual_pct",
    "diff",
    "diff_pct",
    "z_score",
)


def _iso_date(value: object) -> str:
    """Normalise a trade date to ``YYYY-MM-DD``; ISO strings sort by date."""
    if isinstance(value, _dt.date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return _dt.date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"trade date must look like 2026-10-23, got {value!r}") from exc


def _upsert_by_date(
    path: str, fields: Sequence[str], day: str, new_rows: Sequence[Dict[str, str]]
) -> int:
    """Replace every row dated ``day`` in a CSV history with ``new_rows``.

    Re-running a day is therefore safe.  The file is written to a temporary
    file first and then renamed, so an interrupted run never leaves a
    half-written history.  Returns ``len(new_rows)``.
    """
    kept: List[Dict[str, str]] = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(fields) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} lacks columns: {sorted(missing)}")
            kept = [
                {field: row[field] for field in fields}
                for row in reader
                if row["date"] != day
            ]
    rows = sorted(list(kept) + list(new_rows), key=lambda row: (row["date"], row["code"]))
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".history-", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return len(new_rows)


def save_residuals(
    results: Sequence[ValuationResult], trade_date: object, path: str
) -> int:
    """Store one day's valuation results in the history CSV.

    Rows already stored for the same date are replaced, so re-running a day is
    safe.  Returns the number of rows written for ``trade_date``.
    """
    day = _iso_date(trade_date)
    new_rows = [
        {
            "date": day,
            "code": r.code,
            "parity": f"{r.parity:.6f}",
            "actual_premium_pct": f"{r.actual_premium_pct:.6f}",
            "fitted_premium_pct": f"{r.fitted_premium_pct:.6f}",
            "residual_pct": f"{r.residual_pct:.6f}",
            "diff": f"{r.diff:.6f}",
            "diff_pct": f"{r.diff_pct:.6f}",
            "z_score": f"{r.z_score:.6f}",
        }
        for r in results
    ]
    return _upsert_by_date(path, HISTORY_FIELDS, day, new_rows)


def load_residual_history(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Return ``{code: [(date, residual_pct), ...]}`` sorted by date.

    A missing file means no history yet and returns an empty dict.
    """
    history: Dict[str, List[Tuple[str, float]]] = {}
    if not os.path.exists(path):
        return history
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            history.setdefault(row["code"], []).append(
                (row["date"], float(row["residual_pct"]))
            )
    for series in history.values():
        series.sort()
    return history


def percentile_rank(past: Sequence[float], value: float) -> float:
    """Percent of ``past`` below ``value``; ties count half.

    0 means today's residual is lower than every past day in the window, i.e.
    the bond is the cheapest it has been versus its own norm; 100 means the
    most expensive.
    """
    if not past:
        raise ValueError("past must not be empty")
    below = equal = 0
    for item in past:
        # History is stored with 6 decimals, so values within 1e-6 count as ties.
        if math.isclose(item, value, abs_tol=1e-6):
            equal += 1
        elif item < value:
            below += 1
    return 100.0 * (below + 0.5 * equal) / len(past)


@dataclass(frozen=True)
class SelfHistoryStat:
    """Today's residual measured against the bond's own past residuals."""

    window: int
    n_obs: int  # past days actually available inside the window
    percentile: Optional[float]  # None when n_obs < min_obs
    mean: Optional[float]  # the bond's normal residual over the window
    self_z: Optional[float]  # (today - mean) / std over the window


def self_history_stat(
    series: Sequence[Tuple[str, float]],
    trade_date: object,
    today_residual: float,
    window: int,
    min_obs: int,
) -> SelfHistoryStat:
    """Rank ``today_residual`` inside the last ``window`` stored days before ``trade_date``.

    The window counts the bond's own stored trading days, so days it was not
    valued (suspension, not yet listed) are skipped rather than left empty.
    Days on or after ``trade_date`` are never used, so re-running an old date
    does not look into the future.
    """
    day = _iso_date(trade_date)
    past = [value for date, value in series if date < day][-window:]
    if len(past) < min_obs:
        return SelfHistoryStat(window, len(past), None, None, None)
    mean = fmean(past)
    std = pstdev(past)
    return SelfHistoryStat(
        window=window,
        n_obs=len(past),
        percentile=percentile_rank(past, today_residual),
        mean=mean,
        self_z=None if std == 0 else (today_residual - mean) / std,
    )


@dataclass(frozen=True)
class DailyView:
    """One bond's cross-sectional result plus its own-history statistics."""

    result: ValuationResult
    history: Tuple[SelfHistoryStat, ...]

    def stat(self, window: int) -> SelfHistoryStat:
        for item in self.history:
            if item.window == window:
                return item
        raise KeyError(window)


def daily_valuation(
    records: Sequence[Record],
    feature_names: Sequence[str],
    trade_date: object,
    history_path: str,
    *,
    windows: Sequence[int] = (60, 120),
    min_obs: int = 20,
    log_features: Sequence[str] = ("issue_size",),
    neighbor_count: int | None = None,
) -> List[DailyView]:
    """Daily job: cross-sectional valuation, own-history percentile, then save.

    1. Value every bond against its nearest peers (``group_valuation``).
    2. For each bond, take its last ``window`` stored residuals strictly before
       ``trade_date`` and rank today's residual inside them.
    3. Save today's residuals so tomorrow's run can use them.

    A window with fewer than ``min(min_obs, window)`` past days reports no
    percentile, because a rank from a handful of days is not meaningful.
    """
    if not windows or any(window < 2 for window in windows):
        raise ValueError("each window must be at least 2 trading days")
    if min_obs < 2:
        raise ValueError("min_obs must be at least 2")
    day = _iso_date(trade_date)
    results = group_valuation(
        records,
        feature_names,
        log_features=log_features,
        neighbor_count=neighbor_count,
    )
    views = build_views(results, load_residual_history(history_path), day, windows, min_obs)
    save_residuals(results, day, history_path)
    return views


def build_views(
    results: Sequence[ValuationResult],
    history: Mapping[str, Sequence[Tuple[str, float]]],
    trade_date: object,
    windows: Sequence[int],
    min_obs: int,
) -> List[DailyView]:
    """Attach own-history statistics to today's results (nothing is saved)."""
    day = _iso_date(trade_date)
    return [
        DailyView(
            result=result,
            history=tuple(
                self_history_stat(
                    history.get(result.code, []),
                    day,
                    result.residual_pct,
                    window,
                    min(min_obs, window),
                )
                for window in windows
            ),
        )
        for result in results
    ]


def print_daily(views: Sequence[DailyView], windows: Sequence[int]) -> None:
    """Print the daily table.  Low resid/z = cheap versus peers today;
    low pct = cheap versus the bond's own recent history."""
    first = windows[0]
    head = "code   parity  resid_pp      z" + "".join(f"  pct{w:<4}" for w in windows)
    print(head + f"  selfz{first}  norm{first}")
    for view in views:
        r = view.result
        cells = ""
        for window in windows:
            s = view.stat(window)
            cells += "      -  " if s.percentile is None else f"  {s.percentile:6.1f} "
        s0 = view.stat(first)
        selfz = "     -" if s0.self_z is None else f"{s0.self_z:6.2f}"
        norm = "    -" if s0.mean is None else f"{s0.mean:+5.2f}"
        print(
            f"{r.code:6} {r.parity:6.2f} {r.residual_pct:+9.2f} {r.z_score:6.2f}"
            f"{cells}  {selfz}   {norm}"
        )


# ---------------------------------------------------------------------------
# Risk screen
#
# Bonds whose price is driven by credit / delisting risk (or by a pending
# forced call) follow a different pricing logic from the parity-premium curve.
# If they stay in the sample they bend their neighbours' curves, inflate the
# standard deviation behind every z-score, and usually rank as "cheapest"
# themselves.  The screen removes them before neighbours are searched, so they
# take no part in curve fitting, z-scores or the stored residual history.
# ---------------------------------------------------------------------------

_TRUE_TEXT = {"1", "1.0", "true", "yes", "y", "是"}


def _flag(row: Record, key: str) -> bool:
    """True when an optional 0/1 column is set; a missing column counts as 0."""
    value = row.get(key)
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in _TRUE_TEXT


def _optional_number(row: Record, key: str) -> Optional[float]:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def screen_risky(
    records: Iterable[Record],
    *,
    min_stock_price: float = 1.2,
    floor_ratio: float = 0.95,
) -> Tuple[List[Dict[str, object]], List[Tuple[str, List[str]]]]:
    """Split today's bonds into (kept, dropped-with-reasons).

    Optional columns, all checked only when present:
    - ``is_st``: 1 when the underlying stock is ST / *ST;
    - ``delisting``: 1 when the stock is in the delisting-arrangement period
      or delisting is already locked in;
    - ``stock_price``: below ``min_stock_price`` means close to the 1-yuan
      face-value delisting line (20 straight closes below 1 yuan);
    - ``suspended``: 1 when the bond or stock is suspended (stale price);
    - ``call_announced``: 1 when a forced call is announced (premium heads to 0);
    - ``bond_floor``: a close below ``floor_ratio`` x bond floor means the
      market is pricing credit risk, not the conversion option.
    """
    kept: List[Dict[str, object]] = []
    dropped: List[Tuple[str, List[str]]] = []
    for row in enrich_market_fields(records):
        reasons: List[str] = []
        if _flag(row, "delisting"):
            reasons.append("退市整理期/已锁定退市")
        if _flag(row, "is_st"):
            reasons.append("正股 ST")
        stock = _optional_number(row, "stock_price")
        if stock is not None and stock < min_stock_price:
            reasons.append(f"正股价 {stock:.2f} 元，接近 1 元面值退市线")
        if _flag(row, "suspended"):
            reasons.append("停牌")
        if _flag(row, "call_announced"):
            reasons.append("已公告强赎")
        floor = _optional_number(row, "bond_floor")
        close = _number(row, "close")
        if floor is not None and floor > 0 and close < floor_ratio * floor:
            reasons.append(f"价格 {close:.2f} 低于债底 {floor:.2f} 的 {floor_ratio:.0%}")
        if reasons:
            dropped.append((_code(row), reasons))
        else:
            kept.append(row)
    return kept, dropped


def print_dropped(dropped: Sequence[Tuple[str, List[str]]]) -> None:
    if not dropped:
        print("风险筛查：没有券被剔除\n")
        return
    print(f"风险筛查：剔除 {len(dropped)} 只（不参与找近邻、拟合、z 值，也不写入残差历史）")
    for code, reasons in dropped:
        print(f"  {code}: " + "；".join(reasons))
    print()


# ---------------------------------------------------------------------------
# Risk watch
#
# The residual history cannot tell credit trouble from a genuine bargain: both
# look "cheaper than usual", and the residual has already removed the falling
# stock price through the parity curve.  The watch therefore keeps a separate
# history of raw market fields for EVERY bond (screened-out ones included) and
# looks at the bond's own trend in price versus bond floor, stock price versus
# the 1-yuan line, parity drawdown, price drawdown and (optionally) YTM.  It
# prints warnings; it does not remove anything by itself.
# ---------------------------------------------------------------------------

MARKET_FIELDS = (
    "date",
    "code",
    "close",
    "conv",
    "stock_price",
    "bond_floor",
    "ytm",
    "is_st",
    "delisting",
)

LEVEL_ORDER = {"高": 0, "中": 1, "提示": 2}


@dataclass(frozen=True)
class RiskWatchConfig:
    near_floor_ratio: float = 1.05  # close within 5% above the bond floor
    low_stock_price: float = 1.5  # stock price getting close to 1 yuan
    parity_drop_days: int = 60
    parity_drop_pct: float = -20.0  # parity down 20% or more over 60 days
    price_drop_days: int = 20
    price_drop_pct: float = -8.0  # bond price down 8% or more over 20 days
    floor_lookback: int = 20  # count of days below floor in the last 20
    ytm_alert_pct: float = 6.0  # only used when a ytm column is present
    ytm_jump_pp: float = 2.0  # ytm up 2 points or more over price_drop_days


@dataclass(frozen=True)
class RiskAlert:
    code: str
    level: str  # 高 / 中 / 提示
    messages: Tuple[str, ...]
    excluded: bool = False  # True when the hard screen also removed the bond


def save_market_snapshot(records: Iterable[Record], trade_date: object, path: str) -> int:
    """Store today's raw market fields for every bond, screened-out ones included."""
    day = _iso_date(trade_date)

    def cell(row: Record, key: str) -> str:
        value = _optional_number(row, key)
        if value is None and key in ("is_st", "delisting"):
            return "1" if _flag(row, key) else ""
        return "" if value is None else f"{value:.6f}"

    rows = [
        {"date": day, "code": _code(row), **{k: cell(row, k) for k in MARKET_FIELDS[2:]}}
        for row in enrich_market_fields(records)
    ]
    return _upsert_by_date(path, MARKET_FIELDS, day, rows)


def load_market_history(path: str) -> Dict[str, List[Dict[str, Optional[float]]]]:
    """Return ``{code: [row, ...]}`` sorted by date; empty cells become None."""
    history: Dict[str, List[Dict[str, Optional[float]]]] = {}
    if not os.path.exists(path):
        return history
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, Optional[float]] = {}
            for key in MARKET_FIELDS[2:]:
                text = (row.get(key) or "").strip()
                parsed[key] = float(text) if text else None
            parsed["_date"] = row["date"]  # type: ignore[assignment]
            history.setdefault(row["code"], []).append(parsed)
    for series in history.values():
        series.sort(key=lambda item: str(item["_date"]))
    return history


def neighbor_parity_ranges(
    records: Sequence[Record],
    feature_names: Sequence[str],
    *,
    log_features: Sequence[str] = ("issue_size",),
    neighbor_count: int | None = None,
) -> Dict[str, Tuple[float, float]]:
    """Each bond's neighbours' parity range, to flag extrapolated valuations."""
    rows = enrich_market_fields(records)
    if len(rows) < 4:
        return {}
    groups = neighbors_cb_groups(
        rows, feature_names, log_features=log_features, neighbor_count=neighbor_count
    )
    parity = {_code(row): _number(row, "conv") for row in rows}
    return {
        code: (min(parity[p] for p in peers), max(parity[p] for p in peers))
        for code, peers in groups.items()
    }


def _pct_change(series: Sequence[Optional[float]], days: int) -> Optional[float]:
    """Percent change from ``days`` stored days ago to today (last element)."""
    if len(series) <= days:
        return None
    start, end = series[-1 - days], series[-1]
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def risk_watch(
    records: Iterable[Record],
    trade_date: object,
    market_history: Mapping[str, Sequence[Mapping[str, Optional[float]]]],
    *,
    excluded_codes: Iterable[str] = (),
    parity_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
    config: RiskWatchConfig = RiskWatchConfig(),
) -> List[RiskAlert]:
    """Warnings from each bond's own recent history plus today's values.

    Only days strictly before ``trade_date`` are read from ``market_history``;
    today's values come from ``records``.  Rules whose columns are missing
    are skipped.
    """
    day = _iso_date(trade_date)
    excluded = set(excluded_codes)
    alerts: List[RiskAlert] = []
    for row in enrich_market_fields(records):
        code = _code(row)
        past = [item for item in market_history.get(code, []) if str(item["_date"]) < day]
        today = {key: _optional_number(row, key) for key in MARKET_FIELDS[2:]}
        today["close"] = _number(row, "close")
        today["conv"] = _number(row, "conv")
        series = list(past) + [today]

        def col(key: str) -> List[Optional[float]]:
            return [item.get(key) for item in series]

        high: List[str] = []
        mid: List[str] = []
        note: List[str] = []

        close, floor, stock = today["close"], today["bond_floor"], today["stock_price"]
        if floor:
            recent = series[-config.floor_lookback:]
            below = sum(
                1 for item in recent
                if item.get("close") is not None and item.get("bond_floor")
                and item["close"] < item["bond_floor"]  # type: ignore[operator]
            )
            if close < floor:
                high.append(
                    f"价格 {close:.2f} 低于债底 {floor:.2f}"
                    f"（近 {len(recent)} 日有 {below} 日低于债底）"
                )
            elif close < config.near_floor_ratio * floor:
                mid.append(f"价格 {close:.2f} 距债底 {floor:.2f} 不足 {config.near_floor_ratio - 1:.0%}")

        if stock is not None:
            run = 0
            for value in reversed(col("stock_price")):
                if value is not None and value < 1.0:
                    run += 1
                else:
                    break
            if run:
                high.append(
                    f"正股连续 {run} 个交易日低于 1 元"
                    + ("（已满 20 日，触发面值退市）" if run >= 20 else f"（再 {20 - run} 日触发面值退市）")
                )
            elif stock < config.low_stock_price:
                mid.append(f"正股价 {stock:.2f} 元，低于 {config.low_stock_price:.1f} 元")

        parity_change = _pct_change(col("conv"), config.parity_drop_days)
        if parity_change is not None and parity_change <= config.parity_drop_pct:
            mid.append(f"平价 {config.parity_drop_days} 日累计下跌 {abs(parity_change):.1f}%")

        price_change = _pct_change(col("close"), config.price_drop_days)
        if price_change is not None and price_change <= config.price_drop_pct:
            mid.append(f"转债价格 {config.price_drop_days} 日下跌 {abs(price_change):.1f}%")

        ytm = today.get("ytm")
        if ytm is not None:
            ytm_series = col("ytm")
            if ytm >= config.ytm_alert_pct:
                mid.append(f"到期收益率 {ytm:.2f}%，高于 {config.ytm_alert_pct:.1f}%")
            if len(ytm_series) > config.price_drop_days:
                start = ytm_series[-1 - config.price_drop_days]
                if start is not None and ytm - start >= config.ytm_jump_pp:
                    mid.append(f"到期收益率 {config.price_drop_days} 日上升 {ytm - start:.2f} 个百分点")

        if _flag(row, "is_st"):
            high.append("正股 ST")
        if _flag(row, "delisting"):
            high.append("退市整理期/已锁定退市")

        if parity_ranges and code in parity_ranges and code not in excluded:
            lo, hi = parity_ranges[code]
            parity = today["conv"]
            if parity < lo or parity > hi:
                note.append(
                    f"平价 {parity:.2f} 不在近邻平价范围 {lo:.2f}–{hi:.2f} 内，估值是外推的，仅供参考"
                )

        if high:
            alerts.append(RiskAlert(code, "高", tuple(high + mid + note), code in excluded))
        elif mid:
            alerts.append(RiskAlert(code, "中", tuple(mid + note), code in excluded))
        elif note:
            alerts.append(RiskAlert(code, "提示", tuple(note), code in excluded))
    return sorted(alerts, key=lambda a: (LEVEL_ORDER[a.level], a.code))


def print_risk_alerts(alerts: Sequence[RiskAlert]) -> None:
    if not alerts:
        print("风险提醒：无\n")
        return
    counts = {level: sum(1 for a in alerts if a.level == level) for level in LEVEL_ORDER}
    summary = "，".join(f"{level} {n} 只" for level, n in counts.items() if n)
    print(f"风险提醒（{summary}）")
    for alert in alerts:
        tag = "（已被风险筛查剔除）" if alert.excluded else ""
        print(f"  [{alert.level}] {alert.code}{tag}")
        for message in alert.messages:
            print(f"        - {message}")
    print()


def load_records_csv(path: str, encoding: str = "utf-8-sig") -> List[Dict[str, object]]:
    """Read one day's bonds from a CSV with a header row.

    ``code`` stays text; every other non-empty cell becomes a float when it
    looks like a number (thousands separators allowed) and stays text otherwise.
    The file needs ``code``, the market fields accepted by
    ``enrich_market_fields`` (``conv``/``prem``/``close`` or
    ``stock_price``/``conversion_price``/``close``) and the feature columns.
    For Wind exports saved as GBK, pass ``encoding="gbk"``.
    """
    records: List[Dict[str, object]] = []
    with open(path, newline="", encoding=encoding) as handle:
        for row in csv.DictReader(handle):
            record = _parse_cells(row)
            if record:
                records.append(record)
    if not records:
        raise ValueError(f"{path} has no data rows")
    return records


def _parse_cells(row: Mapping[Optional[str], Optional[str]]) -> Dict[str, object]:
    """``code`` and ``date`` stay text; other cells become floats when numeric."""
    record: Dict[str, object] = {}
    for key, cell in row.items():
        if key is None:
            continue
        key = key.strip()
        cell = (cell or "").strip()
        if not key or not cell:
            continue
        if key in ("code", "date"):
            record[key] = cell
            continue
        try:
            record[key] = float(cell.replace(",", ""))
        except ValueError:
            record[key] = cell
    return record


# ---------------------------------------------------------------------------
# Two-file mode: today's quotes + one history of daily quotes
#
# The history file has the same columns as today's file plus a ``date``
# column, one row per bond per day.  Residual history and risk-watch history
# are both derived from it on every run, so they always match the current
# model settings (features, risk screen); nothing derived is stored.
# ---------------------------------------------------------------------------


def load_history_csv(path: str, encoding: str = "utf-8-sig") -> Dict[str, List[Dict[str, object]]]:
    """Return ``{date: [record, ...]}``; a missing file means no history yet."""
    by_date: Dict[str, List[Dict[str, object]]] = {}
    if not os.path.exists(path):
        return by_date
    with open(path, newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle)
        if "date" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} needs a 'date' column")
        for row in reader:
            record = _parse_cells(row)
            if not record:
                continue
            day = _iso_date(record.pop("date"))
            by_date.setdefault(day, []).append(record)
    return by_date


def save_history_snapshot(
    records: Sequence[Record], trade_date: object, path: str, encoding: str = "utf-8-sig"
) -> int:
    """Append (or replace) today's quotes in the history file, keeping all columns."""
    day = _iso_date(trade_date)
    old_rows: List[Dict[str, str]] = []
    fields: List[str] = ["date"]
    if os.path.exists(path):
        with open(path, newline="", encoding=encoding) as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or ["date"])
            old_rows = [row for row in reader if _iso_date(row["date"]) != day]
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)

    def text(value: object) -> str:
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else repr(value)
        return "" if value is None else str(value)

    new_rows = [{"date": day, **{k: text(v) for k, v in record.items()}} for record in records]
    rows = sorted(old_rows + new_rows, key=lambda row: (_iso_date(row["date"]), row.get("code", "")))
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".history-", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding=encoding) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return len(new_rows)


def residuals_from_history(
    history_by_date: Mapping[str, Sequence[Record]],
    trade_date: object,
    feature_names: Sequence[str],
    *,
    days: int,
    log_features: Sequence[str] = ("issue_size",),
    neighbor_count: int | None = None,
    exclude_risky: bool = False,
    min_stock_price: float = 1.2,
    floor_ratio: float = 0.95,
) -> Tuple[Dict[str, List[Tuple[str, float]]], int]:
    """Re-run the cross-sectional valuation on each of the last ``days`` dates.

    Uses the same features and the same risk screen as today, so past and
    present residuals are measured the same way.  Returns
    ``({code: [(date, residual_pct), ...]}, number_of_dates_valued)``.
    """
    day = _iso_date(trade_date)
    dates = sorted(d for d in history_by_date if d < day)[-days:]
    series: Dict[str, List[Tuple[str, float]]] = {}
    valued = 0
    for date in dates:
        records: Sequence[Record] = history_by_date[date]
        if exclude_risky:
            records, _ = screen_risky(
                records, min_stock_price=min_stock_price, floor_ratio=floor_ratio
            )
        if len(records) < 4:
            continue
        try:
            results = group_valuation(
                records, feature_names, log_features=log_features, neighbor_count=neighbor_count
            )
        except ValueError:
            continue  # e.g. a feature with zero variance that day
        valued += 1
        for result in results:
            series.setdefault(result.code, []).append((date, result.residual_pct))
    return series, valued


def market_series_from_history(
    history_by_date: Mapping[str, Sequence[Record]]
) -> Dict[str, List[Dict[str, Optional[float]]]]:
    """Build the risk-watch history (price, floor, stock price...) for every bond."""
    out: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for date in sorted(history_by_date):
        try:
            rows = enrich_market_fields(history_by_date[date])
        except ValueError:
            continue
        for row in rows:
            item: Dict[str, Optional[float]] = {k: _optional_number(row, k) for k in MARKET_FIELDS[2:]}
            item["is_st"] = 1.0 if _flag(row, "is_st") else 0.0
            item["delisting"] = 1.0 if _flag(row, "delisting") else 0.0
            item["_date"] = date  # type: ignore[assignment]
            out.setdefault(_code(row), []).append(item)
    return out


def _demo_records() -> List[Dict[str, object]]:
    """Synthetic, deterministic data for a runnable example (not investment data)."""
    specs = [
        ("CB01", 86.0, 44.0, 24.0, 1.0, -10.0),
        ("CB02", 90.0, 39.0, 27.0, 2.0, -7.0),
        ("CB03", 94.0, 34.5, 30.0, 3.0, -4.0),
        ("CB04", 98.0, 30.0, 35.0, 4.0, -1.0),
        ("CB05", 102.0, 26.0, 40.0, 5.0, 2.0),
        ("CB06", 106.0, 22.5, 48.0, 6.0, 5.0),
        ("CB07", 110.0, 19.0, 56.0, 7.0, 8.0),
        ("CB08", 114.0, 16.0, 64.0, 8.0, 11.0),
        ("CB09", 118.0, 13.5, 72.0, 9.0, 14.0),
        ("CB10", 122.0, 11.0, 84.0, 10.0, 17.0),
        ("CB11", 126.0, 9.0, 96.0, 11.0, 20.0),
        ("CB12", 130.0, 7.5, 110.0, 12.0, 23.0),
    ]
    records: List[Dict[str, object]] = []
    for index, (code, conv, prem, size, coverage, momentum) in enumerate(specs):
        # Make CB07 visibly cheap so the ranking can be inspected in the demo.
        observed_prem = prem - 6.0 if code == "CB07" else prem + (index % 3 - 1) * 0.6
        records.append(
            {
                "code": code,
                "conv": conv,
                "prem": observed_prem,
                "close": price_from_premium(conv, observed_prem),
                "issue_size": size,
                "analyst_coverage": coverage,
                "momentum_60d": momentum,
            }
        )
    return records


def run_demo() -> None:
    records = _demo_records()
    results = group_valuation(
        records,
        feature_names=("issue_size", "analyst_coverage", "momentum_60d"),
    )
    print("Synthetic example; lower diff/z means more relatively undervalued")
    print("code  parity  actual%  fitted%  close   fair    diff    z")
    for result in results:
        print(
            f"{result.code:5} {result.parity:7.2f} {result.actual_premium_pct:8.2f} "
            f"{result.fitted_premium_pct:8.2f} {result.close:7.2f} "
            f"{result.fair_value:7.2f} {result.diff:7.2f} {result.z_score:6.2f}"
        )


def _trading_days(start: _dt.date, count: int) -> List[_dt.date]:
    """Weekdays from ``start`` (holidays ignored; enough for a simulation)."""
    days: List[_dt.date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += _dt.timedelta(days=1)
    return days


def run_history_demo(days: int = 130, seed: int = 7) -> None:
    """Simulate ``days`` trading days, run the daily job on each, print the last day.

    Synthetic set-up (not market data):
    - every bond's premium moves with a common market factor, which the
      cross-sectional fit should remove;
    - CB04 normally trades about 4 points above its peers; on the last day it
      falls back to about +0.5, so it looks roughly fair against peers but
      cheap against its own history;
    - CB07 is cheap against its peers every day, so its own-history
      percentile stays ordinary.
    """
    rng = random.Random(seed)
    base = _demo_records()
    persistent = {"CB04": 4.0}
    noise = {str(r["code"]): 0.0 for r in base}
    market = 0.0
    features = ("issue_size", "analyst_coverage", "momentum_60d")
    windows = (60, 120)
    dates = _trading_days(_dt.date(2026, 4, 1), days)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "residual_history.csv")
        views: List[DailyView] = []
        for index, day in enumerate(dates):
            last = index == len(dates) - 1
            market = 0.97 * market + rng.gauss(0.0, 0.6)
            today: List[Dict[str, object]] = []
            for record in base:
                code = str(record["code"])
                noise[code] = 0.6 * noise[code] + rng.gauss(0.0, 0.5)
                bump = 0.5 if (last and code == "CB04") else persistent.get(code, 0.0)
                prem = float(record["prem"]) + market + bump + noise[code]  # type: ignore[arg-type]
                row = dict(record)
                row["prem"] = prem
                row["close"] = price_from_premium(float(record["conv"]), prem)  # type: ignore[arg-type]
                today.append(row)
            views = daily_valuation(today, features, day, path, windows=windows)
    print(f"Synthetic {days}-day history, last day {dates[-1].isoformat()} (not market data)")
    print("resid_pp/z: versus peers today (low = cheap).")
    print("pct: today's residual inside the bond's own past window (low = cheap versus its norm).")
    print("norm: the bond's average residual over the first window.\n")
    print_daily(views, windows)


def _self_check() -> None:
    assert math.isclose(conversion_value(12, 10), 120.0)
    assert math.isclose(conversion_premium_pct(138, 120), 15.0)
    assert math.isclose(price_from_premium(120, 15), 138.0)
    results = group_valuation(
        _demo_records(),
        feature_names=("issue_size", "analyst_coverage", "momentum_60d"),
    )
    assert len(results) == 12
    assert all(math.isfinite(result.fair_value) for result in results)
    assert results == sorted(results, key=lambda item: (item.z_score, item.diff, item.code))
    assert all(
        math.isclose(r.residual_pct, r.actual_premium_pct - r.fitted_premium_pct)
        for r in results
    )

    assert math.isclose(percentile_rank([1, 2, 3, 4], 2.5), 50.0)
    assert math.isclose(percentile_rank([1, 2, 3, 4], 2), 37.5)
    assert math.isclose(percentile_rank([1, 2, 3, 4], 0), 0.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.csv")
        assert save_residuals(results, "2026-10-21", path) == 12
        assert save_residuals(results, "2026-10-22", path) == 12
        assert save_residuals(results, "2026-10-22", path) == 12  # rerun replaces
        history = load_residual_history(path)
        assert len(history) == 12 and all(len(s) == 2 for s in history.values())
        # strictly-before rule: a re-run of 2026-10-22 sees only 2026-10-21
        assert self_history_stat(history["CB07"], "2026-10-22", 0.0, 60, 1).n_obs == 1
        assert self_history_stat(history["CB07"], "2026-10-23", 0.0, 60, 1).n_obs == 2
        assert self_history_stat(history["CB07"], "2026-10-23", 0.0, 60, 5).percentile is None
        assert self_history_stat(history["CB07"], "2026-10-23", 0.0, 1, 1).n_obs == 1
    kept, dropped = screen_risky(
        [
            {"code": "OK", "conv": 100, "prem": 20, "stock_price": 10, "bond_floor": 95},
            {"code": "ST", "conv": 100, "prem": 20, "is_st": 1},
            {"code": "PAR", "conv": 40, "prem": 70, "stock_price": 0.9},
            {"code": "FLOOR", "conv": 50, "prem": 40, "bond_floor": 100},
            {"code": "CALL", "conv": 140, "prem": 1, "call_announced": "是"},
        ]
    )
    assert [r["code"] for r in kept] == ["OK"]
    assert [code for code, _ in dropped] == ["ST", "PAR", "FLOOR", "CALL"]

    # risk watch: 3 past days with the stock under 1 yuan, today too -> run of 4
    past_rows = {
        "BAD": [
            {"_date": f"2026-10-0{i}", "close": 90.0, "conv": 45.0, "stock_price": 0.9,
             "bond_floor": 92.0, "ytm": None, "is_st": None, "delisting": None}
            for i in (1, 2, 3)
        ]
    }
    alerts = risk_watch(
        [
            {"code": "BAD", "conv": 44, "close": 85, "stock_price": 0.88, "bond_floor": 92},
            {"code": "GOOD", "conv": 110, "close": 125, "stock_price": 11, "bond_floor": 96},
        ],
        "2026-10-05",
        past_rows,  # type: ignore[arg-type]
    )
    assert [a.code for a in alerts] == ["BAD"] and alerts[0].level == "高"
    assert any("连续 4 个交易日" in msg for msg in alerts[0].messages)
    print("self-check passed")


def _split(text: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--self-check", action="store_true", help="run built-in checks")
    parser.add_argument(
        "--history-demo", action="store_true", help="simulate 130 trading days of residual history"
    )
    parser.add_argument("--input", help="CSV of today's bonds, one row per bond")
    parser.add_argument("--date", help="trade date of --input, YYYY-MM-DD")
    parser.add_argument(
        "--history", default="residual_history.csv", help="residual history CSV (created if missing)"
    )
    parser.add_argument(
        "--features",
        default="issue_size,analyst_coverage,momentum_60d",
        help="comma-separated feature columns used to find neighbours",
    )
    parser.add_argument(
        "--log-features", default="issue_size", help="feature columns to log-transform"
    )
    parser.add_argument("--windows", default="60,120", help="own-history windows in trading days")
    parser.add_argument(
        "--min-obs", type=int, default=20, help="minimum past days before a percentile is reported"
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="encoding of --input (gbk for some Wind exports)")
    parser.add_argument(
        "--exclude-risky",
        action="store_true",
        help="drop ST / near-1-yuan / suspended / call-announced / below-floor bonds before valuation",
    )
    parser.add_argument(
        "--min-stock-price", type=float, default=1.2, help="stock price below this is treated as delisting risk"
    )
    parser.add_argument(
        "--floor-ratio", type=float, default=0.95, help="close below this share of bond_floor is dropped"
    )
    parser.add_argument(
        "--market-history",
        default="market_history.csv",
        help="history of raw market fields used by the risk watch (created if missing)",
    )
    parser.add_argument(
        "--history-data",
        help="two-file mode: history of daily quotes (same columns as --input plus 'date'); "
        "residual and risk histories are derived from it and today's quotes are appended",
    )
    args = parser.parse_args()

    if args.self_check:
        _self_check()
    elif args.history_demo:
        run_history_demo()
    elif args.input:
        if not args.date:
            parser.error("--date is required with --input")
        windows = tuple(int(item) for item in _split(args.windows))
        features = _split(args.features)
        log_features = _split(args.log_features)
        all_records: List[Dict[str, object]] = load_records_csv(args.input, encoding=args.encoding)
        records: Sequence[Record] = all_records
        dropped: List[Tuple[str, List[str]]] = []
        if args.exclude_risky:
            records, dropped = screen_risky(
                all_records, min_stock_price=args.min_stock_price, floor_ratio=args.floor_ratio
            )
            print_dropped(dropped)
        else:
            print("风险筛查：未开启（加 --exclude-risky 开启）\n")

        if args.history_data:
            day = _iso_date(args.date)
            history_by_date = load_history_csv(args.history_data, encoding=args.encoding)
            past_dates = sorted(d for d in history_by_date if d < day)
            residuals, valued = residuals_from_history(
                history_by_date,
                day,
                features,
                days=max(windows),
                log_features=log_features,
                exclude_risky=args.exclude_risky,
                min_stock_price=args.min_stock_price,
                floor_ratio=args.floor_ratio,
            )
            span = f"{past_dates[0]} 至 {past_dates[-1]}" if past_dates else "无"
            print(f"历史行情：{len(past_dates)} 个交易日（{span}），重算了最近 {valued} 日的截面残差\n")

            alerts = risk_watch(
                all_records,
                day,
                market_series_from_history(history_by_date),
                excluded_codes=[code for code, _ in dropped],
                parity_ranges=neighbor_parity_ranges(records, features, log_features=log_features),
            )
            print_risk_alerts(alerts)

            results = group_valuation(records, features, log_features=log_features)
            views = build_views(results, residuals, day, windows, args.min_obs)
            save_history_snapshot(all_records, day, args.history_data, encoding=args.encoding)
            print(f"{day}: {len(views)} bonds valued, today's quotes saved to {args.history_data}\n")
            print_daily(views, windows)
            high = [a.code for a in alerts if a.level == "高" and not a.excluded]
            if high:
                print(f"\n注意：{'、'.join(high)} 有高风险提醒但仍在上表的估值和排名里，"
                      "建议加 --exclude-risky 或先人工核查。")
            return

        alerts = risk_watch(
            all_records,
            args.date,
            load_market_history(args.market_history),
            excluded_codes=[code for code, _ in dropped],
            parity_ranges=neighbor_parity_ranges(records, features, log_features=log_features),
        )
        save_market_snapshot(all_records, args.date, args.market_history)
        print_risk_alerts(alerts)

        views = daily_valuation(
            records,
            features,
            args.date,
            args.history,
            windows=windows,
            min_obs=args.min_obs,
            log_features=log_features,
        )
        print(f"{_iso_date(args.date)}: {len(views)} bonds valued, history saved to {args.history}\n")
        print_daily(views, windows)
        high = [a.code for a in alerts if a.level == "高" and not a.excluded]
        if high:
            print(f"\n注意：{'、'.join(high)} 有高风险提醒但仍在上表的估值和排名里，"
                  "建议加 --exclude-risky 或先人工核查。")
    else:
        run_demo()


if __name__ == "__main__":
    main()