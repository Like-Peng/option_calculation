"""Practical convertible-bond relative valuation.

This module implements the method disclosed in the CICC article
"Convertible-bond 'undervaluation' strategy optimization and Python
implementation" (2021-04-19):

1. Standardize bond/underlying-stock characteristics.
2. Find each bond's nearest comparable bonds (KNN-like Euclidean distance).
3. Fit the local parity-premium curve ``p = a + b*x + c/x``.
4. Convert the fitted premium into a fair bond price and rank the price gaps.
5. Save each day's cross-sectional residual and rank today's residual inside
   the bond's own last 60 / 120 stored trading days.

The implementation deliberately uses only the Python standard library.  It is
therefore directly runnable without pandas, NumPy, scikit-learn, BSM, or a
binomial-tree model.

Rates named ``*_pct`` are percentage points: ``12.5`` means 12.5%.

Usage::

    python option_calculation.py                  # one-day synthetic example
    python option_calculation.py --self-check     # built-in checks
    python option_calculation.py --history-demo   # 130 simulated days + percentiles
    python option_calculation.py --input today.csv --date 2026-10-23 \\
        --history residual_history.csv            # daily job on real data
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


def save_residuals(
    results: Sequence[ValuationResult], trade_date: object, path: str
) -> int:
    """Store one day's valuation results in the history CSV.

    Rows already stored for the same date are replaced, so re-running a day is
    safe.  The file is written to a temporary file first and then renamed, so
    an interrupted run never leaves a half-written history.  Returns the number
    of rows written for ``trade_date``.
    """
    day = _iso_date(trade_date)
    kept: List[Dict[str, str]] = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(HISTORY_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} lacks columns: {sorted(missing)}")
            kept = [
                {field: row[field] for field in HISTORY_FIELDS}
                for row in reader
                if row["date"] != day
            ]

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
    rows = sorted(kept + new_rows, key=lambda row: (row["date"], row["code"]))

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".residuals-", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return len(new_rows)


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
    history = load_residual_history(history_path)
    views = [
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
    save_residuals(results, day, history_path)
    return views


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
            record: Dict[str, object] = {}
            for key, cell in row.items():
                if key is None:
                    continue
                key = key.strip()
                cell = (cell or "").strip()
                if not key or not cell:
                    continue
                if key == "code":
                    record["code"] = cell
                    continue
                try:
                    record[key] = float(cell.replace(",", ""))
                except ValueError:
                    record[key] = cell
            if record:
                records.append(record)
    if not records:
        raise ValueError(f"{path} has no data rows")
    return records


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
    args = parser.parse_args()

    if args.self_check:
        _self_check()
    elif args.history_demo:
        run_history_demo()
    elif args.input:
        if not args.date:
            parser.error("--date is required with --input")
        windows = tuple(int(item) for item in _split(args.windows))
        views = daily_valuation(
            load_records_csv(args.input, encoding=args.encoding),
            _split(args.features),
            args.date,
            args.history,
            windows=windows,
            min_obs=args.min_obs,
            log_features=_split(args.log_features),
        )
        print(f"{_iso_date(args.date)}: {len(views)} bonds valued, history saved to {args.history}\n")
        print_daily(views, windows)
    else:
        run_demo()


if __name__ == "__main__":
    main()