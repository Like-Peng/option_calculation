"""Practical convertible-bond relative valuation.

This module implements the method disclosed in the CICC article
"Convertible-bond 'undervaluation' strategy optimization and Python
implementation" (2021-04-19):

1. Standardize bond/underlying-stock characteristics.
2. Find each bond's nearest comparable bonds (KNN-like Euclidean distance).
3. Fit the local parity-premium curve ``p = a + b*x + c/x``.
4. Convert the fitted premium into a fair bond price and rank the price gaps.

The implementation deliberately uses only the Python standard library.  It is
therefore directly runnable without pandas, NumPy, scikit-learn, BSM, or a
binomial-tree model.

Rates named ``*_pct`` are percentage points: ``12.5`` means 12.5%.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


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
    return face_value * stock_price / conversion_price # 面值 / 转股价 * 股票价格 = 转股价值


def conversion_premium_pct(bond_price: Number, parity: Number) -> float:
    """Return the conversion premium in percentage points."""
    bond_price = float(bond_price)
    parity = float(parity)
    if bond_price < 0:
        raise ValueError("bond_price must be non-negative")
    if parity <= 0:
        raise ValueError("parity must be positive")
    return (bond_price / parity - 1.0) * 100.0  # 转股溢价率 = (转债价格 ÷ 转股价值 - 1) * 100


def price_from_premium(parity: Number, premium_pct: Number) -> float:
    """Convert parity and a premium percentage into a bond price."""
    parity = float(parity)
    if parity <= 0:
        raise ValueError("parity must be positive")
    return parity * (1.0 + float(premium_pct) / 100.0) # 转债价格 = 转股价值 * (1 + 转股溢价率 ÷ 100)


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

    standardized_columns: List[List[float]] = [] #标准化
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
    proposed = min(max(6, math.floor(sample_size / 10)), math.floor(sample_size / 2), 18) # CICC's disclosed rule
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
            distance = sum((left - right) ** 2 for left, right in zip(matrix[i], matrix[j])) #标准化特征的平方距离
            distances.append((distance, peer_code))
        distances.sort(key=lambda item: (item[0], item[1])) #排序
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
        return self.a + self.b * conv + self.c / conv # 拟合曲线：y = a + b*x + c/x

    def predict_price(self, conv: Number) -> float:
        return price_from_premium(conv, self.predict_premium_pct(conv)) # 拟合曲线对应的价格 = 转股价值 * (1 + 拟合曲线 ÷ 100)


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
        diff = close - fair_value # 判断高估还是低估
        provisional.append(
            ValuationResult(
                code=code,
                parity=parity,
                actual_premium_pct=_number(row, "prem"),
                fitted_premium_pct=curve.predict_premium_pct(parity),
                fair_value=fair_value,
                close=close,
                diff=diff,
                diff_pct=diff / fair_value * 100.0, # 比较不同价格水平转债的偏离比例
            )
        )

    diffs = [result.diff for result in provisional] 
    mean_diff = fmean(diffs)
    std_diff = pstdev(diffs)
    results = [
        ValuationResult(
            **{
                **result.__dict__,
                "z_score": 0.0 if std_diff == 0 else (result.diff - mean_diff) / std_diff, # diff的线性标准化
            }
        )
        for result in provisional
    ]
    return sorted(results, key=lambda result: (result.z_score, result.diff, result.code))


# Compatibility names corresponding to the article's published snippets.
def neighborsCbGroups(  # 先找同类券
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


def olsForGroup(  # 用同类券拟合曲线并算合理价格
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
    print("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="run built-in checks")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
    else:
        run_demo()


if __name__ == "__main__":
    main()


################################
#目标转债CB01
#    ↓
#根据发行规模、分析师覆盖、趋势等特征
#    ↓
#neighborsCbGroups找出相似转债
#    ↓
#取得这些相似转债的平价和溢价率
#    ↓
#olsForGroup拟合曲线
#    ↓
#把CB01的平价代入曲线
#    ↓
#得到CB01的合理溢价率和合理价格
#    ↓
#实际价格与合理价格比较
#    ↓
#diff = 实际价格 - 合理价格
#    ↓
#对 CB02、CB03……所有转债重复以上过程
#    ↓
#将所有转债的 diff 标准化为 z_score
#    ↓
#按照 z_score 从小到大排序
#    ↓
#获得全市场相对低估排序
#    ↓
#加入信用、退市、强赎、流动性和趋势过滤
#    ↓
#选择排名靠前的一组转债构建组合
