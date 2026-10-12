# ibexQuant\src\ibexQuant\analytics\summary.py

"""
Summaries: the performance and trade tables of a strategy.

Provides:

- :func:`performance_summary` — every return metric of one or more
  strategies, optionally against a benchmark, as a :class:`PerformanceSummary`.
- :func:`trade_summary` — every trade metric of one strategy, as a
  :class:`TradeSummary`.
- :data:`METRIC_UNITS` — the unit of every row, so presentation code formats
  each metric without guessing.

Summaries compute nothing new: every value is the call of the corresponding
analytics function, in a fixed order, under the same conventions. Each
summary prints as plain text (``print(summary)``) for routine use.

Conventions
-----------
- Metrics are rows, series are columns, the factsheet orientation. Rows are
  indexed by ``(section, metric)``, where ``metric`` is the name of the
  function that computes it (``sharpe_ratio``, ``max_drawdown``).
- Values are raw floats: ``0.1534`` is 15.34%. Formatting belongs to
  presentation.
- By default the performance table compares every column over their
  **common period**; the period actually used is part of the result.
- Following GIPS, returns are not annualized over less than one year: CAGR,
  excess CAGR and Calmar are ``NaN`` for a column with fewer than
  ``periods_per_year`` observations. Volatility and Sharpe remain, since
  annualizing them rescales rather than extrapolates a return.
- Benchmark-relative rows are ``NaN`` in the benchmark's own column.

What does not belong here
-------------------------
Labels for display, number formatting and layout, which belong to style,
plotting and reports.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from ibexQuant._validation import require_integer, require_real
from ibexQuant.analytics._validation import (
    require_benchmark,
    require_observations,
    require_returns,
)
from ibexQuant.analytics.distribution import (
    best_period,
    conditional_value_at_risk,
    kurtosis,
    positive_ratio,
    skewness,
    value_at_risk,
    worst_period,
)
from ibexQuant.analytics.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from ibexQuant.analytics.relative import (
    alpha,
    beta,
    correlation,
    down_capture,
    excess_cagr,
    information_ratio,
    tracking_error,
    up_capture,
)
from ibexQuant.analytics.trades import (
    average_holding_period,
    average_loss,
    average_win,
    best_trade,
    expectancy,
    expectancy_t_stat,
    exposure,
    gross_loss,
    gross_profit,
    loss_rate,
    max_consecutive_losses,
    max_consecutive_wins,
    payoff_ratio,
    profit_factor,
    trade_count,
    trade_frequency,
    trade_standard_deviation,
    win_rate,
    worst_trade,
)

MetricUnit = Literal["percent", "ratio", "number", "count", "periods", "result", "rate"]

# What each unit means for presentation:
# - percent: a fraction to show as a percentage (0.15 is 15%);
# - ratio: a dimensionless ratio (Sharpe, beta, profit factor);
# - number: a dimensionless statistic (skewness, a t-statistic);
# - count: an integer count of trades;
# - periods: a number of bars;
# - result: the unit of the trade results given (returns or PnL);
# - rate: trades per bar, or per year when annualized.
METRIC_UNITS: Final[Mapping[str, MetricUnit]] = {
    "total_return": "percent",
    "cagr": "percent",
    "annualized_volatility": "percent",
    "max_drawdown": "percent",
    "max_drawdown_duration": "periods",
    "sharpe_ratio": "ratio",
    "sortino_ratio": "ratio",
    "calmar_ratio": "ratio",
    "skewness": "number",
    "kurtosis": "number",
    "value_at_risk": "percent",
    "conditional_value_at_risk": "percent",
    "best_period": "percent",
    "worst_period": "percent",
    "positive_ratio": "percent",
    "beta": "ratio",
    "alpha": "percent",
    "correlation": "ratio",
    "tracking_error": "percent",
    "information_ratio": "ratio",
    "up_capture": "ratio",
    "down_capture": "ratio",
    "excess_cagr": "percent",
    "trade_count": "count",
    "win_rate": "percent",
    "loss_rate": "percent",
    "expectancy": "result",
    "trade_standard_deviation": "result",
    "expectancy_t_stat": "number",
    "average_win": "result",
    "average_loss": "result",
    "payoff_ratio": "ratio",
    "gross_profit": "result",
    "gross_loss": "result",
    "profit_factor": "ratio",
    "best_trade": "result",
    "worst_trade": "result",
    "max_consecutive_wins": "count",
    "max_consecutive_losses": "count",
    "trade_frequency": "rate",
    "exposure": "percent",
    "average_holding_period": "periods",
}

# Annualizing these extrapolates a return, which GIPS forbids below one year.
GIPS_ANNUALIZED_RETURNS: Final[frozenset[str]] = frozenset({"cagr", "excess_cagr", "calmar_ratio"})

DEFAULT_STRATEGY_NAME: Final[str] = "strategy"
DEFAULT_BENCHMARK_NAME: Final[str] = "benchmark"
TEXT_DECIMALS: Final[int] = 4
TEXT_MISSING: Final[str] = "n/a"


# --- Results ---


@dataclass(frozen=True, eq=False)
class PerformanceSummary:
    """
    Performance table of one or more strategies, with its period and conventions.

    Attributes
    ----------
    metrics : pd.DataFrame
        Raw float values; rows indexed by ``(section, metric)``, one column
        per series, the benchmark last when present.
    start, end : pd.Timestamp
        First and last timestamp of the period used.
    periods : int
        Rows in the period used.
    years : float
        ``periods / periods_per_year``.
    periods_per_year : int
        Annualization factor used.
    risk_free : float or str
        Annual rate used, or the name of the periodic series used.
    confidence : float
        Confidence level of value at risk and conditional value at risk.
    common_period : bool
        Whether every column was measured over the common period.
    short_columns : tuple
        Columns with less than one year of observations, whose annualized
        returns are suppressed (GIPS).
    """

    metrics: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    periods: int
    years: float
    periods_per_year: int
    risk_free: float | str
    confidence: float
    common_period: bool
    short_columns: tuple[Hashable, ...]

    def to_text(self) -> str:
        """
        Render the summary as plain text: header, conventions and table.

        Returns
        -------
        str
            Raw values with four decimals; ``n/a`` where a metric is
            undefined or suppressed.
        """
        scope = "common period" if self.common_period else "each series over its own period"
        risk_free = (
            f"{self.risk_free:.4f} annual" if isinstance(self.risk_free, float) else self.risk_free
        )
        lines = [
            "Performance summary",
            f"Period: {self.start.date()} -> {self.end.date()} "
            f"({self.periods} periods, {self.years:.2f} years, {scope})",
            f"Conventions: {self.periods_per_year} periods per year; risk-free {risk_free}; "
            f"VaR and CVaR at {self.confidence:.0%}",
            "",
            _format_table(self.metrics),
        ]
        if self.short_columns:
            names = ", ".join(str(column) for column in self.short_columns)
            lines += [
                "",
                f"Less than one year of data for {names}: CAGR, excess CAGR and Calmar are not "
                "annualized (GIPS).",
            ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


@dataclass(frozen=True, eq=False)
class TradeSummary:
    """
    Trade table of one strategy.

    Attributes
    ----------
    metrics : pd.Series
        Raw float values indexed by ``(section, metric)``; the ``timing``
        section is present only when trade times were given.
    annualized_frequency : bool
        Whether ``trade_frequency`` is per year rather than per bar.
    """

    metrics: pd.Series
    annualized_frequency: bool

    def to_text(self) -> str:
        """
        Render the summary as plain text.

        Returns
        -------
        str
            Raw values with four decimals; ``n/a`` where a metric is
            undefined.
        """
        frequency = "per year" if self.annualized_frequency else "per bar"
        lines = [
            "Trade summary",
            f"Trade frequency {frequency}; result metrics in the unit of the trade results",
            "",
            _format_table(self.metrics.to_frame("value")),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


# --- Performance ---


def performance_summary(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | None = None,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
    confidence: float = 0.95,
    common_period: bool = True,
) -> PerformanceSummary:
    """
    Build the performance table of one or more strategies.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series, optional
        Periodic simple returns of the benchmark, on the same index. Adds a
        benchmark column and the ``benchmark`` section.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series).
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).
    confidence : float, default 0.95
        Confidence level of value at risk and conditional value at risk.
    common_period : bool, default True
        Measure every column over the period where all of them, the
        benchmark included, are observed. If False, each column is measured
        over its own period.

    Returns
    -------
    PerformanceSummary
        The table, its period and the conventions used.

    Raises
    ------
    TypeError, ValueError
        If an input breaks the analytics contract, *returns* is empty, the
        benchmark index differs, a column name clashes with the benchmark's,
        the series have no common period, or a parameter is invalid.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    confidence = require_real("confidence", confidence, above=0.0, below=1.0)
    strategies = _strategies_frame(returns)

    table = strategies
    market: pd.Series | None = None
    if benchmark is not None:
        market = require_benchmark(returns, benchmark)
        if market.name is None:
            market = market.copy()
            market.name = DEFAULT_BENCHMARK_NAME
        if market.name in strategies.columns:
            raise ValueError(
                f"benchmark name {market.name!r} clashes with a returns column; rename one."
            )
        table = strategies.assign(**{str(market.name): market})

    if common_period:
        observed = table.notna().all(axis=1)
        if not observed.any():
            raise ValueError("returns and benchmark have no period in common.")
        table = table.loc[observed]
    else:
        observed = table.notna().any(axis=1)
        table = table.loc[observed.idxmax() : observed[::-1].idxmax()]

    rows: list[tuple[str, str, pd.Series]] = [
        ("return", "total_return", total_return(table)),
        ("return", "cagr", cagr(table, periods_per_year=periods_per_year)),
        (
            "risk",
            "annualized_volatility",
            annualized_volatility(table, periods_per_year=periods_per_year),
        ),
        ("risk", "max_drawdown", max_drawdown(table)),
        ("risk", "max_drawdown_duration", max_drawdown_duration(table)),
        (
            "risk_adjusted",
            "sharpe_ratio",
            sharpe_ratio(table, risk_free, periods_per_year=periods_per_year),
        ),
        (
            "risk_adjusted",
            "sortino_ratio",
            sortino_ratio(table, risk_free, periods_per_year=periods_per_year),
        ),
        ("risk_adjusted", "calmar_ratio", calmar_ratio(table, periods_per_year=periods_per_year)),
        ("distribution", "skewness", skewness(table)),
        ("distribution", "kurtosis", kurtosis(table)),
        ("distribution", "value_at_risk", value_at_risk(table, confidence)),
        (
            "distribution",
            "conditional_value_at_risk",
            conditional_value_at_risk(table, confidence),
        ),
        ("distribution", "best_period", best_period(table)),
        ("distribution", "worst_period", worst_period(table)),
        ("distribution", "positive_ratio", positive_ratio(table)),
    ]

    if market is not None:
        strategy_part = table[strategies.columns]
        market_part = table[market.name]
        relative = {
            "beta": beta(strategy_part, market_part, risk_free, periods_per_year=periods_per_year),
            "alpha": alpha(
                strategy_part, market_part, risk_free, periods_per_year=periods_per_year
            ),
            "correlation": correlation(strategy_part, market_part),
            "tracking_error": tracking_error(
                strategy_part, market_part, periods_per_year=periods_per_year
            ),
            "information_ratio": information_ratio(
                strategy_part, market_part, periods_per_year=periods_per_year
            ),
            "up_capture": up_capture(
                strategy_part, market_part, periods_per_year=periods_per_year
            ),
            "down_capture": down_capture(
                strategy_part, market_part, periods_per_year=periods_per_year
            ),
            "excess_cagr": excess_cagr(
                strategy_part, market_part, periods_per_year=periods_per_year
            ),
        }
        for name, values in relative.items():
            # The benchmark's own column has no meaningful relative value.
            rows.append(("benchmark", name, values.reindex(table.columns)))

    metrics = pd.DataFrame(
        [values for _, _, values in rows],
        index=pd.MultiIndex.from_tuples(
            [(section, name) for section, name, _ in rows], names=["section", "metric"]
        ),
        columns=table.columns,
        dtype="float64",
    )
    short = table.notna().sum() < periods_per_year
    if short.any():
        annualized = metrics.index.get_level_values("metric").isin(GIPS_ANNUALIZED_RETURNS)
        metrics.loc[annualized, short[short].index] = np.nan

    return PerformanceSummary(
        metrics=metrics,
        start=pd.Timestamp(table.index[0]),
        end=pd.Timestamp(table.index[-1]),
        periods=len(table),
        years=len(table) / periods_per_year,
        periods_per_year=periods_per_year,
        risk_free=_describe_risk_free(risk_free),
        confidence=confidence,
        common_period=common_period,
        short_columns=tuple(short[short].index),
    )


# --- Trades ---


def trade_summary(
    trade_results: pd.Series,
    trades: pd.DataFrame | None = None,
    sessions: pd.DatetimeIndex | None = None,
    *,
    annualized: bool = False,
    periods_per_year: int | None = None,
) -> TradeSummary:
    """
    Build the trade table of one strategy.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade, in chronological order of exit.
    trades : pd.DataFrame, optional
        One row per trade with ``entry_time`` and ``exit_time``. Together
        with *sessions*, adds the ``timing`` section.
    sessions : pd.DatetimeIndex, optional
        Every session of the period, traded or not.
    annualized : bool, default False
        Report trade frequency per year instead of per bar.
    periods_per_year : int, optional
        Sessions in one year; required when *annualized* is True.

    Returns
    -------
    TradeSummary
        The table and whether its frequency is annualized.

    Raises
    ------
    TypeError, ValueError
        If an input breaks its contract, only one of *trades* and *sessions*
        is given, or *periods_per_year* does not match *annualized*.
    """
    if (trades is None) != (sessions is None):
        raise ValueError("trades and sessions must be given together for timing metrics.")

    values: dict[tuple[str, str], float] = {
        ("outcome", "trade_count"): float(trade_count(trade_results)),
        ("outcome", "win_rate"): win_rate(trade_results),
        ("outcome", "loss_rate"): loss_rate(trade_results),
        ("outcome", "expectancy"): expectancy(trade_results),
        ("outcome", "trade_standard_deviation"): trade_standard_deviation(trade_results),
        ("outcome", "expectancy_t_stat"): expectancy_t_stat(trade_results),
        ("outcome", "average_win"): average_win(trade_results),
        ("outcome", "average_loss"): average_loss(trade_results),
        ("outcome", "payoff_ratio"): payoff_ratio(trade_results),
        ("outcome", "gross_profit"): gross_profit(trade_results),
        ("outcome", "gross_loss"): gross_loss(trade_results),
        ("outcome", "profit_factor"): profit_factor(trade_results),
        ("outcome", "best_trade"): best_trade(trade_results),
        ("outcome", "worst_trade"): worst_trade(trade_results),
        ("outcome", "max_consecutive_wins"): float(max_consecutive_wins(trade_results)),
        ("outcome", "max_consecutive_losses"): float(max_consecutive_losses(trade_results)),
    }

    if trades is not None and sessions is not None:
        values[("timing", "trade_frequency")] = trade_frequency(
            trades, sessions, annualized=annualized, periods_per_year=periods_per_year
        )
        values[("timing", "exposure")] = exposure(trades, sessions)
        values[("timing", "average_holding_period")] = average_holding_period(trades, sessions)

    metrics = pd.Series(
        list(values.values()),
        index=pd.MultiIndex.from_tuples(list(values), names=["section", "metric"]),
        dtype="float64",
        name="value",
    )
    return TradeSummary(metrics=metrics, annualized_frequency=annualized)


# --- Helpers ---


def _strategies_frame(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Validated strategies as columns; an unnamed Series becomes ``strategy``."""
    if isinstance(returns, pd.Series):
        series = require_observations(require_returns(returns))
        name: Hashable = DEFAULT_STRATEGY_NAME if series.name is None else series.name
        return series.to_frame(name)
    # Anything that is not a Series is validated as a DataFrame and rejected there.
    return require_observations(require_returns(returns))


def _describe_risk_free(risk_free: float | pd.Series) -> float | str:
    """The annual rate, or the name of the periodic series, for the record."""
    if isinstance(risk_free, pd.Series):
        return "periodic series" if risk_free.name is None else str(risk_free.name)
    return float(risk_free)


def _format_table(table: pd.DataFrame) -> str:
    """Raw text rendering shared by both summaries."""
    return table.to_string(
        float_format=lambda value: f"{value:.{TEXT_DECIMALS}f}", na_rep=TEXT_MISSING
    )
