from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_CONFIG: Dict[str, Any] = {
    "shares_outstanding": 12_000_000,
    "fcff": {
        "cash_flows": [18_500_000, 20_725_000, 23_048_750, 25_518_000, 28_195_000],
        "discount_rate": 0.085,
        "terminal": {
            "method": "perpetual_growth",
            "growth_rate": 0.025,
        },
        "net_debt": 75_000_000,
        "non_operating_assets": 12_500_000,
        "notes": "Five-year FCFF forecast with a conservative terminal growth assumption.",
    },
    "fcfe": {
        "cash_flows": [12_000_000, 13_200_000, 14_520_000, 15_972_000, 17_569_000],
        "discount_rate": 0.098,
        "terminal": {
            "method": "perpetual_growth",
            "growth_rate": 0.03,
        },
        "notes": "FCFE derived after net borrowing and preferred dividends.",
    },
    "dividend_discount": {
        "dividends": [1.25, 1.38, 1.52, 1.67, 1.84],
        "discount_rate": 0.095,
        "terminal": {
            "method": "perpetual_growth",
            "growth_rate": 0.04,
        },
        "payout_ratio": 0.42,
        "notes": "Per-share dividend forecast following a two-year ramp to long-run growth.",
    },
}


try:  # pragma: no cover - optional dependency
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yf = None  # type: ignore[assignment]


def require_yfinance() -> Any:
    if yf is not None:
        return yf
    try:
        return importlib.import_module("yfinance")
    except ImportError as exc:  # pragma: no cover - user environment guard
        raise ImportError(
            "The yfinance package is required when using the --ticker option. "
            "Install it with `pip install yfinance`."
        ) from exc


@dataclass
class ModelResult:
    model_name: str
    enterprise_value: Optional[float]
    equity_value: Optional[float]
    per_share_value: Optional[float]
    pv_cash_flows: float
    pv_terminal_value: float
    undiscounted_terminal_value: float
    assumptions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULT_CONFIG
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("The configuration file must define a JSON object.")
    return data


def discount_cash_flows(cash_flows: Iterable[float], discount_rate: float) -> Dict[str, Any]:
    pv_sum = 0.0
    breakdown: List[Dict[str, float]] = []

    for idx, cash_flow in enumerate(cash_flows, start=1):
        discount_factor = (1.0 + discount_rate) ** idx
        present_value = cash_flow / discount_factor
        breakdown.append(
            {
                "year": idx,
                "cash_flow": cash_flow,
                "discount_factor": discount_factor,
                "present_value": present_value,
            }
        )
        pv_sum += present_value

    return {"pv": pv_sum, "breakdown": breakdown}


def get_statement_value(frame: Any, labels: Sequence[str], column: Any, default: float = 0.0) -> float:
    if frame is None or getattr(frame, "empty", False):
        return float(default)
    for label in labels:
        if label in frame.index:
            raw_value = frame.at[label, column]
            if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
                continue
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                continue
    return float(default)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def calculate_wacc(
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
    debt_ratio: float,
) -> float:
    """
    Compute the weighted average cost of capital (WACC).

    Parameters
    ----------
    cost_of_equity : float
        Required return demanded by equity investors (e.g., via CAPM).
    cost_of_debt : float
        Pre-tax cost of debt financing.
    tax_rate : float
        Marginal tax rate applied to interest expense.
    debt_ratio : float
        Portion of the capital structure financed with debt (D / (D + E)).
        Must fall between 0 and 1.

    Returns
    -------
    float
        The overall blended discount rate for all capital providers.
    """

    debt_ratio = clamp(debt_ratio, 0.0, 1.0)
    equity_ratio = 1.0 - debt_ratio
    after_tax_cost_of_debt = cost_of_debt * (1.0 - clamp(tax_rate, 0.0, 1.0))
    return equity_ratio * cost_of_equity + debt_ratio * after_tax_cost_of_debt


def infer_growth_rate(
    history: Sequence[float],
    default: float,
    min_growth: float,
    max_growth: float,
) -> float:
    usable = [float(v) for v in history if v is not None and abs(float(v)) > 1e-6]
    if len(usable) >= 2 and usable[0] > 0 and usable[-1] > 0:
        periods = len(usable) - 1
        try:
            growth = (usable[-1] / usable[0]) ** (1.0 / periods) - 1.0
        except (ZeroDivisionError, OverflowError, ValueError):
            growth = default
    else:
        growth = default
    return clamp(growth, min_growth, max_growth)


def project_forward(base: float, growth: float, years: int) -> List[float]:
    projections: List[float] = []
    current = float(base)
    for _ in range(years):
        current *= 1.0 + growth
        projections.append(current)
    return projections


def build_config_from_yfinance(symbol: str, forecast_years: int = 5) -> Dict[str, Any]:
    yf_module = require_yfinance()
    ticker = yf_module.Ticker(symbol)

    cashflow_df = ticker.cashflow
    if cashflow_df is None or cashflow_df.empty:
        raise ValueError(f"No cash flow statement data available for ticker {symbol}.")

    periods = list(cashflow_df.columns)
    if not periods:
        raise ValueError(f"Cash flow statement for {symbol} is missing period data.")

    periods_chrono = list(reversed(periods))

    fcff_history: List[float] = []
    fcfe_history: List[float] = []

    CFO_ROWS = (
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
        "Net Cash Provided By Operating Activities",
    )
    CAPEX_ROWS = ("Capital Expenditures", "CapitalExpenditures")
    NET_BORROWINGS_ROWS = ("Net Borrowings", "NetBorrowings")
    DIVIDENDS_PAID_ROWS = ("Cash Dividends Paid", "Dividends Paid", "CashDividendsPaid")

    for period in periods_chrono:
        operating_cash_flow = get_statement_value(cashflow_df, CFO_ROWS, period, default=0.0)
        capital_expenditures = get_statement_value(cashflow_df, CAPEX_ROWS, period, default=0.0)
        net_borrowings = get_statement_value(cashflow_df, NET_BORROWINGS_ROWS, period, default=0.0)
        fcff_history.append(operating_cash_flow + capital_expenditures)
        fcfe_history.append(operating_cash_flow + capital_expenditures + net_borrowings)

    if not fcff_history:
        raise ValueError(f"Unable to derive FCFF history for ticker {symbol}.")

    fcff_base = fcff_history[-1]
    fcfe_base = fcfe_history[-1] if fcfe_history else fcff_base

    fcff_growth = infer_growth_rate(fcff_history, default=0.04, min_growth=-0.15, max_growth=0.18)
    fcfe_growth = infer_growth_rate(fcfe_history, default=0.05, min_growth=-0.2, max_growth=0.2)

    fcff_forecast = project_forward(fcff_base, fcff_growth, forecast_years)
    fcfe_forecast = project_forward(fcfe_base, fcfe_growth, forecast_years)

    dividends_series = getattr(ticker, "dividends", None)
    dividend_forecast: Optional[List[float]] = None
    dividend_growth = 0.04
    if dividends_series is not None and not dividends_series.empty:
        dividends_series = dividends_series.sort_index()
        annual_dividends = dividends_series.resample("YE").sum()
        annual_dividends = annual_dividends[annual_dividends > 0]
        if not annual_dividends.empty:
            dividend_history = [float(value) for value in annual_dividends.tail(6).tolist()]
            dividend_growth = infer_growth_rate(dividend_history, default=0.04, min_growth=0.0, max_growth=0.1)
            last_dividend = float(annual_dividends.iloc[-1])
            dividend_forecast = project_forward(last_dividend, dividend_growth, forecast_years)

    balance_sheet_df = ticker.balance_sheet
    net_debt = 0.0
    if balance_sheet_df is not None and not balance_sheet_df.empty:
        balance_periods = list(balance_sheet_df.columns)
        latest_balance_period = balance_periods[0]
        total_debt = get_statement_value(
            balance_sheet_df,
            ("Total Debt", "Long Term Debt", "Short Long Term Debt"),
            latest_balance_period,
            default=0.0,
        )
        cash_and_equivalents = get_statement_value(
            balance_sheet_df,
            ("Cash And Cash Equivalents", "Cash", "Cash Equivalents"),
            latest_balance_period,
            default=0.0,
        )
        net_debt = total_debt - cash_and_equivalents

    shares_outstanding: Optional[float] = None
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info is not None:
        shares_outstanding = getattr(fast_info, "shares_outstanding", None)
    if not shares_outstanding:
        info = getattr(ticker, "info", {}) or {}
        shares_outstanding = info.get("sharesOutstanding") or info.get("shares_outstanding")
    if shares_outstanding:
        shares_outstanding = float(shares_outstanding)

    info = getattr(ticker, "info", {}) or {}
    beta = None
    if fast_info is not None:
        beta = getattr(fast_info, "beta", None)
    if not beta:
        beta = info.get("beta")
    if not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        beta = 1.0

    risk_free_rate = 0.04
    market_risk_premium = 0.05
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    cost_of_equity = clamp(cost_of_equity, 0.06, 0.16)

    raw_yield = info.get("yield") if isinstance(info, dict) else None
    if isinstance(raw_yield, (int, float)) and math.isfinite(raw_yield):
        cost_of_debt_input = float(raw_yield)
    else:
        cost_of_debt_input = 0.05
    cost_of_debt = clamp(cost_of_debt_input, 0.03, 0.08)

    raw_tax_rate = info.get("effectiveTaxRate") if isinstance(info, dict) else None
    if isinstance(raw_tax_rate, (int, float)) and math.isfinite(raw_tax_rate):
        effective_tax_rate = float(raw_tax_rate)
    else:
        effective_tax_rate = 0.21
    tax_rate = clamp(effective_tax_rate, 0.0, 0.35)
    target_leverage = 0.3
    wacc = calculate_wacc(cost_of_equity, cost_of_debt, tax_rate, target_leverage)
    wacc = clamp(wacc, 0.07, 0.12)

    fcff_terminal_growth = clamp(0.5 * fcff_growth + 0.02, 0.01, 0.03)
    fcfe_terminal_growth = clamp(0.5 * fcfe_growth + 0.02, 0.0, 0.04)
    dividend_terminal_growth = clamp(0.5 * dividend_growth + 0.02, 0.0, 0.04)

    dividends_paid_latest = get_statement_value(
        cashflow_df,
        DIVIDENDS_PAID_ROWS,
        periods[0],
        default=0.0,
    )
    financials_df = ticker.financials
    payout_ratio: Optional[float] = None
    if financials_df is not None and not financials_df.empty:
        financials_periods = list(financials_df.columns)
        if financials_periods:
            net_income_latest = get_statement_value(
                financials_df,
                ("Net Income", "NetIncome"),
                financials_periods[0],
                default=0.0,
            )
            if net_income_latest > 0:
                payout_ratio = clamp(abs(dividends_paid_latest) / net_income_latest, 0.0, 1.0)

    config: Dict[str, Any] = {
        "shares_outstanding": shares_outstanding,
        "fcff": {
            "cash_flows": fcff_forecast,
            "discount_rate": wacc,
            "terminal": {
                "method": "perpetual_growth",
                "growth_rate": fcff_terminal_growth,
                "cash_flow": fcff_forecast[-1],
            },
            "net_debt": net_debt,
            "non_operating_assets": 0.0,
            "notes": f"FCFF forecast derived from operating cash flow and capital expenditure history for {symbol}.",
        },
        "fcfe": {
            "cash_flows": fcfe_forecast,
            "discount_rate": cost_of_equity,
            "terminal": {
                "method": "perpetual_growth",
                "growth_rate": fcfe_terminal_growth,
                "cash_flow": fcfe_forecast[-1],
            },
            "notes": f"FCFE forecast derived from operating cash flow, capital expenditure, and net borrowing history for {symbol}.",
        },
    }

    if dividend_forecast:
        config["dividend_discount"] = {
            "dividends": dividend_forecast,
            "discount_rate": cost_of_equity,
            "terminal": {
                "method": "perpetual_growth",
                "growth_rate": dividend_terminal_growth,
                "cash_flow": dividend_forecast[-1],
            },
            "payout_ratio": payout_ratio,
            "notes": f"Dividends per share extrapolated from trailing cash distributions for {symbol}.",
        }

    return config


def compute_terminal_value(
    cash_flows: List[float],
    discount_rate: float,
    terminal_spec: Dict[str, Any],
) -> Dict[str, float]:
    if not cash_flows:
        raise ValueError("At least one cash flow is required to compute a terminal value.")

    method = terminal_spec.get("method", "perpetual_growth").lower()
    horizon = int(terminal_spec.get("horizon_year", len(cash_flows)))

    if method == "perpetual_growth":
        growth_rate = float(terminal_spec["growth_rate"])
        if discount_rate <= growth_rate:
            raise ValueError(
                f"Discount rate ({discount_rate:.4f}) must exceed terminal growth ({growth_rate:.4f}) for a perpetual growth model."
            )
        base_cash_flow = float(terminal_spec.get("cash_flow", cash_flows[-1]))
        terminal_value = base_cash_flow * (1.0 + growth_rate) / (discount_rate - growth_rate)
    elif method == "exit_multiple":
        multiple = float(terminal_spec["multiple"])
        base_metric = float(terminal_spec.get("cash_flow", cash_flows[-1]))
        terminal_value = multiple * base_metric
    elif method == "explicit":
        terminal_value = float(terminal_spec["value"])
    else:
        raise ValueError(f"Unsupported terminal value method: {method!r}")

    discounted = terminal_value / (1.0 + discount_rate) ** horizon

    return {
        "terminal_value": terminal_value,
        "discounted_terminal_value": discounted,
        "horizon_year": horizon,
        "method": method,
    }


def fcff_valuation(params: Dict[str, Any], shares_outstanding: Optional[float]) -> ModelResult:
    cash_flows = list(map(float, params["cash_flows"]))
    wacc = float(params["discount_rate"])

    pv_data = discount_cash_flows(cash_flows, wacc)
    terminal_data = compute_terminal_value(cash_flows, wacc, params.get("terminal", {}))

    enterprise_value = pv_data["pv"] + terminal_data["discounted_terminal_value"]
    net_debt = float(params.get("net_debt", 0.0))
    non_operating_assets = float(params.get("non_operating_assets", 0.0))
    equity_value = enterprise_value - net_debt + non_operating_assets
    per_share = equity_value / shares_outstanding if shares_outstanding else None

    assumptions = {
        "discount_rate": wacc,
        "cash_flow_years": len(cash_flows),
        "net_debt": net_debt,
        "non_operating_assets": non_operating_assets,
        "notes": params.get("notes"),
        "terminal": terminal_data,
        "pv_breakdown": pv_data["breakdown"],
    }

    return ModelResult(
        model_name="FCFF (Free Cash Flow to Firm)",
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        per_share_value=per_share,
        pv_cash_flows=pv_data["pv"],
        pv_terminal_value=terminal_data["discounted_terminal_value"],
        undiscounted_terminal_value=terminal_data["terminal_value"],
        assumptions=assumptions,
    )


def fcfe_valuation(params: Dict[str, Any], shares_outstanding: Optional[float]) -> ModelResult:
    cash_flows = list(map(float, params["cash_flows"]))
    cost_of_equity = float(params["discount_rate"])

    pv_data = discount_cash_flows(cash_flows, cost_of_equity)
    terminal_data = compute_terminal_value(cash_flows, cost_of_equity, params.get("terminal", {}))

    equity_value = pv_data["pv"] + terminal_data["discounted_terminal_value"]
    per_share = equity_value / shares_outstanding if shares_outstanding else None

    assumptions = {
        "discount_rate": cost_of_equity,
        "cash_flow_years": len(cash_flows),
        "notes": params.get("notes"),
        "terminal": terminal_data,
        "pv_breakdown": pv_data["breakdown"],
    }

    return ModelResult(
        model_name="FCFE (Free Cash Flow to Equity)",
        enterprise_value=None,
        equity_value=equity_value,
        per_share_value=per_share,
        pv_cash_flows=pv_data["pv"],
        pv_terminal_value=terminal_data["discounted_terminal_value"],
        undiscounted_terminal_value=terminal_data["terminal_value"],
        assumptions=assumptions,
    )


def dividend_discount_valuation(
    params: Dict[str, Any],
    shares_outstanding: Optional[float],
) -> ModelResult:
    dividends = list(map(float, params["dividends"]))
    cost_of_equity = float(params["discount_rate"])

    pv_data = discount_cash_flows(dividends, cost_of_equity)
    terminal_data = compute_terminal_value(dividends, cost_of_equity, params.get("terminal", {}))

    per_share_value = pv_data["pv"] + terminal_data["discounted_terminal_value"]
    equity_value = per_share_value * shares_outstanding if shares_outstanding else None

    assumptions = {
        "discount_rate": cost_of_equity,
        "dividend_years": len(dividends),
        "notes": params.get("notes"),
        "payout_ratio": params.get("payout_ratio"),
        "terminal": terminal_data,
        "pv_breakdown": pv_data["breakdown"],
    }

    return ModelResult(
        model_name="Dividend Discount Model (Gordon)",
        enterprise_value=None,
        equity_value=equity_value,
        per_share_value=per_share_value,
        pv_cash_flows=pv_data["pv"],
        pv_terminal_value=terminal_data["discounted_terminal_value"],
        undiscounted_terminal_value=terminal_data["terminal_value"],
        assumptions=assumptions,
    )


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "-"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs_value >= 1_000:
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def render_table(results: List[ModelResult]) -> str:
    headers = [
        "Model",
        "Enterprise Value",
        "Equity Value",
        "Per Share",
        "PV Explicit",
        "PV Terminal",
    ]
    col_widths = [max(len(header), 16) for header in headers]
    rows: List[List[str]] = []

    for result in results:
        row = [
            result.model_name,
            format_currency(result.enterprise_value),
            format_currency(result.equity_value),
            format_currency(result.per_share_value),
            format_currency(result.pv_cash_flows),
            format_currency(result.pv_terminal_value),
        ]
        rows.append(row)

    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def format_row(row_values: List[str]) -> str:
        return "  ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(row_values))

    output_lines = [format_row(headers), format_row(["-" * width for width in col_widths])]
    output_lines.extend(format_row(row) for row in rows)
    return "\n".join(output_lines)


def render_verbose(results: List[ModelResult]) -> str:
    lines: List[str] = []
    for result in results:
        lines.append(f"{result.model_name}")
        lines.append("-" * len(result.model_name))
        lines.append(f"Enterprise value: {format_currency(result.enterprise_value)}")
        lines.append(f"Equity value:     {format_currency(result.equity_value)}")
        lines.append(f"Per-share value:  {format_currency(result.per_share_value)}")
        lines.append(f"Present value of explicit forecast: {format_currency(result.pv_cash_flows)}")
        lines.append(f"Present value of terminal value:     {format_currency(result.pv_terminal_value)}")
        terminal_info = result.assumptions.get("terminal", {})
        if terminal_info:
            lines.append(
                f"Terminal method: {terminal_info.get('method', 'unknown')} | "
                f"Undiscounted: {format_currency(result.undiscounted_terminal_value)} | "
                f"Horizon: year {terminal_info.get('horizon_year')}"
            )
        discount_rate = result.assumptions.get("discount_rate")
        if discount_rate is not None:
            lines.append(f"Discount rate: {format_rate(discount_rate)}")
        notes = result.assumptions.get("notes")
        if notes:
            lines.append(f"Notes: {notes}")
        lines.append("")
    return "\n".join(lines).rstrip()


def evaluate_models(config: Dict[str, Any], shares_override: Optional[float]) -> List[ModelResult]:
    shares_outstanding = shares_override or config.get("shares_outstanding")
    if shares_outstanding is not None:
        shares_outstanding = float(shares_outstanding)

    results: List[ModelResult] = []

    if "fcff" in config:
        results.append(fcff_valuation(config["fcff"], shares_outstanding))
    if "fcfe" in config:
        results.append(fcfe_valuation(config["fcfe"], shares_outstanding))
    if "dividend_discount" in config:
        results.append(dividend_discount_valuation(config["dividend_discount"], shares_outstanding))

    if not results:
        raise ValueError("No valuation models were specified in the configuration.")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discounted cash flow comparison tool with FCFF, FCFE, and dividend discount models."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON file containing forecast assumptions. Defaults to an illustrative example.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Load forecasts automatically using yfinance for the provided ticker symbol.",
    )
    parser.add_argument(
        "--shares-outstanding",
        type=float,
        help="Override shares outstanding when computing per-share values.",
    )
    parser.add_argument(
        "--output-format",
        choices=("table", "verbose", "json"),
        default="table",
        help="Choose between tabular, verbose, or JSON output formats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.ticker:
            config = build_config_from_yfinance(args.ticker)
        else:
            config = load_config(args.config)
        results = evaluate_models(config, args.shares_outstanding)
    except Exception as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Error: {exc}") from exc

    if args.output_format == "json":
        payload = {"results": [result.to_dict() for result in results]}
        print(json.dumps(payload, indent=2))
    elif args.output_format == "verbose":
        print(render_verbose(results))
    else:
        print(render_table(results))


if __name__ == "__main__":
    main()

