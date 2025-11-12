# Discounted Cash Flow Comparison App

This project provides a command-line utility for valuing a business with several of the most widely used discounted cash flow approaches. The tool delivers both a side-by-side summary and detailed assumptions for:

- `FCFF (Free Cash Flow to Firm)` using the weighted-average cost of capital (WACC)
- `FCFE (Free Cash Flow to Equity)` using the cost of equity
- `Dividend Discount Model (Gordon Growth)` for dividend-paying firms

The script consolidates the results so you can compare how each methodology values the same set of operating forecasts.

## Requirements

- Python 3.9 or later
- Optional: [`yfinance`](https://pypi.org/project/yfinance/) for live ticker retrieval (`pip install yfinance`)

## Quick start

```bash
python3 dcf.py
```

Running the script without arguments uses the illustrative assumptions embedded in `dcf.py` and prints a comparison table.

### Pull data from Yahoo Finance

Provide a ticker to derive five-year forecasts directly from Yahoo Finance via `yfinance`:

```bash
python3 dcf.py --ticker AAPL
```

The script estimates FCFF, FCFE, and dividend growth trajectories from historical statements, computes discount rates from a simple CAPM heuristic, and then falls back to the embedded defaults if data is missing. Network access and the `yfinance` package are required for this mode.

### Optional arguments

- `--ticker AAPL` — auto-build forecasts using Yahoo Finance data (requires `yfinance`)
- `--config /path/to/assumptions.json` — load custom forecasts and terminal assumptions
- `--shares-outstanding 15000000` — override share count when producing per-share outputs
- `--output-format table|verbose|json` — choose a compact table (default), a more descriptive narrative, or JSON

## Configuration file

You can supply a JSON document with the inputs for any subset of the models. Omitted models are skipped. The top-level schema is:

```json
{
  "shares_outstanding": 12000000,
  "fcff": {
    "cash_flows": [18500000, 20725000, 23048750, 25518000, 28195000],
    "discount_rate": 0.085,
    "terminal": {
      "method": "perpetual_growth",
      "growth_rate": 0.025
    },
    "net_debt": 75000000,
    "non_operating_assets": 12500000,
    "notes": "Five-year FCFF forecast with a conservative terminal growth assumption."
  },
  "fcfe": {
    "cash_flows": [12000000, 13200000, 14520000, 15972000, 17569000],
    "discount_rate": 0.098,
    "terminal": {
      "method": "perpetual_growth",
      "growth_rate": 0.03
    },
    "notes": "FCFE derived after net borrowing and preferred dividends."
  },
  "dividend_discount": {
    "dividends": [1.25, 1.38, 1.52, 1.67, 1.84],
    "discount_rate": 0.095,
    "terminal": {
      "method": "perpetual_growth",
      "growth_rate": 0.04
    },
    "payout_ratio": 0.42,
    "notes": "Per-share dividend forecast following a two-year ramp to long-run growth."
  }
}
```

Terminal specifications accept three methods:

- `perpetual_growth` — supply `growth_rate` (and optional `cash_flow` if it differs from the final explicit forecast)
- `exit_multiple` — supply `multiple` and optionally a `cash_flow` or metric level in the terminal year
- `explicit` — supply an explicit `value` for the terminal value

## Model overview

- **FCFF (Free Cash Flow to Firm)** discounts operating cash flows available to all capital providers using WACC. The model outputs an enterprise value, which is then adjusted for net debt and non-operating assets to arrive at equity value.
- **FCFE (Free Cash Flow to Equity)** discounts cash flows available purely to equity holders using the cost of equity. Because the cash flows already account for debt financing, the output is directly an equity value.
- **Dividend Discount Model (Gordon Growth)** focuses on per-share dividends and applies the Gordon growth perpetuity to the final period. When a share count is provided, the resulting per-share value is scaled into total equity value for comparability.

Each model reports both the present value of the explicit forecast period and the contribution from the terminal value so you can quickly gauge how assumptions are driving the valuation.

## Output formats

- **table** (default) — concise side-by-side comparison
- **verbose** — narrative summary that includes discount rates, terminal horizon, and notes
- **json** — structured output suitable for downstream analysis or reporting

Example verbose usage:

```bash
python3 dcf.py --output-format verbose
```

## Next steps

- Swap in your own assumptions via a JSON file to evaluate alternative scenarios.
- Extend `dcf.py` with additional models (e.g., residual income) following the same pattern if you need more coverage.

