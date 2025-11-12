import math
import unittest

from dcf import (
    ModelResult,
    calculate_wacc,
    compute_terminal_value,
    discount_cash_flows,
    dividend_discount_valuation,
    evaluate_models,
    fcfe_valuation,
    fcff_valuation,
)


class DiscountCashFlowTests(unittest.TestCase):
    def test_discount_cash_flows(self) -> None:
        flows = [100.0, 110.0]
        rate = 0.1
        result = discount_cash_flows(flows, rate)

        expected_pv = 100.0 / 1.1 + 110.0 / (1.1**2)
        self.assertAlmostEqual(result["pv"], expected_pv, places=6)
        self.assertEqual(len(result["breakdown"]), 2)
        self.assertAlmostEqual(result["breakdown"][0]["present_value"], 100.0 / 1.1, places=6)

    def test_compute_terminal_value_perpetual_growth(self) -> None:
        flows = [100.0, 110.0]
        rate = 0.1
        terminal = {"method": "perpetual_growth", "growth_rate": 0.03}
        result = compute_terminal_value(flows, rate, terminal)

        expected_terminal = 110.0 * 1.03 / (rate - 0.03)
        expected_discounted = expected_terminal / (1.0 + rate) ** len(flows)

        self.assertAlmostEqual(result["terminal_value"], expected_terminal, places=6)
        self.assertAlmostEqual(result["discounted_terminal_value"], expected_discounted, places=6)
        self.assertEqual(result["method"], "perpetual_growth")
        self.assertEqual(result["horizon_year"], len(flows))

    def test_compute_terminal_value_exit_multiple(self) -> None:
        flows = [50.0]
        rate = 0.08
        terminal = {"method": "exit_multiple", "multiple": 8.5, "cash_flow": 60.0, "horizon_year": 1}
        result = compute_terminal_value(flows, rate, terminal)

        expected_terminal = 8.5 * 60.0
        expected_discounted = expected_terminal / (1.0 + rate)

        self.assertAlmostEqual(result["terminal_value"], expected_terminal, places=6)
        self.assertAlmostEqual(result["discounted_terminal_value"], expected_discounted, places=6)
        self.assertEqual(result["method"], "exit_multiple")
        self.assertEqual(result["horizon_year"], 1)

    def test_compute_terminal_value_explicit(self) -> None:
        flows = [75.0]
        rate = 0.09
        terminal = {"method": "explicit", "value": 900.0, "horizon_year": 2}
        result = compute_terminal_value(flows, rate, terminal)

        expected_discounted = 900.0 / (1.0 + rate) ** 2
        self.assertEqual(result["terminal_value"], 900.0)
        self.assertAlmostEqual(result["discounted_terminal_value"], expected_discounted, places=6)
        self.assertEqual(result["method"], "explicit")
        self.assertEqual(result["horizon_year"], 2)

    def test_calculate_wacc(self) -> None:
        wacc = calculate_wacc(cost_of_equity=0.12, cost_of_debt=0.06, tax_rate=0.25, debt_ratio=0.4)
        expected = 0.12 * 0.6 + 0.06 * (1.0 - 0.25) * 0.4
        self.assertAlmostEqual(wacc, expected, places=6)


class ValuationModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shares_outstanding = 100.0
        self.fcff_params = {
            "cash_flows": [100.0, 110.0],
            "discount_rate": 0.1,
            "terminal": {"method": "perpetual_growth", "growth_rate": 0.03},
            "net_debt": 50.0,
            "non_operating_assets": 10.0,
        }
        self.fcfe_params = {
            "cash_flows": [80.0, 90.0],
            "discount_rate": 0.12,
            "terminal": {"method": "perpetual_growth", "growth_rate": 0.04},
        }
        self.ddm_params = {
            "dividends": [1.0, 1.1],
            "discount_rate": 0.09,
            "terminal": {"method": "perpetual_growth", "growth_rate": 0.04},
            "payout_ratio": 0.55,
        }

    def test_fcff_valuation(self) -> None:
        result = fcff_valuation(self.fcff_params, self.shares_outstanding)

        self.assertIsInstance(result, ModelResult)
        expected_pv = 100.0 / 1.1 + 110.0 / (1.1**2)
        expected_terminal = 110.0 * 1.03 / (0.1 - 0.03) / (1.1**2)
        expected_ev = expected_pv + expected_terminal
        expected_equity = expected_ev - 50.0 + 10.0
        expected_per_share = expected_equity / self.shares_outstanding

        self.assertAlmostEqual(result.pv_cash_flows, expected_pv, places=6)
        self.assertAlmostEqual(result.pv_terminal_value, expected_terminal, places=6)
        self.assertAlmostEqual(result.enterprise_value, expected_ev, places=6)
        self.assertAlmostEqual(result.equity_value, expected_equity, places=6)
        self.assertAlmostEqual(result.per_share_value, expected_per_share, places=6)

    def test_fcfe_valuation(self) -> None:
        result = fcfe_valuation(self.fcfe_params, self.shares_outstanding)

        expected_pv = 80.0 / 1.12 + 90.0 / (1.12**2)
        expected_terminal = 90.0 * 1.04 / (0.12 - 0.04) / (1.12**2)
        expected_equity = expected_pv + expected_terminal

        self.assertAlmostEqual(result.pv_cash_flows, expected_pv, places=6)
        self.assertAlmostEqual(result.pv_terminal_value, expected_terminal, places=6)
        self.assertAlmostEqual(result.equity_value, expected_equity, places=6)
        self.assertAlmostEqual(result.per_share_value, expected_equity / self.shares_outstanding, places=6)
        self.assertIsNone(result.enterprise_value)

    def test_dividend_discount_valuation(self) -> None:
        result = dividend_discount_valuation(self.ddm_params, self.shares_outstanding)

        expected_pv = 1.0 / 1.09 + 1.1 / (1.09**2)
        expected_terminal = 1.1 * 1.04 / (0.09 - 0.04) / (1.09**2)
        expected_per_share = expected_pv + expected_terminal
        expected_equity = expected_per_share * self.shares_outstanding

        self.assertAlmostEqual(result.pv_cash_flows, expected_pv, places=6)
        self.assertAlmostEqual(result.pv_terminal_value, expected_terminal, places=6)
        self.assertAlmostEqual(result.per_share_value, expected_per_share, places=6)
        self.assertAlmostEqual(result.equity_value, expected_equity, places=6)
        self.assertIsNone(result.enterprise_value)

    def test_evaluate_models(self) -> None:
        config = {
            "shares_outstanding": self.shares_outstanding,
            "fcff": self.fcff_params,
            "fcfe": self.fcfe_params,
            "dividend_discount": self.ddm_params,
        }

        results = evaluate_models(config, None)
        self.assertEqual(len(results), 3)
        model_names = {result.model_name for result in results}
        self.assertSetEqual(
            model_names,
            {
                "FCFF (Free Cash Flow to Firm)",
                "FCFE (Free Cash Flow to Equity)",
                "Dividend Discount Model (Gordon)",
            },
        )


if __name__ == "__main__":
    unittest.main()

