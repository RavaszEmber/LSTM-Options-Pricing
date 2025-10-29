# test Black-Scholes Model (black-scholes.py)

import unittest
from black_scholes import BlackScholesModel

class TestBlackScholesModel(unittest.TestCase):

    def setUp(self):
        # Init model object before every test
        self.bs_model = BlackScholesModel()

    def test_basic_test(self):
        # Testing against this calculator (https://goodcalculators.com/black-scholes-calculator/, which agrees with https://www.omnicalculator.com/finance/black-scholes)
        predicted_price = self.bs_model.predicted_call_price(
            current_price=31.55,
            exercise_price=22.75,
            time_to_maturity=3.5,
            volatility=0.05,
            risk_free_rate=0.50
        )
        expected_price = 27.60
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_at_the_money(self):
        predicted_price = self.bs_model.predicted_call_price(
            current_price=100,
            exercise_price=100,
            time_to_maturity=0.5,
            volatility=0.2,
            risk_free_rate=0.05
        )
        expected_price = 6.89
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_in_the_money(self):
        predicted_price = self.bs_model.predicted_call_price(
            current_price=120,
            exercise_price=100,
            time_to_maturity=1.0,
            volatility=0.25,
            risk_free_rate=0.03
        )
        expected_price = 25.91
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_out_of_the_money(self):
        predicted_price = self.bs_model.predicted_call_price(
            current_price=80,
            exercise_price=100,
            time_to_maturity=0.25,
            volatility=0.3,
            risk_free_rate=0.04
        )
        expected_price = 0.47
        self.assertAlmostEqual(predicted_price, expected_price, places=2)





