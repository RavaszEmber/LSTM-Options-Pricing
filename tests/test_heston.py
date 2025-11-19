# test Black-Scholes Model (black-scholes.py)

import unittest
from heston import HestonModel

class TestHestonModel(unittest.TestCase):

    def setUp(self):
        # Init model object before every test
        self.heston_model = HestonModel()
        pass

    def test_basic_test(self):
        # Testing against this calculator (https://kluge.in-chemnitz.de/tools/pricer/heston_price.php)
        predicted_price = self.heston_model.predicted_call_price(current_price=31.55, exercise_price=22.75, time_to_maturity=3.5, risk_free_rate=0.40, initial_volatility=0.05 ** 2, longterm_avg_vol= 0.05 ** 2, vol_of_vol= 0.3, strength_of_mean_reversion= 2, correlation= -0.5, lambda_val= 0)
        # predicted_price = self.heston_model.predicted_call_price(kappa=2, lambda_val=0, rho=-0.5, eta=0.3, K=22.75, r=0.40, theta=0.05 ** 2, V_0=0.05 ** 2, S_0=31.55, T=3.5)
        expected_price = 25.72842137 # The online calculator caps the domestic rate (given from r). The r I use here is too high. 27.60 on BS with r = 0.50
        # Changed r to 0.40 to check against calculator
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_at_the_money(self):
        predicted_price = self.heston_model.predicted_call_price(current_price=100, exercise_price=100, time_to_maturity=0.5, risk_free_rate=0.05, initial_volatility=0.2 ** 2, longterm_avg_vol= 0.2 ** 2, vol_of_vol= 0.3, strength_of_mean_reversion= 2, correlation= -0.5, lambda_val= 0)
        expected_price = 6.819712804 # The online calculator gives this value. 6.89 on BS
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_in_the_money(self):
        predicted_price = self.heston_model.predicted_call_price(current_price=120, exercise_price=100, time_to_maturity=1, risk_free_rate=0.03, initial_volatility=0.25 ** 2, longterm_avg_vol= 0.25 ** 2, vol_of_vol= 0.3, strength_of_mean_reversion= 2, correlation= -0.5, lambda_val= 0)
        expected_price = 26.26835625 # Online calculator gives this value. 25.91 on BS
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_out_of_the_money(self):
        predicted_price = self.heston_model.predicted_call_price(current_price=80, exercise_price=100, time_to_maturity=0.25, risk_free_rate=0.04, initial_volatility=0.3 ** 2, longterm_avg_vol= 0.3 ** 2, vol_of_vol= 0.3, strength_of_mean_reversion= 2, correlation= -0.5, lambda_val= 0)
        expected_price = 0.3278156152 # Online calculator gives this value. 0.47 on BS
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_verify_other_settings(self):
        predicted_price = self.heston_model.predicted_call_price(current_price=80, exercise_price=100, time_to_maturity=0.25, risk_free_rate=0.10, initial_volatility=0.1 ** 2, longterm_avg_vol= 0.5 ** 2, vol_of_vol= 0.2, strength_of_mean_reversion= 3, correlation= -0.25, lambda_val= 0)
        expected_price = 0.4359045169 # Online calculator gives this value.
        self.assertAlmostEqual(predicted_price, expected_price, places=2)





