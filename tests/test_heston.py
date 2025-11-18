# test Black-Scholes Model (black-scholes.py)

import unittest
from heston import HestonModel

class TestHestonModel(unittest.TestCase):

    def setUp(self):
        # Init model object before every test
        # self.heston_model = HestonModel() # Each model is probably different
        pass

    def test_basic_test(self):
        # Testing against this calculator (https://kluge.in-chemnitz.de/tools/pricer/heston_price.php)
        heston_model = HestonModel(kappa=2, lambda_val=0, rho=-0.5, eta=0.3, K=22.75, r=0.40, theta=0.05 ** 2, V_0=0.05 ** 2, S_0=31.55, T=3.5) # theta and V_0 might have to be squared (or sqrt)
        predicted_price = heston_model.call_price()
        expected_price = 25.72842137 # The online calculator caps the domestic rate (given from r). The r I use here is too high. 27.60 on BS with r = 0.50
        # Changed r to 0.40 to check against calculator
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_at_the_money(self):
        heston_model = HestonModel(kappa=2, lambda_val=0, rho=-0.5, eta=0.3, K=100, r=0.05, theta=0.2 ** 2, V_0=0.2 ** 2, S_0=100, T=0.5)
        predicted_price = heston_model.call_price()
        expected_price = 6.819712804 # The online calculator gives this value. 6.89 on BS
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_in_the_money(self):
        heston_model = HestonModel(kappa=2, lambda_val=0, rho=-0.5, eta=0.3, K=100, r=0.03, theta=0.25 ** 2, V_0=0.25 ** 2, S_0=120, T=1)
        predicted_price = heston_model.call_price()
        expected_price = 26.26835625 # Online calculator gives this value. 25.91 on BS
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_out_of_the_money(self):
        heston_model = HestonModel(kappa=2, lambda_val=0, rho=-0.5, eta=0.3, K=100, r=0.04, theta=0.3 ** 2, V_0=0.3 ** 2, S_0=80, T=0.25)
        predicted_price = heston_model.call_price()
        expected_price = 0.3278156152 # Online calculator gives this value. 0.47 on BS
        self.assertAlmostEqual(predicted_price, expected_price, places=2)

    def test_verify_other_settings(self):
        heston_model = HestonModel(kappa=3, lambda_val=0, rho=-0.25, eta=0.2, K=100, r=0.10, theta=0.5 ** 2, V_0=0.1 ** 2, S_0=80, T=0.25)
        predicted_price = heston_model.call_price()
        expected_price = 0.4359045169 # Online calculator gives this value.
        self.assertAlmostEqual(predicted_price, expected_price, places=2)





