# Implement Heston model from section 4.3.2 in paper https://link.springer.com/article/10.1007/s10203-025-00518-9

# Heston doesn't have a closed form solution. Going to use the characteristic function with numerical integration

# Apparently I can use the COS or Carr-Madan for better performance using Fourier expansion

# TODO This can be cleaned up a lot

import math
import scipy.stats
import scipy.integrate
import cmath
import numpy as np

class HestonModel:
    """
    Contains methods to calculate the predicted call price via the Heston model
    """

    def __init__(self, kappa, lambda_val, rho, eta, K, r, theta, V_0, S_0, T):
        self.kappa = kappa
        self.lambda_val = lambda_val
        self.rho = rho
        self.eta = eta
        self.K = K
        self.r = r
        self.theta = theta
        self.V_0 = V_0
        self.S_0 = S_0
        self.T = T
        pass

    # In my code j is the unit imaginary number (i in the paper) and i is the subscript (j in the paper)

    def b(self, i):
        ans = self.kappa + self.lambda_val + (i - 2) * self.rho * self.eta
        return ans
    
    def P(self, i, T): # u needs to be integrated over, it isnt an input to this
        b_i = self.b(i)

        def integrand(u):

            ########################

            # Could make these lambdas but they're fine as they are

            d_i = cmath.sqrt((b_i - self.rho * self.eta * u * 1j) ** 2 + self.eta ** 2 * (u ** 2 + (-1) ** i * u * 1j))

            g_i = (b_i - self.rho * self.eta * u * 1j + d_i) / (b_i - self.rho * self.eta * u * 1j - d_i)

            C_i = self.r * u * T * 1j + self.kappa * self.theta / (self.eta ** 2) * ((b_i - self.rho * self.eta * u * 1j + d_i) * T - 2 * cmath.log((1 - g_i * cmath.exp(d_i * T)) / (1 - g_i)))
            
            D_i = ((b_i - self.rho * self.eta * u * 1j + d_i) / self.eta **2) * ((1 - cmath.exp(d_i * T)) / (1 - g_i * cmath.exp(d_i * T)))

            f_i = cmath.exp(1j * u * cmath.log(self.S_0) + C_i + D_i * self.V_0)

            ########################

            integrand = ((cmath.exp(-1j * u * cmath.log(self.K)) * f_i) / (1j * u)).real
            return integrand
        
        integrated_value, abserr = scipy.integrate.quad(integrand, 0.00000001, 100, limit=200, epsabs=1e-8, epsrel=1e-8) # Technically this integrates to inf, but limiting to 100 for numerical overflow. Also starting at 0.00000001 instead of 0 to avoid divide by 0
        # abs error is an estimate of the absolute error

        ans = 0.5 + 1/math.pi * integrated_value
        return ans


    # TODO, could clean this up and add a description like below.
    def call_price(self):
        P_1 = self.P(1, self.T)
        P_2 = self.P(2, self.T)
        ans = self.S_0 * P_1 - math.exp(-self.r * self.T) * self.K * P_2
        return ans


    # def predicted_call_price(self, current_price : float, exercise_price : float, time_to_maturity : float, volatility : float, risk_free_rate : float) -> float:
    #     """
    #     Calculate European call option price using Heston model.

    #     Parameters
    #     ----------
    #     current_price : float
    #         The current price of the underlying asset (S_0).
    #     exercise_price : float
    #         The strike (exercise) price of the option (K).
    #     time_to_maturity_days : float
    #         Time until option maturity annualized (T)
    #     volatility : float
    #         Some measure of the annualized volatility of the underlying asset (sigma).
    #     risk_free_rate : float
    #         Annualized risk-free interest rate (r)

    #     Returns
    #     -------
    #     float
    #         The prediced call option price (C).
    #     """


    #     pass






# This is just here for reference

    # These have all been moved inside P
    
    # def d(self, i, u):
    #     b_i = self.b(i)
    #     ans = cmath.sqrt((b_i - self.rho * self.eta * u * 1j) ** 2 + self.eta ** 2 * (u ** 2 + (-1) ** i * u * 1j))
    #     return ans

    # def g(self, i, u):
    #     b_i = self.b(i)
    #     d_i = self.d(i, u)
    #     ans = (b_i - self.rho * self.eta * u * 1j + d_i) / (b_i - self.rho * self.eta * u * 1j - d_i)
    #     return ans

    # def C(self, u, i, T):
    #     b_i = self.b(i)
    #     d_i = self.d(i, u)
    #     g_i = self.g(i, u)
    #     ans = self.r * u * T * 1j + self.kappa * self.theta / (self.eta ** 2) * ((b_i - self.rho * self.eta * u * 1j + d_i) * T - 2 * cmath.log((1 - g_i * cmath.exp(d_i * T)) / (1 - g_i)))
    #     return ans
    
    # def D(self, u, i, T):
    #     b_i = self.b(i)
    #     d_i = self.d(i, u)
    #     g_i = self.g(i, u)
    #     ans = ((b_i - self.rho * self.eta * u * 1j + d_i) / self.eta **2) * ((1 - cmath.exp(d_i * T)) / (1 - g_i * cmath.exp(d_i * T)))
    #     return ans

    # def f(self, u, i, T):
    #     C_i = self.C(u, i, T)
    #     D_i = self.D(u, i, T)
    #     ans = cmath.exp(1j * u * cmath.log(self.S_0) + C_i + D_i * self.V_0)
    #     return ans