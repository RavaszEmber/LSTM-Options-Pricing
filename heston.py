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

    def __init__(self):
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


    def predicted_call_price(self, current_price : float, exercise_price : float, time_to_maturity : float, risk_free_rate : float, initial_volatility : float, longterm_avg_vol : float, vol_of_vol : float, strength_of_mean_reversion : float, correlation : float, lambda_val : float) -> float:
        """
        Calculate European call option price using Heston model.

        Parameters
        ----------
        current_price : float
            The current price of the underlying asset (S_0).
        exercise_price : float
            The strike (exercise) price of the option (K).
        time_to_maturity_days : float
            Time until option maturity annualized (T).
        risk_free_rate : float
            Annualized risk-free interest rate (r).
        initial_volatility : float
            Some measure of the variance (volatility squared) of the underlying on the current day (V_0).
        longterm_avg_vol : float
            The longterm mean of the variance (theta).
        vol_of_vol : float
            The volatility of the variance process (eta).
        strength_of_mean_reversion : float
            The rate of mean reversion of the variance (kappa). Higher kappa reverts the variance to the longterm variance theta faster.
        correlation : float
            The correlation between the underlying asset price and its variance (rho). Usually negative, meaning when a stock price falls its volatility rises.
        lambda_val : float
            A drift adjustment term for the variance process (rho). Often set to 0. 

        Returns
        -------
        float
            The prediced call option price (C).
        """

        self.S_0 = current_price
        self.K = exercise_price
        self.T = time_to_maturity
        self.r = risk_free_rate
        self.V_0 = initial_volatility
        self.theta = longterm_avg_vol
        self.eta = vol_of_vol
        self.kappa = strength_of_mean_reversion
        self.rho = correlation
        self.lambda_val = lambda_val


        P_1 = self.P(1, self.T)
        P_2 = self.P(2, self.T)
        ans = self.S_0 * P_1 - math.exp(-self.r * self.T) * self.K * P_2

        return ans



    def predicted_call_price_vectorized(self, current_price : np.ndarray, exercise_price : np.ndarray, time_to_maturity : np.ndarray, risk_free_rate : np.ndarray, initial_volatility : np.ndarray, longterm_avg_vol : np.ndarray, vol_of_vol : np.ndarray, strength_of_mean_reversion : np.ndarray, correlation : np.ndarray, lambda_val : np.ndarray, batch_size : int = 10000, verbose : bool =False) -> np.ndarray:
        """
        Calculate European call option price using Heston model.
        This is the same as the non-vectorized version, but optimized for broadcasting.

        Parameters
        ----------
        current_price : np.ndarray
            The current price of the underlying asset (S_0).
        exercise_price : np.ndarray
            The strike (exercise) price of the option (K).
        time_to_maturity_days : np.ndarray
            Time until option maturity annualized (T).
        risk_free_rate : np.ndarray
            Annualized risk-free interest rate (r).
        initial_volatility : np.ndarray
            Some measure of the variance (volatility squared) of the underlying on the current day (V_0).
        longterm_avg_vol : np.ndarray
            The longterm mean of the variance (theta).
        vol_of_vol : np.ndarray
            The volatility of the variance process (eta).
        strength_of_mean_reversion : np.ndarray
            The rate of mean reversion of the variance (kappa). Higher kappa reverts the variance to the longterm variance theta faster.
        correlation : np.ndarray
            The correlation between the underlying asset price and its variance (rho). Usually negative, meaning when a stock price falls its volatility rises.
        lambda_val : np.ndarray
            A drift adjustment term for the variance process (rho). Often set to 0.
        batch_size : int
            The batch size to use so that memory stays managable during vectorization (mostly in numerical integration)
        verbose : bool
            Show batch progress

        Returns
        -------
        np.ndarray
            The prediced call option price (C).
        """

        n = len(current_price)
        result = np.empty(n)

        for batch_i in range(0, n, batch_size):
            batch_j = min(batch_i + batch_size, n)
            if verbose:
                print(f'On batch {batch_i//batch_size} of {n//batch_size}')


            # These could be remove but it's functional for now
            local_S_0 = current_price[batch_i : batch_j]
            local_K = exercise_price[batch_i : batch_j]
            local_T = time_to_maturity[batch_i : batch_j]
            local_r = risk_free_rate[batch_i : batch_j]
            local_V_0 = initial_volatility[batch_i : batch_j]
            local_theta = longterm_avg_vol[batch_i : batch_j]
            local_eta = vol_of_vol[batch_i : batch_j]
            local_kappa = strength_of_mean_reversion[batch_i : batch_j]
            local_rho = correlation[batch_i : batch_j]
            local_lambda_val = lambda_val[batch_i : batch_j]

            u_grid = np.linspace(1e-8, 150, 1000) # Grid for numerical integration. Can increase the stop and num (2nd and 3rd arguments) for higher resolution

            # Precompute b_1 and b_2
            b_1 = local_kappa + local_lambda_val + (1 - 2) * local_rho * local_eta
            b_2 = local_kappa + local_lambda_val + (2 - 2) * local_rho * local_eta


            # Reshape everything to broadcast
            S_0_col = local_S_0[:, None]
            K_col = local_K[:, None]
            T_col = local_T[:, None]
            r_col = local_r[:, None]
            V_0_col = local_V_0[:, None]
            theta_col = local_theta[:, None]
            eta_col = local_eta[:, None]
            kappa_col = local_kappa[:, None]
            rho_col = local_rho[:, None]
            lambda_val_col = local_lambda_val[:, None] # This isn't used

            u = u_grid[None, :]

            P_i = [np.nan, np.nan]


            # Calculate P_1 and P_2
            for i in [1, 2]:
                if i == 1:
                    b_col = b_1[:, None]
                elif i == 2:
                    b_col = b_2[:, None]

                d_i = np.sqrt((b_col - rho_col * eta_col * u * 1j) ** 2 + eta_col ** 2 * (u ** 2 + (-1) ** i * u * 1j))

                g_i = (b_col - rho_col * eta_col * u * 1j + d_i) / (b_col - rho_col * eta_col * u * 1j - d_i)

                C_i = r_col * u * T_col * 1j + kappa_col * theta_col / (eta_col ** 2) * ((b_col - rho_col * eta_col * u * 1j + d_i) * T_col - 2 * np.log((1 - g_i * np.exp(d_i * T_col)) / (1 - g_i)))
                
                D_i = ((b_col - rho_col * eta_col * u * 1j + d_i) / eta_col ** 2) * ((1 - np.exp(d_i * T_col)) / (1 - g_i * np.exp(d_i * T_col)))

                f_i = np.exp(1j * u * np.log(S_0_col) + C_i + D_i * V_0_col)
                
                integrand = ((np.exp(-1j * u * np.log(K_col)) * f_i) / (1j * u)).real

                P_i[i-1] = 0.5 + 1/np.pi * np.trapezoid(integrand, u_grid, axis=1) # Crucially change out the integrated value here to the trapezoid "integral" approximation


            P_1 = P_i[0][:, None]
            P_2 = P_i[1][:, None]

            call_prices = S_0_col * P_1 - np.exp(-r_col * T_col) * K_col * P_2


            result[batch_i : batch_j] = call_prices.reshape(-1)

        return result