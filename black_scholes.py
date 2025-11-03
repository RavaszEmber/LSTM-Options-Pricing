# Implement Black-Scholes model from section 4.3.1 in paper https://link.springer.com/article/10.1007/s10203-025-00518-9


# Description from paper
##################################################################

# Assuming the model
# dS_t = r*S_t*dt + sigma * S_t * dW_t
# where S_t is underlying asset price at time t
# sigma is volatility of the underlying asset
# r is the risk-free interest rate
# W is a Wiener process



# Under this model the price for a European call option C for a non-dividend paying asset is given as

# C(S_0, T) = S_0 * N(d_1) - K * exp(-r * T) * N(d_2)

# Where d_1 = (ln(S_0/K) + (r + sigma**2 / 2) * T) / (sigma * T ** 0.5)
# and d_2 = d_1 - sigma * T ** 0.5

# where S_0 is the current price of the underlying,
# K is the exercise price of the option,
# T is the time to maturity (in years)
# N(.) is the cumulative normal distribution

# Note for the LSTM the volatility is either the 90 day moving average volatility of the underlying asset OR the GG

##################################################################

import math
import scipy.stats

class BlackScholesModel:
    """
    Contains methods to calculate the predicted call price via the Black Scholes model
    """

    def __init__(self):
        pass

    def predicted_call_price(self, current_price : float, exercise_price : float, time_to_maturity : float, volatility : float, risk_free_rate : float) -> float:
        """
        Calculate European call option price using Black-Scholes model.

        Parameters
        ----------
        current_price : float
            The current price of the underlying asset (S_0).
        exercise_price : float
            The strike (exercise) price of the option (K).
        time_to_maturity_days : float
            Time until option maturity annualized (T)
        volatility : float
            Some measure of the annualized volatility of the underlying asset (sigma).
        risk_free_rate : float
            Annualized risk-free interest rate (r)

        Returns
        -------
        float
            The prediced call option price (C).
        """

        # d_1 = (ln(S_0/K) + (r + sigma**2 / 2) * T) * (sigma * T ** 0.5) NOTE: This original formula from the paper had a typo, the term with the risk free rate should multiply by T, not divide 
        d_1 = (math.log(current_price / exercise_price) + (risk_free_rate + volatility**2 / 2) * time_to_maturity) / (volatility * time_to_maturity ** 0.5)

        # d_2 = d_1 - sigma * T ** 0.5
        d_2 = d_1 - volatility * time_to_maturity ** 0.5

        # C(S_0, T) = S_0 * N(d_1) - K * exp(-r * T) * N(d_2)
        predicted_call_price = current_price * scipy.stats.norm.cdf(d_1) - exercise_price * math.exp(-risk_free_rate * time_to_maturity) * scipy.stats.norm.cdf(d_2)

        return predicted_call_price
