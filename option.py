import numpy as np
import scipy as sp

class Option:
    """
    Class used to price options and calculate basic greeks based on Black-Scholes equation.
    """

    def __init__(self,  K, sigma, r=0):
        '''
        Initialize the option class with S and K and sigma and T
        This class assumes the option to be based on a non dividend-paying underlying stock
        and therefore in the Black-Scholes equation, b = r.
        :param K: Strike price of option
        :param sigma: Annual volatility of underlying stock
        :param r: Interest rate
        '''
        self.K = K
        self.sigma = sigma
        self.r = r

    def _d1_d2(self, S, T):
        """
        Helper function calculation d_1 and d_2 as used in Black-Scholes equation
        :param S: Price of underlying stock
        :param T: Time to maturity of option
        :return: d_1 and d_2 as a float
        """
        d_1 = (np.log(S / self.K) + (self.r + (self.sigma ** 2 / 2)) * T) / (self.sigma * np.sqrt(T))
        d_2 = d_1 - self.sigma * np.sqrt(T)

        return d_1, d_2

    def N(self, x):
        """
        Standard normal distribution cdf
        :param x: point of evaluation
        :return: The standard normal distribution cdf at x
        """
        return sp.special.ndtr(x)

    def n(self, x):
        """
        Standard normal distribution pdf
        :param x: point of evaluation
        :return: The standard normal distribution pdf at x
        """
        return np.exp((-x**2)/2) / np.sqrt(2*np.pi)


    def black_scholes_eu_call(self, S, T):
        """
        Calculate the value of a european call option using the solution to the
        Black-Scholes equation, and the given inputs in the Option class
        :return: value of a single european call option as a float
        :param: S: Price of underlying stock
        :param: T: Time to maturity of option
        """
        S = np.asarray(S, dtype=float)
        T = np.asarray(T, dtype=float)

        C = np.zeros_like(S, dtype=float)

        #checking that option is not at expiry date, to avoid division by zero
        alive = T > 1e-8

        if np.any(alive):
            d1, d2 = self._d1_d2(S[alive], T[alive])

            C[alive] = (
                    S[alive] * self.N(d1)
                    - self.K * np.exp(-self.r * T[alive]) * self.N(d2)
            )

        # Expired options
        C[~alive] = np.maximum(S[~alive] - self.K, 0.0)

        return C

    def get_delta(self, S, T):
        """
        Calculate the delta based on the Black-Scholes equation
        :param S: Price of underlying stock
        :param T: Time to maturity of option
        :return: Delta as a float
        """
        #Edge case when very close to expiry
        S = np.asarray(S, dtype=float)
        T = np.asarray(T, dtype=float)

        delta = np.zeros_like(S)

        alive = T > 1e-8
        d1, _ = self._d1_d2(S[alive], T[alive])
        delta[alive] = self.N(d1)

        delta[~alive] = (S[~alive] > self.K).astype(float)

        return delta

    def get_gamma(self, S, T):
        """
        Calculate the gamma based on the Black-Scholes equation
        :param S: Price of underlying stock
        :param T: Time to maturity of option
        :return: Gamma as a float
        """
        #Edge case when very close to expiry
        S = np.asarray(S, dtype=float)
        T = np.asarray(T, dtype=float)

        gamma = np.zeros_like(S)

        alive = T > 1e-8
        d1, _ = self._d1_d2(S[alive], T[alive])
        gamma[alive] = self.n(d1) / (S[alive] * self.sigma * np.sqrt(T[alive]))

        return gamma

    def get_theta(self, S, T):
        """
        Calculate the theta based on the Black-Scholes equation
        :param S: Price of underlying stock
        :param T: Time to maturity of option
        :return: Theta as a float
        """
        S = np.asarray(S, dtype=float)
        T = np.asarray(T, dtype=float)

        theta = np.zeros_like(S, dtype=float)

        alive = T > 1e-8
        if np.any(alive):
            d1, d2 = self._d1_d2(S[alive], T[alive])

            theta[alive] = (
                    -(S[alive] * self.n(d1) * self.sigma) / (2.0 * np.sqrt(T[alive]))
                    - self.r * self.K * np.exp(-self.r * T[alive]) * self.N(d2)
            )

        # Set theta at expiry to zero
        theta[~alive] = 0.0

        return theta


