import numpy as np


class BrownianMotion:
    """Define the Brownian Motion class"""

    def __init__(self, x_0=0, t=1.0, delta_t=1/256, mu=0.0, sigma=1.0):
        """
        Initialize the Brownian Motion parameters
        :param x_0: Start value of the Brownian Motion
        :param t: Time
        :param delta_t: timestep size
        :param mu: Drift of brownian motion per one unit of time
        :param sigma: Volatility parameter (standard deviation per sqrt(time))
        """
        self.x_0 = x_0
        self.t = t
        self.delta_t = delta_t
        self.mu = mu
        self.sigma = sigma
        self.time_steps = int(t/delta_t)

    def geometric_brownian_motion(self, seed):
        """
        Creating a geometric brownian motion using the variables given to the class
        :return: np.array of size t/delta_t with function values at each time step
        """
        total_steps = int(self.t / self.delta_t)

        rng = np.random.default_rng(seed)
        Z = rng.normal(size=total_steps - 1)

        log_increments = (self.mu - 0.5 * self.sigma**2) * self.delta_t + self.sigma * np.sqrt(self.delta_t) * Z

        log_x = np.empty(total_steps)
        log_x[0] = np.log(self.x_0)
        log_x[1:] = log_x[0] + np.cumsum(log_increments)

        x = np.exp(log_x)

        return np.array(x)