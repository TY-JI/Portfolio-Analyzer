import numpy as np

# TODO consider implementing a constrained optimizer

class Optimizer:
    def __init__(self, expected_returns, covariance, risk_aversion):
        self.expected_returns = expected_returns
        self.covariance = covariance
        self.risk_aversion = risk_aversion

    def optimize(self):
        return np.linalg.solve(
            self.risk_aversion * self.covariance,
            self.expected_returns
        )