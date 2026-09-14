import numpy as np

class EquilibriumReturns:
    def __init__(self, covariance):
        self.covariance = covariance

    def pi(self,market_caps, market_return=0.08, risk_free_rate=0.04):
        market_weights = self.market_weights(market_caps)

        return (
            self.risk_aversion(market_weights, market_return, risk_free_rate) 
            * self.covariance 
            @ self.market_weights(market_caps)
        )

    def market_weights(self, market_caps):
        market_caps = np.asarray(market_caps,dtype=float)

        return market_caps / market_caps.sum()

    def risk_aversion(self, market_weights, market_return=0.08, risk_free_rate=0.04):
        market_varience = market_weights.T @ self.covariance @ market_weights

        excess_return = market_return - risk_free_rate

        return excess_return / market_varience