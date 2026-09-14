from data import MarketDataLoader
from validation import DataValidator
from returns import Returns
from covariance import Covariance
from blackLitterman import BlackLitterman
from equilibriumReturns import EquilibriumReturns
from optimizer import Optimizer

def main():
    # Loading market data
    loader = MarketDataLoader()
    data = loader.download()
    # Getting close prices
    close_prices = loader.get_close_prices(data)

    # Validating data 
    # Checks for missing values, negatives and duplicates
    validator = DataValidator()
    validator.validate(close_prices)

    # Calculating % returns + removal of first row containing missing data
    returns_calculator = Returns()
    returns = returns_calculator.remove_missing_returns(
        returns_calculator.calculate_returns(close_prices)
    )

    covariance = Covariance()
    ledoit_wolf = covariance.ledoit_wolf(returns)
    sample_cov_matrix = covariance.sample_covariance(returns,returns.shape[0]-1)
    #print(f'ledoit wolf: {ledoit_wolf}')
    #print(f'sample covariance matrix: {sample_cov_matrix}')

    eq = EquilibriumReturns(covariance=ledoit_wolf)

    market_caps = loader.market_caps()
    market_weights = eq.market_weights(market_caps)

    bl = BlackLitterman(
        covariance=ledoit_wolf,
        pi=eq.pi(market_caps)
    )
    expected_returns = bl.expected_returns()
    risk_aversion = eq.risk_aversion(market_weights)

    opt = Optimizer(expected_returns,ledoit_wolf,risk_aversion)

    print(f'Market weights: {market_weights}')
    print(f'Optimized weights: {opt.optimize()}')

if __name__=='__main__':
    main()