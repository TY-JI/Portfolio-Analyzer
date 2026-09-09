from data import MarketDataLoader
from validation import DataValidator
from returns import Returns
from covariance import Covariance

def main():
    tickers = [
    'AAPL',
    'MSFT',
    'GOOGL',
    'JPM',
    'XOM'
    ]

    # Loading market data
    loader = MarketDataLoader(tickers)
    data = loader.download(
        start='2015-01-01',
        end='2026-01-01'
        )
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

    # Calculating sample covariance matrix
    covariance = Covariance()

    # Testing ledoit wolf function
    covariance.ledoit_wolf(returns)


if __name__=='__main__':
    main()