from data import MarketDataLoader
from validation import DataValidator

def main():
    tickers = [
    'AAPL',
    'MSFT',
    'GOOGL',
    'JPM',
    'XOM'
    ]

    loader = MarketDataLoader(tickers)

    data = loader.download(
        start='2015-01-01',
        end='2026-01-01'
        )

    close_prices = loader.get_close_prices(data)

    validator = DataValidator()

    validator.validate(close_prices)

if __name__=='__main__':
    main()