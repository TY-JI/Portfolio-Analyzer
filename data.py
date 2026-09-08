import yfinance as yf

# TODO
# add proper error handling for detected correctly formatted date inputs
# add option to select desired time frame

class MarketDataLoader:

    def __init__(self, tickers):
        self.tickers = tickers

    def download(self, start, end):
        data = yf.download(
            self.tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )

        return data
        
    def get_close_prices(self, data):
        return data['Close']