class Returns:

    # TODO
    # perhaps include some checks to confirm dropna 
    # isn't killing missing data other than the first row

    def calculate_returns(self, prices):
        return prices.pct_change()
        # Calculates the percentage change between consecutive rows within each column

    def remove_missing_returns(self, returns):
        return returns.dropna()