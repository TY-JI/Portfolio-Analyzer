class DataValidator:

    def validate(self,data):
        self.check_duplicate_indexes(data)
        self.check_duplicate_tickers(data)
        self.check_missing_values(data)
        self.check_non_positive_prices(data)

    def check_missing_values(self, data):
        missing = data.isna().sum()
        # .isna will check for missing values
        # .sum will collapse all columns summing values of 1 where missing values were

        if missing.any():
            raise ValueError(
                f'Missing Values Detected:\n{missing[missing > 0]}'
            )

    def check_duplicate_indexes(self, data):
        if data.index.duplicated().any():
            duplicates = data.index[data.index.duplicated()]
            # Returns list of dates (indexes) at positions where data.index.duplicated() is True
            raise ValueError(
                f'Duplicate Index Values detected:\n{duplicates}'
            )

    def check_non_positive_prices(self, data):
        if (data <= 0).any().any():
            invalid_prices = data[data <= 0].stack()
            # Keeps all values that satisfy the (data <= 0) condition in the data dataframe
            # collapses columns vertically with .stack()
            raise ValueError(
                f'Detection of Invalid Prices:\n{invalid_prices}'
            )

    def check_duplicate_tickers(self, data):
        duplicates = data.columns[data.columns.duplicated()]
        # Similar logic to the duplicate check in check_duplicate_indexes

        if len(duplicates) > 0:
            raise ValueError(
                f'Duplicate Tickers Detected:\n{duplicates}'
            )