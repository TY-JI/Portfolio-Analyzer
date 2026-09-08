class Covariance:

    def calculate_sample_cov(self, returns):
        return returns.cov()
