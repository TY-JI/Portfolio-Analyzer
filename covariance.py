import numpy as np

class Covariance:

    def ledoit_wolf(self, returns):
        T, N = returns.shape
        demeaned_returns = self.demean(returns)
        n = T - 1

        S = self.calculate_sample_covariance(demeaned_returns, n)
        F = self.calculate_target_matrix(S, N)
        pi_hat = self.calculate_pi_hat(demeaned_returns,S, T)

        print("Pi hat:", pi_hat)
        print(F)

    def calculate_pi_hat(self, demeaned_returns, S, T):
        covariance_contributions = (self.calculate_covariance_contributions(demeaned_returns))
        deviation = self.calculate_covariance_deviations(S,covariance_contributions)

        squared_deviation = deviation ** 2
        pi_hat_matrix = squared_deviation.sum(axis=0) / T
        pi_hat = pi_hat_matrix.sum()

        return pi_hat

    def demean(self, returns):
        return returns - returns.mean()

    def calculate_sample_covariance(self, demeaned_returns, n):
        return (demeaned_returns.T @ demeaned_returns) / n

    def calculate_covariance_contributions(self, demeaned_returns):
        covariance_contributions = np.einsum(
            'ti,tj->tij',
            demeaned_returns,
            demeaned_returns
        )
        return covariance_contributions

    def calculate_covariance_deviations(self,S,covariance_contributions):
        deviation = covariance_contributions - S.to_numpy()
        return deviation

    def calculate_target_matrix(self, S, N):
        variances = np.diag(S)
        # Extract sample variances of all securities from the diagonal of sample cov matrix

        standard_deviations = np.sqrt(variances)
        # Calculate standard deviations of securities

        standard_deviations_outer = np.outer(
            standard_deviations,
            standard_deviations
        )
        # Constructing matrix of products of standard deviations sigma_i * sigma_j

        correlation_matrix = S / standard_deviations_outer
        # Recovering correlation matrix

        r_bar = (np.sum(correlation_matrix) - N) / (N*(N - 1))
        # - N eliminates 1s along the diagonal and since there are N - 1 other asses for every asset we have N(N - 1) in the denominator
        
        F = r_bar * standard_deviations_outer
        np.fill_diagonal(F, variances)

        return F






