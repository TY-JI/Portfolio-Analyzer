import numpy as np

class Covariance:

    def ledoit_wolf(self, returns):
        T, N = returns.shape
        demeaned_returns = self.demean(returns)
        n = T - 1

        S = self.calculate_sample_covariance(demeaned_returns, n)
        F, r_bar = self.calculate_target_matrix(S, N)
        pi_hat = self.calculate_pi_hat(
            demeaned_returns,
            S,
            T
        )

        print(self.calculate_pi_hat(demeaned_returns,S,T)-self.calculate_rho(demeaned_returns, S, T, r_bar))


    def calculate_pi_hat_matrix(self, demeaned_returns, S, T):
        deviation = self.calculate_covariance_deviations(S,demeaned_returns)

        squared_deviation = deviation ** 2
        pi_hat_matrix = squared_deviation.sum(axis=0) / T
        return pi_hat_matrix

    def calculate_pi_hat(self, demeaned_returns, S, T): 
        pi_hat = self.calculate_pi_hat_matrix(demeaned_returns, S,T).sum()
        return pi_hat

    def calculate_rho(self, demeaned_returns, S, T, r_bar):
        rho_diag = np.diag(self.calculate_pi_hat_matrix(demeaned_returns, S, T)).sum()

        v_ii_ij = np.einsum(
            'ti,tij->ij',
            demeaned_returns.to_numpy()**2 - np.diag(S),
            self.calculate_covariance_deviations(S,demeaned_returns)
        )/T
        
        v_jj_ij = v_ii_ij.T

        standard_deviations = np.sqrt(np.diag(S))
        standard_deviation_ratio = np.outer(
            standard_deviations,
            1/standard_deviations
        )

        theta = (r_bar/2) * ((1/standard_deviation_ratio) * v_ii_ij + standard_deviation_ratio * v_jj_ij)

        np.fill_diagonal(theta,0)
        rho_off_diag = theta.sum()

        return rho_diag + rho_off_diag

    def calculate_sample_covariance(self, demeaned_returns, n):
        return (demeaned_returns.T @ demeaned_returns) / n

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

        r_bar = (np.sum(correlation_matrix.to_numpy()) - N) / (N*(N - 1))
        # - N eliminates 1s along the diagonal and since there are N - 1 other asses for every asset we have N(N - 1) in the denominator
        
        F = r_bar * standard_deviations_outer
        np.fill_diagonal(F, variances)

        return F, r_bar

    def calculate_covariance_contributions(self, demeaned_returns):
        covariance_contributions = np.einsum(
            'ti,tj->tij',
            demeaned_returns,
            demeaned_returns
        )
        return covariance_contributions

    def calculate_covariance_deviations(self,S,demeaned_returns):
        deviation = self.calculate_covariance_contributions(demeaned_returns) - S.to_numpy()
        return deviation

    def demean(self, returns):
        return returns - returns.mean()



