import numpy as np

class Covariance:

    def ledoit_wolf(self, returns):
        T, N = returns.shape
        demeaned_returns = self.demean(returns)
        S = self.sample_covariance(demeaned_returns, T)
        F, r_bar = self.target_matrix(S, N)

        pi_hat = self.pi_hat(demeaned_returns,S,T)
        rho_hat = self.rho_hat(demeaned_returns,S,T,r_bar)
        gamma_hat = self.gamma_hat(S,F)
        #print(f'pi hat: {pi_hat}')
        #print(f'rho hat: {rho_hat}')
        #print(f'gamma hat: {gamma_hat}')

        k = (pi_hat-rho_hat)/gamma_hat
        #print(f'k: {k}')
        delta = max(0,min(1,k/T))
        #print(f'delta: {delta}')

        ledoit_wolf = (delta*F) + (1-delta)*S
        return ledoit_wolf



    def pi_hat_matrix(self, demeaned_returns, S, T):
        squared_deviation = self.covariance_deviations(S,demeaned_returns) ** 2
        pi_hat_matrix = squared_deviation.sum(axis=0) / T
        return pi_hat_matrix

    def pi_hat(self, demeaned_returns, S, T): 
        return self.pi_hat_matrix(
            demeaned_returns,
            S,
            T
        ).sum()



    def rho_hat(self, demeaned_returns, S, T, r_bar):
        rho_diag = np.diag(self.pi_hat_matrix(demeaned_returns, S, T)).sum()

        v_ii_ij = np.einsum(
            'ti,tij->ij',
            demeaned_returns.to_numpy()**2 - np.diag(S),
            self.covariance_deviations(S,demeaned_returns)
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



    def gamma_hat(self, S, F):
        return np.sum((np.array(S)-np.array(F))**2)


    
    def target_matrix(self, S, N):
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



    def demean(self, returns):
        return returns - returns.mean()
    
    def sample_covariance(self, demeaned_returns, T):
        return (demeaned_returns.T @ demeaned_returns) / T

    def period_covariances(self, demeaned_returns):
        covariances = np.einsum(
            'ti,tj->tij',
            demeaned_returns,
            demeaned_returns
        )
        return covariances

    def covariance_deviations(self,S,demeaned_returns):
        deviation = self.period_covariances(demeaned_returns) - S.to_numpy()
        return deviation



