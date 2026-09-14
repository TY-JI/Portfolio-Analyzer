import numpy as np

# TODO look into tau

class BlackLitterman:

    def __init__(self, covariance, pi, tau=0.025, P=None, q=None, omega=None):
        self.covariance = np.asarray(covariance, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        self.tau = tau
        self.P = None if P is None else np.asarray(P, dtype=float)
        self.q = None if q is None else np.asarray(q, dtype=float)
        self.omega = None if omega is None else np.asarray(omega, dtype=float)
        # Default values set to none in case no views are given

    # Using Woodbury Matrix Identity
    def expected_returns(self):

        if self.P is None:
            return self.pi

        self.validate_inputs()
        self.validate_views()

        tau_covariance = self.tau * self.covariance

        view_covariance = self.P @ (self.tau * self.covariance) @ self.P.T + self.omega

        mean_adjustment = (
            tau_covariance
            @ self.P.T
            @ np.linalg.solve(
                view_covariance,
                self.q - self.P @ self.pi
            )
        )

        expected_returns = self.pi + mean_adjustment

        return expected_returns

    def validate_views(self):
        P = self.P
        q = self.q
        omega = self.omega

        # No views is a valid configuration
        if P is None and q is None and omega is None:
            return

        # Either all views are provided or none are
        if P is None or q is None or omega is None:
            raise ValueError(
                "P, q, and omega must either all be provided or all be None."
            )

        if P.ndim != 2:
            raise ValueError(
                "P must be a two-dimensional matrix."
            )

        n_assets = len(self.pi)

        if P.shape[1] != n_assets:
            raise ValueError(
                "P must have one column for each asset."
            )

        if q.ndim != 1:
            raise ValueError(
                "q must be a one-dimensional vector."
            )

        if len(q) != P.shape[0]:
            raise ValueError(
                "q must contain one value for each view."
            )

        if omega.ndim != 2:
            raise ValueError(
                "Omega must be a two-dimensional matrix."
            )

        if omega.shape != (P.shape[0], P.shape[0]):
            raise ValueError(
                "Omega must be square with one row and column for each view."
            )

        if not np.all(np.isfinite(P)):
            raise ValueError(
                "P contains NaN or infinite values."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "q contains NaN or infinite values."
            )

        if not np.all(np.isfinite(omega)):
            raise ValueError(
                "Omega contains NaN or infinite values."
            )

        if not np.allclose(omega, omega.T):
            raise ValueError(
                "Omega must be symmetric."
            )

        try:
            np.linalg.cholesky(omega)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Omega must be positive definite."
            )
        
    def validate_inputs(self):
        covariance = self.covariance
        pi = self.pi
        tau = self.tau

        if covariance.ndim != 2:
            raise ValueError(
                "Covariance matrix must be two-dimensional."
            )

        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError(
                "Covariance matrix must be square."
            )

        if pi.ndim != 1:
            raise ValueError(
                "Pi must be a one-dimensional vector."
            )

        if covariance.shape[0] != len(pi):
            raise ValueError(
                "Covariance matrix and pi must contain "
                "the same number of assets."
            )

        if not np.all(np.isfinite(covariance)):
            raise ValueError(
                "Covariance matrix contains NaN or infinite values."
            )

        if not np.all(np.isfinite(pi)):
            raise ValueError(
                "Pi contains NaN or infinite values."
            )

        if not np.allclose(covariance, covariance.T):
            raise ValueError(
                "Covariance matrix must be symmetric."
            )

        if tau <= 0:
            raise ValueError(
                "Tau must be positive."
            )

        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Covariance matrix must be positive definite."
            )