from SMC.utils import SMCSampler, ExperimentParameters

import autograd.numpy as np
from scipy import stats


class TunedISMC(SMCSampler):
    """Implement N. Chopin (2001) approach to construct a tuned Independence
    Sampler Kernel."""

    def __init__(self, parameters: ExperimentParameters):
        super().__init__(parameters)

    def move_through_kernel(self, X, lambda_t):
        """Single move for each x in X using Independence Sampler properly
        calibrated."""

        # First calculate mean and covariance
        E = np.mean(X, axis=0)
        assert E.shape == (self.param.d,), "Wrong shape for particles mean."
        cov = np.cov(X, rowvar=False, bias=True)
        V = np.diag(np.diag(cov))
        inv_V = np.diag(1/np.diag(cov))
        assert V.shape == (self.param.d, self.param.d), \
            "Wrong shape for particles variance-covariance matrix."

        # Simulate new observations
        g = stats.multivariate_normal(E, V)
        Z = g.rvs(self.param.N)

        # Compute probabilities
        log_pdf_x = -0.5 * np.einsum('ij,ij->i', X - E,
                                     np.dot(inv_V, (X - E).T).T)
        log_pdf_z = -0.5 * np.einsum('ij,ij->i', Z - E,
                                     np.dot(inv_V, (Z - E).T).T)

        log_proba = log_pdf_x - log_pdf_z
        log_proba += self.param.log_gamma_t(lambda_t, Z) \
            - self.param.log_gamma_t(lambda_t, X)

        # Keep new moves or not
        log_acc = np.log(np.random.rand(len(X)))
        keep_it = log_acc < log_proba
        keep_it = keep_it[:, np.newaxis]

        return Z * keep_it + X * (1 - keep_it)

    def __str__(self):
        if self.X is not None:
            return f"Tuned ISMC sampler containing {self.param.N} particles " \
                f"in dimension {self.param.d}."
        else:
            "Tuned ISMC sampler to be runned."
