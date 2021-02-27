from ..utils import SMCSampler, ExperimentParameters

import autograd.numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


class Pretuning(SMCSampler):
    """Implement tuned HMC approach of studied paper."""

    def __init__(self, parameters: ExperimentParameters):
        super().__init__(parameters)
        self.epsilon_star = 0.1
        self.Lmax = 100

    def move_through_kernel(self, X, lambda_t):
        """Non-iterated version : single move for each x in X."""

        # Kernel builds itself before moving
        M, inv_M, epsilon, L = self.build_kernel(X, lambda_t)

        # Sample p
        p = stats.multivariate_normal(np.zeros(self.param.d),
                                      M).rvs(self.param.N)

        # Apply the leapfrog integration
        Z, p_new = self.leapfrog_integrator(X, p, lambda_t, L,
                                            epsilon, inv_M)
        log_proba = self.param.log_gamma_t(lambda_t, X) - \
            self.param.log_gamma_t(lambda_t, Z) + \
            (np.einsum('ij,ij->i', p_new, np.dot(p_new, inv_M)) -
             np.einsum('ij,ij->i', p, np.dot(p, inv_M))) / 2

        # Keep new moves or not
        log_acc = np.log(np.random.rand(len(X)))
        keep_it = log_acc < log_proba
        keep_it = keep_it[:, np.newaxis]

        X = Z * keep_it + X * (1 - keep_it)

        return X

    def build_kernel(self, X, lambda_t):

        N = self.param.N
        d = self.param.d

        # Matrix M
        diag_cov = np.diag(np.cov(X, rowvar=False))
        inv_M = diag_cov * np.eye(d)
        M = np.diag(np.nan_to_num(1 / diag_cov))
        self._inv_M = inv_M

        # Sample epsilon and L
        epsilon = stats.uniform(scale=self.epsilon_star).rvs(size=N)
        L = np.random.randint(1, self.Lmax + 1, size=N)

        # Sample p
        p = stats.multivariate_normal(np.zeros(d), M).rvs(N)

        # Apply the leapfrog integration
        Z, p_new = self.leapfrog_integrator(X, p, lambda_t, L,
                                            epsilon, inv_M)

        # Calculate variation of energy
        DeltaE = self.param.log_gamma_t(lambda_t, X) - \
            self.param.log_gamma_t(lambda_t, Z) + \
            (np.einsum('ij,ij->i', p_new, np.dot(p_new, inv_M)) -
             np.einsum('ij,ij->i', p, np.dot(p, inv_M))) / 2

        # Calculate ESJD
        ESJD_estimate = self._ESJD_estimator(X, Z, inv_M, DeltaE, L)
        ESJD_sample_prob = ESJD_estimate / ESJD_estimate.sum()

        # Calculate epsilon_star based on quantile regression
        data_reg = pd.DataFrame({'diff_energy': np.abs(DeltaE),
                                 'eps': epsilon**2})
        model = smf.quantreg('diff_energy ~ eps', data_reg)
        res = model.fit(q=.5)

        quotien = (np.abs(np.log(0.9)) - res.params['Intercept']) / \
            res.params['eps']
        if (quotien > 0):
            self.epsilon_star = np.sqrt(quotien)

        # Sample epsilon and L
        ESJD_sample = np.random.choice(N, N, replace=True, p=ESJD_sample_prob)
        new_epsilon = epsilon[ESJD_sample]
        new_L = L[ESJD_sample]

        return M, inv_M, new_epsilon, new_L

    def leapfrog_integrator(self, X_init, p_init, lambda_t, L,
                            epsilon, inv_M):
        '''Generic leapfrog intergrator with given parameters).
        X_init, p_init, L and espilon are arrays'''

        X = X_init.copy()
        p = p_init.copy()

        L_copy = L.copy()

        for iter in range(np.max(L.astype(int))):
            to_update = (np.maximum(0, L_copy) > 0) * 1
            p += 0.5 * np.einsum(
                'i,ij->ij', to_update,
                np.einsum('i,ij->ij', epsilon,
                          self.param.grad_log_gamma_t(lambda_t, X)))
            X += np.einsum(
                'i,ij->ij', to_update,
                np.einsum('i,ij->ij', epsilon,
                          np.dot(p, inv_M)))
            p += 0.5 * np.einsum(
                'i,ij->ij', to_update,
                np.einsum('i,ij->ij', epsilon,
                          self.param.grad_log_gamma_t(lambda_t, X)))
            L_copy -= 1
        return X, p

    def _ESJD_estimator(self, x, y, inv_M, DeltaE, L):
        diff_x_y = x - y
        dist_x_y = np.diag(diff_x_y @ inv_M @ diff_x_y.T)
        return dist_x_y * np.minimum(1, np.exp(DeltaE)) / L

    def __str__(self):
        if self.X is not None:
            return f"Pretuning sampler containing {self.param.N} particles " \
                f"in dimension {self.param.d}."
        else:
            "Pretuning sampler to be runned."
