from ..utils import SMCSampler, ExperimentParameters

import autograd.numpy as np
from scipy import stats


class FearnheadTaylor(SMCSampler):
    """Implement tuned HMC approach of studied paper."""

    def __init__(self, parameters: ExperimentParameters):
        super().__init__(parameters)
        self.hist_X = []
        self.hist_Z = []
        self.hist_DeltaE = []
        self.hist_epsilon = [stats.uniform(scale=0.1).rvs(size=self.param.N)]
        self.hist_L = [np.random.randint(1, 101, size=self.param.N)]

    def move_through_kernel(self, X, lambda_t):
        """Non-iterated version : single move for each x in X."""

        # Kernel builds itself before moving
        M, inv_M, epsilon, L = self.build_kernel(X)

        p = stats.multivariate_normal(np.zeros(self.param.d),
                                      M).rvs(self.param.N)
        # Z = np.empty((self.param.N, self.param.d))

        # Compute new particles and log of probability of keeping them
        Z, p_new_bis = self.leapfrog_integrator(X, p, lambda_t, L,
                                                epsilon, inv_M)
        log_proba = self.param.log_gamma_t(lambda_t, X) - \
            self.param.log_gamma_t(lambda_t, Z) + \
            (np.einsum('ij,ij->i', p_new_bis,
                       np.dot(p_new_bis, inv_M)) -
             np.einsum('ij,ij->i', p,
                       np.dot(p, inv_M))) / 2

        # Keep new moves or not
        log_acc = np.log(np.random.rand(len(X)))
        keep_it = log_acc < log_proba
        keep_it = keep_it[:, np.newaxis]
        X = Z * keep_it + X * (1 - keep_it)

        # Saving values of Z and DeltaE
        self.hist_Z.append(Z)
        self.hist_DeltaE.append(log_proba)

        return X

    def _ESJD_estimator(self, x, y, inv_M, DeltaE, L):
        diff_x_y = x - y
        dist_x_y = np.diag(diff_x_y @ inv_M @ diff_x_y.T)
        return dist_x_y * np.minimum(1, np.exp(DeltaE)) / L

    def build_kernel(self, X):

        # Matrix M
        diag_cov = np.diag(np.cov(X, rowvar=False))
        inv_M = np.diag(diag_cov)
        M = np.diag(np.nan_to_num(1 / diag_cov))

        # Handle first move case:
        if len(self.hist_X) == 0:

            self._inv_M = inv_M
            new_epsilon = self.hist_epsilon[-1]
            new_L = self.hist_L[-1]

            self.hist_X.append(X.copy())

            return M, inv_M, new_epsilon, new_L

        # Compute ESJD estimator
        ESJD_estimate = self._ESJD_estimator(self.hist_X[-1],
                                             self.hist_Z[-1],
                                             self._inv_M,
                                             self.hist_DeltaE[-1],
                                             self.hist_L[-1])
        ESJD_sample_prob = ESJD_estimate / ESJD_estimate.sum()

        # update inv_M
        self._inv_M = inv_M

        # Sample according to good values of ESJD
        ESJD_sample = np.random.choice(self.param.N, self.param.N,
                                       replace=True, p=ESJD_sample_prob)

        new_epsilon = np.ones(self.param.N)
        new_L = np.ones(self.param.N)

        # Perturbation kernel
        new_epsilon = np.abs(
            stats.multivariate_normal(self.hist_epsilon[-1][ESJD_sample],
                                      0.015 * np.eye(self.param.N)).rvs()
            )
        new_epsilon[new_epsilon > 0.7] = 0.7

        chosen_L = self.hist_L[-1][ESJD_sample]
        new_L = chosen_L + np.random.choice(np.array([- 1, 0, 1]),
                                            self.param.N)
        new_L[new_L < 1] = 1

        # Saving values of parameters and X
        self.hist_epsilon.append(new_epsilon.copy())
        self.hist_L.append(new_L.copy())
        self.hist_X.append(X.copy())

        return M, inv_M, new_epsilon, new_L

    def leapfrog_integrator(self, X_init, p_init, lambda_t, L,
                            epsilon, inv_M):
        '''Generic leapfrog intergrator with given parameters).
        X_init, p_init, L and espilon are arrays'''

        X = X_init.copy()
        p = p_init.copy()
        assert np.all(np.abs(inv_M - inv_M.T) < 1e-5), "inv_M not symmetric !"

        L_copy = L.copy()

        for iter in range(np.max(L.astype(int))):
            to_update = (np.maximum(0, L_copy) > 0) * 1
            p += 0.5 * np.einsum(
                'i,ij->ij', to_update,
                np.einsum('i,ij->ij',
                          epsilon,
                          self.param.grad_log_gamma_t(lambda_t, X)))
            X += np.einsum(
                'i,ij->ij', to_update,
                np.einsum('i,ij->ij',
                          epsilon,
                          np.dot(p, inv_M)))
            p += 0.5 * np.einsum(
                'i,ij->ij', to_update,
                np.einsum('i,ij->ij',
                          epsilon,
                          self.param.grad_log_gamma_t(lambda_t, X)))
            L_copy -= 1
        return X, p

    def __str__(self):
        if self.X is not None:
            return f"Fearnhead Taylor sampler containing {self.param.N} " \
                f"particles in dimension {self.param.d}."
        else:
            "Fearnhead Taylor sampler to be runned."
