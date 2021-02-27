from ..utils import SMCSampler, ExperimentParameters

import autograd.numpy as np
from scipy import stats


class HMSC(SMCSampler):
    """Implement basic SMC with non tuned SMC kernel."""

    def __init__(self, parameters: ExperimentParameters, epsilon=0.05, L=5):
        super().__init__(parameters)
        self.epsilon = epsilon
        self.L = L

    def _leapfrog_integrator(self, X_init, p_init, lambda_t):

        _X = X_init.copy()
        _p = p_init.copy()

        for iter in range(self.L):

            _p += (self.epsilon / 2) * self.param.grad_log_gamma_t(lambda_t,
                                                                   _X)
            _X += self.epsilon * _p
            _p += (self.epsilon / 2) * self.param.grad_log_gamma_t(lambda_t,
                                                                   _X)

        return _X, _p

    def move_through_kernel(self, X, lambda_t):
        """Non-iterated version : single move for each x in X."""

        N = self.param.N
        d = self.param.d

        p = stats.multivariate_normal(np.zeros(d), np.eye(self.param.d)).rvs(N)
        Z, p_new = self._leapfrog_integrator(X, p, lambda_t)

        log_proba = self.param.log_gamma_t(lambda_t, X) - \
            self.param.log_gamma_t(lambda_t, Z)
        log_proba += (np.einsum('ij,ij->i', p_new, p_new) -
                      np.einsum('ij,ij->i', p, p)) / 2

        # Keep new moves or not
        log_acc = np.log(np.random.rand(len(X)))
        keep_it = log_acc < log_proba
        keep_it = keep_it[:, np.newaxis]

        return Z * keep_it + X * (1 - keep_it)

    def __str__(self):
        if self.X is not None:
            return f"HSMC sampler containing {self.param.N} particles " \
                f"in dimension {self.param.d}."
        else:
            "HSMC sampler to be runned."
