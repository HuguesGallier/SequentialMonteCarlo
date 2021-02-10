from SMC.utils import SMCSampler, ExperimentParameters

import autograd.numpy as np
from scipy import stats


class MALASMC(SMCSampler):
    """Implement basic SMC with MALA Kernel."""

    def __init__(self, parameters: ExperimentParameters, sigma=0.2):
        super().__init__(parameters)
        self.sigma = sigma

    def move_through_kernel(self, X, lambda_t):
        """Non-iterated version : single move for each x in X."""

        N = self.param.N
        d = self.param.d
        self.sigma = d ** (-1/3)

        eps = stats.multivariate_normal(np.zeros(d), np.eye(d)).rvs(N)

        mean_old = X + 0.5 * self.sigma * self.param.grad_log_gamma_t(lambda_t,
                                                                      X)
        Z = mean_old + self.sigma * eps
        mean_new = Z + 0.5 * self.sigma * self.param.grad_log_gamma_t(lambda_t,
                                                                      Z)

        # performs np.dot(Z[i] - mean_old, Z[i] - mean_old) for all i
        q_x_z = - 0.5 * np.einsum('ij,ij->i', Z - mean_old, Z - mean_old) / \
            (self.sigma ** 2)
        q_z_x = - 0.5 * np.einsum('ij,ij->i', X - mean_new, X - mean_new) / \
            (self.sigma ** 2)
        prob = self.param.log_gamma_t(lambda_t, Z) + q_z_x \
            - (q_x_z + self.param.log_gamma_t(lambda_t, X))

        # Keep new moves or not
        log_acc = np.log(np.random.rand(len(X)))
        keep_it = log_acc < prob
        keep_it = keep_it[:, np.newaxis]

        return Z * keep_it + X * (1 - keep_it)

    def __str__(self):
        if self.X is not None:
            return f"MALA sampler containing {self.param.N} particles in" \
                f"dimension {self.param.d}."
        else:
            "MALA sampler to be runned."
