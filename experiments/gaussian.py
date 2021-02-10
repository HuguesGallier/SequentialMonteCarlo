from ..utils import ExperimentParameters

import autograd.numpy as np
from scipy import stats


def gaussian_experiment(dimension, particles=1000, alpha=0.5):
    """ Wrapper for the Gaussian-Gaussian experiment described
    in the studied paper. """

    N = particles
    d = dimension
    Id = np.eye(d)
    correlation = 0.7*np.ones((d, d))
    np.fill_diagonal(correlation, 1)
    diag_var = np.linspace(start=0.1, stop=10, num=d)
    theta_d = np.dot(np.diag(diag_var**0.5), correlation)\
        .dot(np.diag(diag_var ** 0.5))
    inv_theta = np.linalg.inv(theta_d)

    def _log_prior(X):
        return - 0.5 * np.einsum('ij,ij->i', X, X) -\
            np.log((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(Id)))

    def _log_likelihood(X):
        return 0.5 * np.einsum('ij,ij->i', X, X) - \
            0.5 * np.einsum('ij,ij->i', X - 2 * np.ones(X.shape),
                            np.dot((X - 2 * np.ones(X.shape)), inv_theta.T)) \
            + np.log((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(Id))) - \
            np.log((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(theta_d)))

    def _grad_log_gamma_t(lambda_t, X):
        return (lambda_t - 1) * X - lambda_t * np.dot(inv_theta, (X - 2).T).T

    _experiment1 = ExperimentParameters(
        log_prior=_log_prior,
        log_likelihood=_log_likelihood,
        pi0=stats.multivariate_normal(np.zeros(d), Id),
        N=N, d=d, alpha=alpha,
        grad_log_gamma_t=_grad_log_gamma_t
    )
    return _experiment1
