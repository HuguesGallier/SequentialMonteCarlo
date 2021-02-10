import autograd.numpy as np

from autograd import grad
from dataclasses import dataclass
from scipy import stats, optimize
from tqdm import tqdm
from types import FunctionType


@dataclass
class ExperimentParameters:
    """Store experiment parameters.

    Parameters:
    log_prior: function-- log of the prior distribution
    log-likelihood: function -- log of the likelihood
    pi0: scipy.stats distribution to simulate at time t=0
    grad_log_gamma_t: gradient of the log of the ratio between
        consecutive temperate distributions
    N: number of particles to simulate
    d: dimension
    alpha: rule to choose the next temperature at time t. """

    log_prior: FunctionType
    log_likelihood: FunctionType
    pi0: stats._distn_infrastructure.rv_frozen
    grad_log_gamma_t: FunctionType = None
    N: int = 1000
    d: int = 10
    alpha: float = 0.5

    def __post_init__(self):
        def _grad_log_gamma_t(lambda_t, X):
            gradl = grad(lambda x: self.log_gamma_t(lambda_t,
                                                    np.array([x]))[0])
            result = np.array([gradl(X[i]) for i in range(len(X))])
            return result
        self.grad_log_gamma_t = self.grad_log_gamma_t or \
            _grad_log_gamma_t

    def log_gamma_t(self, lambda_t, X):
        return self.log_prior(X) + self.log_likelihood(X) * lambda_t

    def log_ratio_gamma(self, lambda_t, lambda_t_prime, X):
        """Performs ratio of gamma function lambda_t_prime / gamma
        function lambda_t."""
        return self.log_likelihood(X) * (lambda_t_prime - lambda_t)


class SMCSampler:
    """Implement Sequential Monte Carlo methods using tempering.

    Parameter:
    exp_parameters: parameter of the experiment at hand -- dataclass
        described in utils containing all needed parameters."""

    def __init__(self, exp_parameters: ExperimentParameters):

        assert isinstance(exp_parameters, ExperimentParameters), \
            "Please use proper dataclass."

        self.param = exp_parameters
        self.X = None  # liste
        self.weights = np.zeros(exp_parameters.N)
        self.constants_ratio = []
        self.lambdas = [0]

    def run(self):  # algo 1
        t = 1
        lambda_t = 0
        while lambda_t < 1:

            if t == 1:
                X = self.param.pi0.rvs(size=self.param.N)

            else:
                X = self.move_through_kernel(X, lambda_t)       # algo 3, 5, 6

            lambda_tp1 = self.choose_next_lambda(X, lambda_t)
            log_weights = self.compute_log_weights(X, lambda_t, lambda_tp1)
            lambda_t = lambda_tp1

            # resampling
            weights = np.exp(log_weights)
            weights = weights / weights.sum()
            X = X[np.random.choice(np.arange(len(X)), size=len(X), p=weights)]

            self.X = X
            self.weights = weights
            self.constants_ratio += [np.mean(weights)]
            self.lambdas += [lambda_tp1]

            t += 1

    def move_through_kernel(self, X, lambda_t):
        raise NotImplementedError("Do not try to run this parent class.")

    def compute_log_weights(self, X, lambda_t, lambda_tp1):
        # cut weights arbitratitly to avoid overflow
        return np.minimum(100, self.param.log_ratio_gamma(lambda_t,
                                                          lambda_tp1, X))

    def choose_next_lambda(self, X, lambda_t):
        """Find next exponent according to algorithm 2 (Beskos et al. 2016)."""

        def ESS_alphaN(lbda):
            """Compute ESS in our special case for given lambda_tp1.
            Substract max of weights to avoid overflow.
            """
            log_weights = self.compute_log_weights(X, lambda_t, lbda)
            max_log_weights = np.max(log_weights)
            log_weights -= max_log_weights
            ESS = np.sum(np.exp(log_weights)) ** 2 / \
                np.sum(np.exp(2 * log_weights))
            return ESS - self.param.alpha * self.param.N

        if ESS_alphaN(1) >= 0:
            return 1

        lambda_tp1 = optimize.root_scalar(ESS_alphaN, bracket=[lambda_t, 1],
                                          method='brentq', xtol=1e-2)
        return lambda_tp1.root


class StatsSMC:

    def __init__(self, sampler: SMCSampler):
        """Argument : sampler -- uninstantiated class of SMCSampler."""
        self.sampler = sampler

    def MSE_mean_first_component(self, true_mean, parameters, repetitions):
        """Parameters -- class instantialised"""
        MSE_mean = np.empty(repetitions)
        for r in tqdm(range(repetitions)):

            # run sampler
            sampler = self.sampler(parameters)
            sampler.run()
            X = sampler.X

            # compute MSEs
            mean = np.mean(X[:, 0])
            MSE_mean[r] = np.mean((mean - true_mean)**2)

        return MSE_mean

    def MSE_over_dimensions(self, true_mean, parameters_function,
                            repetitions, dimensions):
        """parameters_functions -- like experiment1"""
        results = {}
        for d in dimensions:
            print('dimension currently computed: ', d)
            parameters = parameters_function(dimension=d)
            results[d] = self.MSE_mean_first_component(true_mean,
                                                       parameters, repetitions)

        return results

    def get_lambdas(self, repetitions, parameters_function, dimension):
        results = []
        for i in range(repetitions):
            sampler = self.sampler(parameters_function(dimension=dimension))
            sampler.run()
            results.append(sampler.lambdas)
        return results

    def evol_lambdas(self, repetitions, parameters_function, dimension):
        result_lambas = self.get_lambdas(repetitions, parameters_function,
                                         dimension)
        length = max(map(len, result_lambas))
        evol = np.array([xi+[1]*(length-len(xi)) for xi in result_lambas])
        return np.mean(evol, axis=0)

    def boxplot_first_component(self, repetitions, parameters_function,
                                dimension):
        results = []
        for i in range(repetitions):
            sampler = self.sampler(parameters_function(dimension=dimension))
            sampler.run()
            results.append(sampler.X[:, 0])
        return results
