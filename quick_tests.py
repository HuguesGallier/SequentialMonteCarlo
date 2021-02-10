import numpy as np
import pandas as pd
import time

from .experiments.gaussian import gaussian_experiment
from .utils import StatsSMC, ExperimentParameters
from .samplers import dict_samplers


def simple_test(sampler="TunedISMC", parameters: ExperimentParameters = None,
                parameters_function=gaussian_experiment,
                dimension=50, particles=1000, alpha=0.5, **kwargs):
    """ Run a single time the selectioned sampler. Return runned sampler.
    Also prints execution time.

    sampler: one of FearnheadTaylor, HMSC, MALASMC,
        Pretuning, TunedISMC.
    parameters: parameters of the experiment to be runned.
        Must be wrapped into proper Dataclass.
    parameters_function: Alternative to parameters. Allows
        for specifying some parameters in function call.
        One of parameters or parameters_function has to be set.
    dimension: dimension of the particles to sample. Not necessary
        if parameters is given.
    particles: number of particles to sample. Not necessary
        if parameters is given.
    alpha: rule for choosing next temperature. Not necessary
        if parameters is given.
    """

    sampler = dict_samplers.get(sampler)
    if parameters is None:
        parameters = parameters_function(dimension=50, particles=1000,
                                         alpha=0.5)
    sampler = sampler(parameters, **kwargs)

    start = time.time()
    sampler.run()
    execution_time = time.time() - start

    print(f"{particles} particles in dimension {dimension} have been"
          f"sampled in {round(execution_time, 2)} seconds.")

    return sampler


def performance_test(sampler="TunedISMC",
                     parameters_function=gaussian_experiment,
                     dimensions=[5, 10, 50, 100], particles=1000, alpha=0.5,
                     expectation=2, repetitions=40):
    """ Study the MSE for the mean of the first component. Return a dictionary
    with the MSE for each experiment at each dimension tested.
    Results are ploted.

    sampler: one of FearnheadTaylor, HMSC, MALASMC,
        Pretuning, TunedISMC.
    parameters_function: specifies parameters of the experiments at each
        dimension.
    dimensions: dimensions (of the particles) that we want to test.
    particles: number of particles to sampleat each dimension.
    alpha: rule for choosing next temperature. Not necessary
        if parameters is given.
    expectation: expectation of the first dimension of the sampled particles.
    """

    _sampler = dict_samplers.get(sampler)

    mse_over_dim = StatsSMC(_sampler).MSE_over_dimensions(
        true_mean=2,
        parameters_function=gaussian_experiment,
        dimensions=[5, 10, 50, 100],
        repetitions=40)

    mean_mse_over_dim = pd.DataFrame(
        [{d: np.mean(e) for d, e in mse_over_dim.items()}.values()],
        index=[sampler],
        columns=dimensions).T

    mean_mse_over_dim.plot(title='Empirical MSE')

    return mse_over_dim
