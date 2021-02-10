## Brief description

This project is an implementation of the article: [Adaptive tuning of Hamiltonian Monte Carlo within sequential Monte Carlo](https://arxiv.org/pdf/1808.07730.pdf) (Alexander Buchholz, Nicolas Chopin, Pierre E. Jacob - Bayesian Analysis, advance publication, 31 July 2020). It consists in algorithms designed to tune MCMC kernels, and more specifically HMC kernels, in SMC samplers.

## Context

By starting to sample particles from an initial distribution π0, the **Sequential Monte Carlo** (SMC) samplers allow to approach a target distribution π by iterating a sequence of distributions πt: this is called **tempering**. This approach is a powerful alternative to MCMC in **Bayesian computation**. SMCs are robust to multimodality, they can benefit from parallel computing and they allox the computation of the normalizing constants, which can be useful for the choice of the model.

Particle propagation during iterations could be the most difficult part of the SMC and generally relies on MCMC kernels. Tuning the MCMC kernels' parameters aroused interest in recent years and the authors of the paper decided to focus on HMC (Hamiltonian Monte Carlo) kernels. Indeed, originally developed in Physics, HMC has become a standard tool in MCMC, in particular because of its better mixing in high dimensional problems.

The tuning of Markov kernels in SMC samplers can be related to the tuning of MCMC kernels in general. In this paper, the authors compare the methods of automatic tuning of HMC kernels within SMC. They first build on the work of Fearnhead and Taylor : [An Adaptive Sequential Monte Carlo Sampler](https://arxiv.org/pdf/1005.1193.pdf) (Bayesian Anal., Volume 8, Number 2, 411-438, 2013) and adapt their tuning procedure to HMC kernels. They then introduce an alternative approach based on a pre-tuning phase at each intermediate step.

Please consult the notebook for more details.

## Implementation

As mentioned above, we want to simulate new particles under a posterior distribution obtained from a prior distribution and a likelihood. The approach used to do this is tempering through SMC samplers.

5 different SMC samplers with different MCMC kernels are implemented:
- 3 with HMC kernels: 2 with different tuning procedures (**FearnheadTaylor** which is the approach of the authors, and **Pretuning**, which is the approach of Buchholtz et al.). Another HMC kernel without tuning (**HSMC**) is also proposed.
- 1 with a MALA kernel (**MALASMC**).
- 1 with an Independence Sampler using a tuning procedure (**TunedISMC**). This algorithm is described in [A sequential particle filter method for static models](https://pdfs.semanticscholar.org/47bc/c2b86f3b4ec2742e7ef1ac2868077f2aae92.pdf)
(Nicolas Chopin - Biometrika, 89, 3, pp. 539–551, 2002).

If you want to see briefly the performances of the aformentioned samplers, you can use the functions *simple_test* and *performance_test* in quick_test.py.

## Authors

Hugues Gallier, Guillaume Hoffmann, Christos Katsoulakis
