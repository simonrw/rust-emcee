#!/usr/bin/env python


import numpy as np
import emcee


def lnprior(params):
    return 0.0

def lnlike(params, x, y):
    model = params[0] * x + params[1]
    residuals = y - model
    return -np.sum(residuals ** 2)

def lnprob(params, x, y):
    lnp = lnprior(params)
    if np.isfinite(lnp):
        return lnp + lnlike(params, x, y)
    return -np.inf

if __name__ == '__main__':
    real_m, real_c = 2, 5

    real_x = np.sort(np.random.uniform(0, 10, 20))
    real_y = real_m * real_x + real_c
    noise = np.random.normal(0, 3, real_x.shape)
    observed_y = real_y + noise

    p0 = np.array([0, 0])

    nwalkers = 10
    niters = 100
    sampler = emcee.EnsembleSampler(nwalkers, len(p0), lnprob,
                                    args=(real_x, observed_y))

    pos = np.array([p0 + 1E-5 * np.random.randn()
                    for _ in range(nwalkers)])
    sampler.run_mcmc(pos, niters)
