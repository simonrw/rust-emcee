#!/usr/bin/env python


import emcee
import numpy as np

np.random.seed(42)

x = np.sort(np.random.uniform(0, 10, 20))
y = (2 * x + 5) + np.random.randn(x.size)

def lnprior(params):
    return 0.0

def lnlike(params):
    model = params[0] * x + params[1]
    residuals = y - model
    return -np.sum(residuals ** 2)

def lnprob(params):
    lnp = lnprior(params)
    if np.isfinite(lnp):
        return lnp + lnlike(params)
    return -np.inf


if __name__ == '__main__':
    print('x: {}'.format(x))
    print('y: {}'.format(y))
    p0 = np.array([0.0, 0.0])
    nwalkers = 4
    pos = np.array([p0 + np.random.randn() * 1E-5 for _ in range(nwalkers)])

    print('raw lnprob: {}'.format(lnprob(p0)))

    print('pos: {}'.format(pos))

    sampler = emcee.EnsembleSampler(nwalkers, p0.size, lnprob)
    print(sampler._get_lnprob(pos))
