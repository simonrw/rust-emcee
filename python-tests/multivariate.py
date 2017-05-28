#!/usr/bin/env python


import emcee
import numpy as np


logprecision = -4


def lnprob_gaussian(x, icov):
    return -np.dot(x, np.dot(icov, x)) / 2.0


def check_sampler(sampler, N, p0, mean, cov):

    for _ in sampler.sample(p0, iterations=N):
        pass

    assert np.mean(sampler.acceptance_fraction) > 0.25
    assert np.all(sampler.acceptance_fraction > 0)

    chain = sampler.flatchain
    maxdiff = 10 ** logprecision
    assert np.all((np.mean(chain, axis=0) - mean) ** 2 / N ** 2
                  < maxdiff)
    assert np.all((np.cov(chain, rowvar=0) - cov) ** 2 / N ** 2
                  < maxdiff)


if __name__ == '__main__':
    np.random.seed(42)

    nwalkers = 100
    ndim = 5
    N = 1000

    mean = np.zeros(ndim)
    cov = 0.5 - np.random.rand(ndim ** 2) \
        .reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)
    icov = np.linalg.inv(cov)
    p0 = [0.1 * np.random.randn(ndim)
          for i in range(nwalkers)]
    truth = np.random.multivariate_normal(mean, cov, 100000)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob_gaussian, args=[icov, ])
    check_sampler(sampler, N * nwalkers, p0, mean, cov)
