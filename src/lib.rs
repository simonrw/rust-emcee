//! `emcee` "The MCMC Hammer"
//!
//! A re-implementation of [`emcee`][1] in Rust. This library includes an implementation
//! of Goodman & Weare's [Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble
//! sampler][2]. All credit for this crate belongs to [Dan Foreman-Mackey][3], the original
//! author of [`emcee`][1]
//!
//! ## Attribution
//!
//! If you make use of emcee in your work, please cite Dan's paper
//! ([arXiv](http://arxiv.org/abs/1202.3665),
//! [ADS](http://adsabs.harvard.edu/abs/2013PASP..125..306F),
//! [`BibTeX`]
//! (http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2013PASP..125..306F&data_type=BIBTEX)).
//!
//! ## Basic usage
//!
//! ### Implementing models
//!
//! The sampler requires a struct that implements [`emcee::Prob`][emcee-prob], for example:
//!
//! ```rust
//! use emcee::{Guess, Prob};
//!
//! struct Model;
//!
//! impl Prob for Model {
//!     fn lnlike(&self, params: &Guess) -> f64 {
//!         // Insert actual implementation here
//!         0f64
//!     }
//!
//!     fn lnprior(&self, params: &Guess) -> f64 {
//!         // Insert actual implementation here
//!         0f64
//!     }
//! }
//! ```
//!
//! The trait has a default implementation for [`lnprob`][emcee-lnprob] which computes the product
//! of the likelihood and prior probability (sum in log space) as per Bayes' rule.  Invalid prior
//! values are marked by returning -[`std::f64::INFINITY`][std-infinity] from the priors function.
//! Note your implementation is likely to need external data. This data should be included with
//! your `Model` class, for example:
//!
//! ```rust
//! # use emcee::{Guess, Prob};
//! struct Model<'a> {
//!     x: &'a [f64],
//!     y: &'a [f64],
//! }
//!
//! // Linear model y = m * x + c
//! impl<'a> Prob for Model<'a> {
//!     fn lnlike(&self, params: &Guess) -> f64 {
//!         let m = params[0];
//!         let c = params[1];
//!
//!         -0.5 * self.x.iter().zip(self.y)
//!             .map(|(xval, yval)| {
//!                 let model = m * xval + c;
//!                 let residual = (yval - model).powf(2.0);
//!                 residual
//!             }).sum::<f64>()
//!     }
//!
//!     fn lnprior(&self, params: &Guess) -> f64 {
//!         // unimformative priors
//!         0.0f64
//!     }
//! }
//!
//! ```
//!
//! ### Initial guess
//!
//! Next, construct an initial guess. A [`Guess`][emcee-guess] represents a proposal parameter
//! vector:
//!
//! ```rust
//! use emcee::Guess;
//!
//! let initial_guess = Guess::new(&[0.0f64, 0.0f64]);
//! ```
//!
//! The sampler implemented by this create uses multiple *walkers*, and as such the initial
//! guess must be replicated once per walker, and typically dispersed from the initial position
//! to aid exploration of the problem parameter space. This can be achieved with the
//! [`create_initial_guess`][emcee-create-initial-guess] method:
//!
//! ```rust
//! # use emcee::Guess;
//! # let initial_guess = Guess::new(&[0.0f64, 0.0f64]);
//! let nwalkers = 100;
//! let perturbed_guess = initial_guess.create_initial_guess(nwalkers);
//! assert_eq!(perturbed_guess.len(), nwalkers);
//! ```
//!
//! ### Constructing a sampler
//!
//! The sampler generates new parameter vectors, assess the probability using a user-supplied
//! probability model, accepts more likely parameter vectors and iterates for a number of
//! iterations.
//!
//! The sampler needs to know the number of walkers to use, which must be an even number
//! and at least twice the size of your parameter vector. It also needs the size of your
//! parameter vector, and your probability struct (which implements [`Prob`][emcee-prob]):
//!
//! ```rust
//! # use emcee::{Guess, Prob};
//! let nwalkers = 100;
//! let ndim = 2;  // m and c
//!
//! // Build a linear model y = m * x + c (see above)
//! # struct Model<'a> {
//! #     x: &'a [f64],
//! #     y: &'a [f64],
//! # }
//! # // Linear model y = m * x + c
//! # impl<'a> Prob for Model<'a> {
//! #     fn lnlike(&self, params: &Guess) -> f64 {
//! #         let m = params[0];
//! #         let c = params[1];
//! #         -0.5 * self.x.iter().zip(self.y)
//! #             .map(|(xval, yval)| {
//! #                 let model = m * xval + c;
//! #                 let residual = (yval - model).powf(2.0);
//! #                 residual
//! #             }).sum::<f64>()
//! #     }
//! #     fn lnprior(&self, params: &Guess) -> f64 {
//! #         // unimformative priors
//! #         0.0f64
//! #     }
//! # }
//!
//! let initial_x = [0.0f64, 1.0f64, 2.0f64];
//! let initial_y = [5.0f64, 7.0f64, 9.0f64];
//!
//! let model = Model {
//!     x: &initial_x,
//!     y: &initial_y,
//! };
//!
//! let sampler = emcee::EnsembleSampler::new(nwalkers, ndim, &model)
//!     .expect("could not create sampler");
//! ```
//!
//! Then run the sampler:
//!
//! ```rust
//! # use emcee::{Guess, Prob};
//! # let nwalkers = 100;
//! # let ndim = 2;  // m and c
//! # struct Model<'a> {
//! #     x: &'a [f64],
//! #     y: &'a [f64],
//! # }
//! # // Linear model y = m * x + c
//! # impl<'a> Prob for Model<'a> {
//! #     fn lnlike(&self, params: &Guess) -> f64 {
//! #         let m = params[0];
//! #         let c = params[1];
//! #         -0.5 * self.x.iter().zip(self.y)
//! #             .map(|(xval, yval)| {
//! #                 let model = m * xval + c;
//! #                 let residual = (yval - model).powf(2.0);
//! #                 residual
//! #             }).sum::<f64>()
//! #     }
//! #     fn lnprior(&self, params: &Guess) -> f64 {
//! #         // unimformative priors
//! #         0.0f64
//! #     }
//! # }
//! #
//! # let initial_x = [0.0f64, 1.0f64, 2.0f64];
//! # let initial_y = [5.0f64, 7.0f64, 9.0f64];
//! #
//! # let model = Model {
//! #     x: &initial_x,
//! #     y: &initial_y,
//! # };
//! #
//! # let sampler = emcee::EnsembleSampler::new(nwalkers, ndim, &model)
//! #     .expect("could not create sampler");
//! #
//! # let initial_guess = Guess::new(&[0.0f64, 0.0f64]);
//! # let perturbed_guess = initial_guess.create_initial_guess(nwalkers);
//! let niterations = 100;
//! sampler.run_mcmc(&perturbed_guess, niterations).expect("error running sampler");
//! ```
//!
//! #### Iterative sampling
//!
//! It is sometimes useful to get the internal values proposed and evaluated
//! during each proposal step of the sampler. In the Python version, the
//! method `sample` is a generator which can be iterated over to evaluate
//! the sample steps.
//!
//! In this Rust version, we provide this feature by exposing the
//! [`sample`][emcee-sample] method, which takes a callback, which is called
//! once per iteration with a single [`Step`][emcee-step] object. For
//! example:
//!
//! ```rust
//! # use emcee::{Guess, Prob};
//! # let nwalkers = 100;
//! # let ndim = 2;  // m and c
//! # struct Model<'a> {
//! #     x: &'a [f64],
//! #     y: &'a [f64],
//! # }
//! # // Linear model y = m * x + c
//! # impl<'a> Prob for Model<'a> {
//! #     fn lnlike(&self, params: &Guess) -> f64 {
//! #         let m = params[0];
//! #         let c = params[1];
//! #         -0.5 * self.x.iter().zip(self.y)
//! #             .map(|(xval, yval)| {
//! #                 let model = m * xval + c;
//! #                 let residual = (yval - model).powf(2.0);
//! #                 residual
//! #             }).sum::<f64>()
//! #     }
//! #     fn lnprior(&self, params: &Guess) -> f64 {
//! #         // unimformative priors
//! #         0.0f64
//! #     }
//! # }
//! #
//! # let initial_x = [0.0f64, 1.0f64, 2.0f64];
//! # let initial_y = [5.0f64, 7.0f64, 9.0f64];
//! #
//! # let model = Model {
//! #     x: &initial_x,
//! #     y: &initial_y,
//! # };
//! #
//! # let sampler = emcee::EnsembleSampler::new(nwalkers, ndim, &model)
//! #     .expect("could not create sampler");
//! #
//! # let initial_guess = Guess::new(&[0.0f64, 0.0f64]);
//! # let perturbed_guess = initial_guess.create_initial_guess(nwalkers);
//! # let niterations = 100;
//! sampler.sample(&perturbed_guess, niterations, |step| {
//!     println!("Current iteration: {}", step.iteration);
//!     println!("Current guess vectors: {:?}", step.pos);
//!     println!("Current log posterior probabilities: {:?}", step.lnprob);
//! });
//! ```
//!
//! ### Studying the results
//!
//! The samples are stored in the sampler's `flatchain` which is constructed through the
//! [`flatchain`][emcee-flatchain] method on the sampler:
//!
//! ```rust
//! # use emcee::{Guess, Prob};
//! # let nwalkers = 100;
//! # let ndim = 2;  // m and c
//! # struct Model<'a> {
//! #     x: &'a [f64],
//! #     y: &'a [f64],
//! # }
//! # // Linear model y = m * x + c
//! # impl<'a> Prob for Model<'a> {
//! #     fn lnlike(&self, params: &Guess) -> f64 {
//! #         let m = params[0];
//! #         let c = params[1];
//! #         -0.5 * self.x.iter().zip(self.y)
//! #             .map(|(xval, yval)| {
//! #                 let model = m * xval + c;
//! #                 let residual = (yval - model).powf(2.0);
//! #                 residual
//! #             }).sum::<f64>()
//! #     }
//! #     fn lnprior(&self, params: &Guess) -> f64 {
//! #         // unimformative priors
//! #         0.0f64
//! #     }
//! # }
//! #
//! # let initial_x = [0.0f64, 1.0f64, 2.0f64];
//! # let initial_y = [5.0f64, 7.0f64, 9.0f64];
//! #
//! # let model = Model {
//! #     x: &initial_x,
//! #     y: &initial_y,
//! # };
//! #
//! # let sampler = emcee::EnsembleSampler::new(nwalkers, ndim, &model)
//! #     .expect("could not create sampler");
//! #
//! # let initial_guess = Guess::new(&[0.0f64, 0.0f64]);
//! # let perturbed_guess = initial_guess.create_initial_guess(nwalkers);
//! # let niterations = 100;
//! # sampler.run_mcmc(&perturbed_guess, niterations).expect("error running sampler");
//! let flatchain = sampler.flatchain().unwrap();
//!
//! for (i, guess) in flatchain.iter().enumerate() {
//!     // Skip possible "burn-in" phase
//!     if i < 50 * nwalkers {
//!         continue;
//!     }
//!
//!     println!("Iteration {}; m={}, c={}", i, guess[0], guess[1]);
//! }
//! ```
//!
//! [1]: http://dan.iel.fm/emcee/current/
//! [2]: http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml
//! [3]: http://dan.iel.fm/
//! [emcee-prob]: trait.Prob.html
//! [emcee-guess]: struct.Guess.html
//! [emcee-lnprob]: trait.Prob.html#method.lnprob
//! [std-infinity]: https://doc.rust-lang.org/std/f64/constant.INFINITY.html
//! [emcee-create-initial-guess]: struct.Guess.html#method.create_initial_guess
//! [emcee-flatchain]: struct.EnsembleSampler.html#method.flatchain
//! [emcee-sample]: struct.EnsembleSampler.html#method.sample
//! [emcee-step]: struct.Step.html

#![warn(missing_docs)]

extern crate rand;

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

pub mod errors;
mod guess;
mod prob;
mod stretch;
mod stores;

use std::borrow::Cow;
use std::cell::RefCell;
use rand::{StdRng, Rng, SeedableRng};
use rand::distributions::{Range, IndependentSample};

use errors::*;
pub use guess::Guess;
pub use prob::Prob;

use stretch::Stretch;
use stores::{Chain, ProbStore};

/// Struct representing the current iteration evaluation
///
/// This struct is used with [`sample`][sample], which supplies a callback to
/// each loop step. An instance of this struct is passed to the callback.
///
/// Technical note: the `pos` and `lnprob` slices are implemented as `Cow`
/// wrapped objects, as during iteration, their contents are references (as the
/// values are stored with a longer lived lifetime in the sampler method),
/// whereas the function must return _owned_ values.
///
/// [sample]: struct.EnsembleSampler.html#method.sample
#[derive(Debug)]
pub struct Step<'a> {
    /// The current list of parameters, one for each walker
    pub pos: Cow<'a, [Guess]>,

    /// The log posterior probabilities of the values contained in `pos`, one for each walker
    pub lnprob: Cow<'a, [f64]>,

    /// The current iteration number
    pub iteration: usize,
}

/// Affine-invariant Markov-chain Monte Carlo sampler
pub struct EnsembleSampler<'a, T: Prob + 'a> {
    nwalkers: usize,
    lnprob: &'a T,
    dim: usize,
    proposal_scale: f64,

    // Mutable members
    rng: RefCell<Box<Rng>>,
    naccepted: RefCell<Vec<usize>>,
    iterations: RefCell<usize>,
    chain: RefCell<Option<Chain>>,
    probstore: RefCell<Option<ProbStore>>,
    initial_state: RefCell<Option<Step<'a>>>,

    /// Allow disabling of storing the chain
    storechain: bool,

    /// Thin the stored chains by this much
    pub thin: usize,
}

impl<'a, T: Prob + 'a> EnsembleSampler<'a, T> {
    /// Create a new `EnsembleSampler`
    ///
    /// Errors are handled by returning a [`Result`](errors/type.Result.html) which contains
    /// [`EmceeError::InvalidInputs`](errors/enum.EmceeError.html) error variant for the following
    /// errors:
    ///
    /// * the number of walkers must be even * the number of walkers must be at least twice the
    /// number of parameters
    pub fn new(nwalkers: usize, dim: usize, lnprob: &'a T) -> Result<Self> {
        if nwalkers % 2 != 0 {
            return Err(EmceeError::InvalidInputs("the number of walkers must be even".into()));
        }

        if nwalkers <= 2 * dim {
            let msg = "the number of walkers should be more than \
                       twice the dimension of your parameter space";
            return Err(EmceeError::InvalidInputs(msg.into()));
        }

        Ok(EnsembleSampler {
               nwalkers: nwalkers,
               iterations: RefCell::new(0),
               lnprob: lnprob,
               dim: dim,
               naccepted: RefCell::new(vec![0; nwalkers]),
               rng: RefCell::new(Box::new(rand::thread_rng())),
               proposal_scale: 2.0,
               chain: RefCell::new(None),
               probstore: RefCell::new(None),
               storechain: true,
               thin: 1,
               initial_state: RefCell::new(None),
           })
    }

    /// Swap the built in random number generator for a seedable one
    ///
    /// This means the random number generation can be reproducable. Seed is whatever
    /// [`SeedableRng.from_seed`]
    /// (https://docs.rs/rand/0.3.15/rand/trait.SeedableRng.html#tymethod.from_seed)
    /// accepts.
    pub fn seed(&self, seed: &[usize]) {
        *self.rng.borrow_mut() = Box::new(StdRng::from_seed(seed));
    }

    /// Run the sampler with a callback called on each iteration
    ///
    /// On each iteration, this function is called with an instance of [`Step`][step] in the new
    /// proposal position. The callback is passed as mutable so it can interact with state from the
    /// calling site.
    ///
    /// [step]: struct.Step.html
    pub fn sample<F>(&self, params: &[Guess], iterations: usize, mut callback: F) -> Result<Step>
        where F: FnMut(Step)
    {

        let mut p = match *self.initial_state.borrow() {
            None => params.to_owned(),
            Some(ref state) => state.pos.clone().into_owned(),
        };

        let mut lnprob = match *self.initial_state.borrow() {
            None => self.get_lnprob(&p)?,
            Some(ref state) => state.lnprob.clone().into_owned(),
        };

        let halfk = self.nwalkers / 2;

        if lnprob.iter().any(|val| val.is_nan()) {
            return Err("The initial lnprob was NaN.".into());
        }

        if self.storechain {
            *self.chain.borrow_mut() = Some(Chain::new(self.dim, self.nwalkers, iterations));
            *self.probstore.borrow_mut() = Some(ProbStore::new(self.nwalkers, iterations));
        }

        self.naccepted.borrow_mut().resize(self.nwalkers, 0);

        for iteration in 0..iterations {

            for ensemble_idx in 0..2 {
                let (first, second) = if ensemble_idx == 0 {
                    p.split_at_mut(halfk)
                } else {
                    let (second, first) = p.split_at_mut(halfk);
                    (first, second)
                };

                let (lnprob_slice, _) = if ensemble_idx == 0 {
                    lnprob.split_at_mut(halfk)
                } else {
                    let (second, first) = lnprob.split_at_mut(halfk);
                    (first, second)
                };

                assert_eq!(first.len(), halfk);
                assert_eq!(second.len(), halfk);
                assert_eq!(lnprob_slice.len(), halfk);

                let stretch = self.propose_stretch(first, second, lnprob_slice)?;

                if stretch.accept.iter().any(|val| *val) {
                    /* Some walkers have accepted new positions, so update the store variables */
                    for walker_idx in 0..halfk {
                        if !stretch.accept[walker_idx] {
                            continue;
                        }

                        lnprob_slice[walker_idx] = stretch.newlnprob[walker_idx];
                        /* Update the param vector values */
                        for (param_idx, param) in stretch.q[walker_idx].values.iter().enumerate() {
                            first[walker_idx][param_idx] = *param;
                        }
                        let real_walker_idx = walker_idx + ensemble_idx * halfk;
                        {
                            let mut naccepted = self.naccepted.borrow_mut();
                            naccepted[real_walker_idx] += 1;
                        }
                    }
                }
            }

            /* Update the store variables with the new parameter values */
            if iteration % self.thin == 0 {
                let iteration = iteration / self.thin;
                for (walker_idx, p_value) in p.iter().enumerate() {
                    let mut chain = self.chain.borrow_mut();
                    if let Some(ref mut chain) = *chain {
                        chain.set_params(walker_idx, iteration, &p_value.values);
                    }

                    let mut store = self.probstore.borrow_mut();
                    if let Some(ref mut store) = *store {
                        store.set_probs(iteration, &lnprob);
                    }
                }
            }

            let step = Step {
                pos: Cow::Borrowed(&p),
                lnprob: Cow::Borrowed(&lnprob),
                iteration: iteration,
            };

            callback(step);

            *self.iterations.borrow_mut() += 1;
        }

        let step = Step {
            pos: Cow::Owned(p),
            lnprob: Cow::Owned(lnprob),
            iteration: iterations - 1,
        };

        Ok(step)
    }

    /// Run the sampling
    ///
    /// This runs the sampler for `niterations` iterations. Errors are signalled by the function
    /// returning a `Result`
    pub fn run_mcmc(&self, p0: &[Guess], niterations: usize) -> Result<Step> {
        self.sample(p0, niterations, |_step| {})
    }


    /// Set the initial state of the sampler
    pub fn set_initial_state(&self, state0: Step<'a>) -> &Self {
        *self.initial_state.borrow_mut() = Some(state0);
        self
    }

    /// Return the samples as computed by the sampler
    pub fn flatchain(&self) -> Option<Vec<Guess>> {
        let mut chain = self.chain.borrow_mut();
        match *chain {
            Some(ref mut chain) => Some(chain.flatchain()),
            None => None,
        }
    }

    /// Return the number of iterations accepted, one value per walker
    pub fn acceptance_fraction(&self) -> Vec<f64> {
        self.naccepted
            .borrow()
            .iter()
            .map(|naccepted| *naccepted as f64 / *self.iterations.borrow() as f64)
            .collect()
    }

    /// Return the sampler to its default state
    pub fn reset(&self) {
        *self.iterations.borrow_mut() = 0;
        self.naccepted.borrow_mut().resize(0, 0);
        *self.chain.borrow_mut() = None;
        *self.probstore.borrow_mut() = None;
    }

    // Internal functions

    fn propose_stretch(&self, p0: &[Guess], p1: &[Guess], lnprob0: &[f64]) -> Result<Stretch> {
        assert_eq!(p0.len() + p1.len(), self.nwalkers);
        let s = p0;
        let c = p1;
        let ns = s.len();
        let nc = c.len();

        // let z_range = Range::new(1.0f64, 2.0f64);
        let rint_range = Range::new(0usize, nc);
        let unit_range = Range::new(0f64, 1f64);

        let mut q = Vec::with_capacity(ns);
        let mut all_zz = Vec::with_capacity(ns);
        for sval in s {
            let zz = ((self.proposal_scale - 1.0) *
                      unit_range.ind_sample(&mut *self.rng.borrow_mut()) +
                      1.0f64)
                    .powf(2.0f64) / self.proposal_scale;
            let rint = rint_range.ind_sample(&mut *self.rng.borrow_mut());

            let mut values = Vec::with_capacity(self.dim);
            for (param_i, s_param) in sval.values.iter().enumerate() {
                let random_c = c[rint][param_i];
                let guess_diff = random_c - s_param;
                let new_value = random_c - zz * guess_diff;
                values.push(new_value);
            }
            q.push(Guess { values });
            all_zz.push(zz);
        }
        assert_eq!(q.len(), all_zz.len());

        let mut out = Stretch::preallocated_accept(ns);
        out.newlnprob = self.get_lnprob(&q)?;
        out.q = q;

        assert_eq!(out.newlnprob.len(), ns);

        for i in 0..ns {
            assert!(all_zz[i] > 0.);
            let lnpdiff = (self.dim as f64 - 1.0) * all_zz[i].ln() + out.newlnprob[i] - lnprob0[i];
            let test_value = unit_range.ind_sample(&mut *self.rng.borrow_mut()).ln();

            if lnpdiff > test_value {
                out.accept[i] = true;
            }
        }
        Ok(out)
    }

    fn get_lnprob(&self, p: &[Guess]) -> Result<Vec<f64>> {
        let mut lnprobs = Vec::with_capacity(p.len());
        for guess in p {
            if guess.contains_infs() {
                return Err("At least one parameter value was infinite".into());
            } else if guess.contains_nans() {
                return Err("At least one parameter value was NaN".into());
            }

            let result = self.lnprob.lnprob(guess);
            if result.is_nan() {
                return Err("NaN value of lnprob".into());
            }

            lnprobs.push(result);
        }

        Ok(lnprobs)
    }
}

#[cfg(test)]
mod tests {
    use rand::distributions::Normal;
    use super::*;

    const REAL_M: f64 = 2.0f64;
    const REAL_C: f64 = 5.0f64;

    struct LinearModel<'a> {
        x: &'a [f64],
        y: &'a [f64],
    }

    impl<'a> LinearModel<'a> {
        fn new(x: &'a [f64], y: &'a [f64]) -> Self {
            LinearModel { x, y }
        }
    }

    impl<'a> Prob for LinearModel<'a> {
        fn lnprior(&self, _params: &Guess) -> f64 {
            0.0f64
        }

        fn lnlike(&self, params: &Guess) -> f64 {
            let m = params[0];
            let c = params[1];
            let sum = self.x
                .iter()
                .zip(self.y)
                .fold(0.0f64, |acc, (x, y)| {
                    let model_value = m * x + c;
                    let residual = y - model_value;
                    acc + residual.powf(2.0)
                });
            -sum
        }
    }


    struct MultivariateProb<'a> {
        icov: &'a [[f64; 5]; 5],
    }

    impl<'a> Prob for MultivariateProb<'a> {
        // Stub methods as they are not used
        fn lnlike(&self, _params: &Guess) -> f64 {
            0.0f64
        }
        fn lnprior(&self, _params: &Guess) -> f64 {
            0.0f64
        }

        fn lnprob(&self, params: &Guess) -> f64 {
            let mut values = [0f64; 5];
            for (i, value) in params.values.iter().enumerate() {
                values[i] = *value;
            }
            let inv_prod = mat_vec_mul(&self.icov, &values);
            -vec_vec_mul(&values, &inv_prod) / 2.0
        }
    }


    #[test]
    fn test_single_sample() {
        let (real_x, observed_y) = generate_dataset(20);
        let foo = LinearModel::new(&real_x, &observed_y);
        let p0 = create_guess();

        let nwalkers = 10;
        let sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();

        let params = p0.create_initial_guess(nwalkers);
        sampler.run_mcmc(&params, 1).unwrap();
        assert_eq!(*sampler.iterations.borrow(), 1);
    }

    #[test]
    fn test_sample_with_callback() {
        let (real_x, observed_y) = generate_dataset(20);
        let foo = LinearModel::new(&real_x, &observed_y);
        let p0 = create_guess();

        let nwalkers = 10;
        let sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();

        let params = p0.create_initial_guess(nwalkers);

        let mut counter = 0;

        sampler.sample(&params, 2, |_step| counter += 1).unwrap();
        assert_eq!(counter, 2);
        assert_eq!(*sampler.iterations.borrow(), 2);
    }

    #[test]
    fn test_run_mcmc() {
        let (real_x, observed_y) = generate_dataset(20);
        let foo = LinearModel::new(&real_x, &observed_y);
        let p0 = create_guess();

        let nwalkers = 10;
        let niters = 100;
        let sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();

        let params = p0.create_initial_guess(nwalkers);
        sampler.run_mcmc(&params, niters).unwrap();
    }

    #[test]
    fn test_with_initial_state() {
        let (real_x, observed_y) = generate_dataset(20);
        let foo = LinearModel::new(&real_x, &observed_y);
        let p0 = create_guess();

        let nwalkers = 10;
        let niters = 100;
        let sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();


        let params = p0.create_initial_guess(nwalkers);
        let lnprob0: Vec<_> = params.iter().map(|val| foo.lnprob(val)).collect();

        let initial_state = Step {
            pos: Cow::Owned(params.clone()),
            lnprob: Cow::Owned(lnprob0.clone()),
            iteration: 0,
        };

        sampler
            .set_initial_state(initial_state)
            .run_mcmc(&params, niters)
            .unwrap();
    }

    #[test]
    fn test_restart_sampler() {
        let (real_x, observed_y) = generate_dataset(20);
        let foo = LinearModel::new(&real_x, &observed_y);
        let p0 = create_guess();

        let nwalkers = 10;
        let niters = 100;
        let sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();


        let params = p0.create_initial_guess(nwalkers);

        let state = sampler.run_mcmc(&params, niters).unwrap();

        sampler.reset();
        sampler
            .set_initial_state(state)
            .run_mcmc(&params, niters)
            .unwrap();
    }

    #[test]
    fn test_lnprob() {
        /* PYTHON OUTPUT
         *  x: [ 0.20584494  0.58083612  1.5599452   1.5601864   1.81824967  1.8340451
         *   2.12339111  2.9122914   3.04242243  3.74540119  4.31945019  5.24756432
         *   5.98658484  6.01115012  7.08072578  7.31993942  8.32442641  8.66176146
         *   9.50714306  9.69909852]
         *
         * y: [  4.39885877   6.47591958   7.21186633   6.70806911  10.10214811
         *    8.4423139    9.31431042   9.39983462  10.54046213  12.60172497
         *   12.4879068   15.87082665  16.37253099  16.73060649  18.55974494
         *   21.49215702  21.63535559  21.26581199  24.83683104  23.17735339]
         *
         *
         * pos: [[  2.08863595e-06   2.08863595e-06]
         * [ -1.95967012e-05  -1.95967012e-05]
         * [ -1.32818605e-05  -1.32818605e-05]
         * [  1.96861236e-06   1.96861236e-06]]

         * result: array([-4613.19497084, -4613.277985  , -4613.25381092, -4613.1954303 ])
         */
        let (real_x, observed_y) = load_baked_dataset();
        let mut pos = Vec::new();
        pos.push(Guess { values: vec![2.08863595e-06, 2.08863595e-06] });
        pos.push(Guess { values: vec![-1.95967012e-05, -1.95967012e-05] });
        pos.push(Guess { values: vec![-1.32818605e-05, -1.32818605e-05] });
        pos.push(Guess { values: vec![1.96861236e-06, 1.96861236e-06] });
        let foo = LinearModel::new(&real_x, &observed_y);

        let nwalkers = 8;
        let sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();
        let lnprob = sampler.get_lnprob(&pos).unwrap();
        let expected: Vec<f64> = vec![-4613.19497084, -4613.277985, -4613.25381092, -4613.1954303];
        for (a, b) in lnprob.iter().zip(expected) {
            assert_approx_eq!(a, b);
        }
    }

    #[test]
    fn test_lnprob_implementations() {
        let p0 = create_guess();
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let expected = -4613.202966359966f64;
        assert_approx_eq!(foo.lnprob(&p0), expected);
    }

    #[test]
    fn test_propose_stretch() {
        let nwalkers = 100;
        let p0 = Guess { values: vec![2.0f64, 5.0f64] };

        let pos = p0.create_initial_guess(nwalkers);
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let sampler = EnsembleSampler::new(nwalkers, p0.values.len(), &foo).unwrap();
        let (a, b) = pos.split_at(nwalkers / 2);

        assert_eq!(a.len(), nwalkers / 2);
        assert_eq!(b.len(), nwalkers / 2);

        let lnprob = sampler.get_lnprob(&pos).unwrap();
        let _stretch = sampler.propose_stretch(&a, &b, &lnprob).unwrap();
    }

    #[test]
    fn test_mcmc_run() {
        let nwalkers = 20;
        let p0 = Guess { values: vec![0f64, 0f64] };
        let mut rng = StdRng::from_seed(&[1, 2, 3, 4]);
        let pos = p0.create_initial_guess_with_rng(nwalkers, &mut rng);
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let niters = 1000;
        let sampler = EnsembleSampler::new(nwalkers, p0.values.len(), &foo).unwrap();
        sampler.seed(&[0]);
        let _ = sampler.run_mcmc(&pos, niters).unwrap();

        let chain = sampler.chain.borrow();
        if let Some(ref chain) = *chain {
            /* Wide margins due to random numbers :( */
            assert_approx_eq!(chain.get(0, 0, niters - 2), 2.0f64, 1.0f64);
            assert_approx_eq!(chain.get(1, 0, niters - 2), 5.0f64, 1.0f64);
        }
    }

    #[test]
    fn test_nwalkers_even() {
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);
        match EnsembleSampler::new(3, 3, &foo) {
            Err(EmceeError::InvalidInputs(msg)) => {
                assert!(msg.contains("number of walkers must be even"));
            }
            _ => panic!("incorrect"),
        }
    }

    #[test]
    fn test_enough_walkers() {
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);
        match EnsembleSampler::new(4, 3, &foo) {
            Err(EmceeError::InvalidInputs(msg)) => {
                assert!(msg.contains("should be more than twice"));
            }
            _ => panic!("incorrect"),
        }
    }

    #[test]
    fn test_lnprob_gaussian() {
        // Testing the probability model for the multivariate test
        let icov = [[318.92634269,
                     531.39511426,
                     -136.10315845,
                     154.17685545,
                     552.308813],
                    [531.39511426,
                     899.91793286,
                     -224.74333441,
                     258.98686842,
                     938.32014715],
                    [-136.10315845,
                     -224.74333441,
                     60.61145495,
                     -66.68898448,
                     -232.52035701],
                    [154.17685545,
                     258.98686842,
                     -66.68898448,
                     83.9979827,
                     266.44429402],
                    [552.308813,
                     938.32014715,
                     -232.52035701,
                     266.44429402,
                     983.33032073]];

        let guesses = &[Guess::new(&[5., 5., 5., 5., 5.]),
                        Guess::new(&[5., 5., 5., 5., 9.]),
                        Guess::new(&[5., 120., 5., 5., 9.])];
        let expecteds = &[-80374.2068729, -138398.513797, -7902962.23125];

        for (guess, expected) in guesses.iter().zip(expecteds) {
            let p = MultivariateProb { icov: (&icov) };
            assert_approx_eq!(p.lnprob(&guess), expected, 1E-4);
        }
    }

    #[test]
    fn test_not_storing_chain() {
        let nwalkers = 20;
        let p0 = Guess { values: vec![0f64, 0f64] };
        let mut rng = StdRng::from_seed(&[1, 2, 3, 4]);
        let pos = p0.create_initial_guess_with_rng(nwalkers, &mut rng);
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let niters = 1000;
        let mut sampler = EnsembleSampler::new(nwalkers, p0.values.len(), &foo).unwrap();
        sampler.seed(&[0]);
        sampler.storechain = false;
        let _ = sampler.run_mcmc(&pos, niters).unwrap();
        assert!(sampler.chain.borrow().is_none());
    }

    #[test]
    fn test_thinning() {
        let nwalkers = 20;
        let p0 = Guess { values: vec![0f64, 0f64] };
        let mut rng = StdRng::from_seed(&[1, 2, 3, 4]);
        let pos = p0.create_initial_guess_with_rng(nwalkers, &mut rng);
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let niters = 1000;
        let mut sampler = EnsembleSampler::new(nwalkers, p0.values.len(), &foo).unwrap();
        sampler.seed(&[0]);
        sampler.thin = 500;
        let _ = sampler.run_mcmc(&pos, niters).unwrap();
        let chain = sampler.chain.borrow();
        match *chain {
            Some(ref chain) => assert_eq!(chain.niterations, niters),
            None => panic!("chain should not be `None`"),
        }
    }

    #[test]
    fn test_multivariate() {
        let nwalkers = 100;
        let ndim = 5;
        let niter = 1000;

        let icov = [[318.92634269,
                     531.39511426,
                     -136.10315845,
                     154.17685545,
                     552.308813],
                    [531.39511426,
                     899.91793286,
                     -224.74333441,
                     258.98686842,
                     938.32014715],
                    [-136.10315845,
                     -224.74333441,
                     60.61145495,
                     -66.68898448,
                     -232.52035701],
                    [154.17685545,
                     258.98686842,
                     -66.68898448,
                     83.9979827,
                     266.44429402],
                    [552.308813,
                     938.32014715,
                     -232.52035701,
                     266.44429402,
                     983.33032073]];
        let model = MultivariateProb { icov: &icov };

        let norm_range = Normal::new(0.0f64, 1.0f64);
        let mut rng = StdRng::from_seed(&[1, 2, 3, 4]);
        let p0: Vec<_> = (0..nwalkers)
            .map(|_| {
                     Guess {
                         values: (0..ndim)
                             .map(|_| 0.1f64 * norm_range.ind_sample(&mut rng) as f64)
                             .collect(),
                     }
                 })
            .collect();

        let sampler = EnsembleSampler::new(nwalkers, ndim, &model).unwrap();
        sampler.seed(&[1]);
        check_sampler(&sampler, niter, &p0);
    }

    // Test helper functions
    fn check_sampler<'a, T: Prob + 'a>(sampler: &EnsembleSampler<'a, T>,
                                       niter: usize,
                                       p0: &[Guess]) {
        let _ = sampler.run_mcmc(&p0, niter).unwrap();

        let chain = sampler.flatchain().unwrap();
        let maxdiff = 1E-4;

        // Check the acceptance fraction
        let acceptance_fraction = sampler.acceptance_fraction();
        assert!(acceptance_fraction.iter().sum::<f64>() / acceptance_fraction.len() as f64 > 0.25);

        let mut invalid_walkers = Vec::new();

        // Small struct to add context to the invalid walker description
        #[derive(Debug)]
        struct I {
            idx: usize,
            fraction: f64,
        }

        for (i, fraction) in acceptance_fraction.iter().enumerate() {
            if *fraction == 0.0f64 {
                invalid_walkers.push(I {
                                         idx: i,
                                         fraction: *fraction,
                                     });
            }
        }

        let split = acceptance_fraction.split_at(acceptance_fraction.len() / 2);
        assert!(invalid_walkers.len() == 0,
                "Found {} invalid walkers: {:?} (AF1: {:?}, AF2: {:?})",
                invalid_walkers.len(),
                invalid_walkers,
                split.0,
                split.1);

        // Check the chain
        let mut result = Guess { values: vec![0.0f64; sampler.dim] };

        for i in 0..sampler.nwalkers * niter {
            for j in 0..sampler.dim {
                result[j] += (chain[i][j] / niter as f64).powf(2.0);
            }
        }

        for value in result.values {
            assert!((value / niter as f64).powf(2.0) < maxdiff,
                    "value: {}, maxdiff: {}",
                    value,
                    maxdiff);
        }
    }

    fn create_guess() -> Guess {
        Guess { values: vec![0.0f64, 0.0f64] }
    }

    fn load_baked_dataset() -> (Vec<f64>, Vec<f64>) {
        let real_x: Vec<f64> = vec![0.20584494, 0.58083612, 1.5599452, 1.5601864, 1.81824967,
                                    1.8340451, 2.12339111, 2.9122914, 3.04242243, 3.74540119,
                                    4.31945019, 5.24756432, 5.98658484, 6.01115012, 7.08072578,
                                    7.31993942, 8.32442641, 8.66176146, 9.50714306, 9.69909852];
        let observed_y: Vec<f64> = vec![4.39885877,
                                        6.47591958,
                                        7.21186633,
                                        6.70806911,
                                        10.10214811,
                                        8.4423139,
                                        9.31431042,
                                        9.39983462,
                                        10.54046213,
                                        12.60172497,
                                        12.4879068,
                                        15.87082665,
                                        16.37253099,
                                        16.73060649,
                                        18.55974494,
                                        21.49215702,
                                        21.63535559,
                                        21.26581199,
                                        24.83683104,
                                        23.17735339];
        (real_x, observed_y)
    }

    fn generate_dataset(size: usize) -> (Vec<f64>, Vec<f64>) {
        use rand::distributions::Normal;

        let mut rng = rand::thread_rng();
        let x_range = Range::new(0f64, 10f64);
        let norm_range = Normal::new(0.0, 3.0);

        let mut real_x: Vec<f64> = (0..size).map(|_| x_range.ind_sample(&mut rng)).collect();
        real_x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let real_y: Vec<f64> = real_x.iter().map(|x| REAL_M * x + REAL_C).collect();
        let observed_y: Vec<f64> = real_y
            .iter()
            .map(|y| y + norm_range.ind_sample(&mut rng) as f64)
            .collect();
        (real_x, observed_y)
    }

    fn mat_vec_mul(m: &[[f64; 5]; 5], v: &[f64; 5]) -> [f64; 5] {
        let mut out = [0.0f64; 5];

        for i in 0..5 {
            out[i] = v[0] * m[i][0] + v[1] * m[i][1] + v[2] * m[i][2] + v[3] * m[i][3] +
                     v[4] * m[i][4];
        }

        out
    }

    fn vec_vec_mul(v1: &[f64; 5], v2: &[f64; 5]) -> f64 {
        let mut out = 0.0f64;
        for i in 0..5 {
            out += v1[i] * v2[i];
        }
        out
    }

    #[test]
    fn test_mat_vec_mul() {
        let v = [1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64];
        let mat = [[1.0f64, 0.0f64, 0.0f64, 5.0f64, 0.0f64],
                   [0.0f64, 1.0f64, 0.0f64, 0.0f64, 0.0f64],
                   [0.0f64, 0.0f64, 1.0f64, 0.0f64, 0.0f64],
                   [0.0f64, 0.0f64, 0.0f64, 1.0f64, 0.0f64],
                   [0.0f64, 0.0f64, 0.0f64, 0.0f64, 1.0f64]];
        let result = mat_vec_mul(&mat, &v);
        assert_eq!(result, [21.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64]);
    }

    #[test]
    fn test_vec_vec_mul() {
        let v = [1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64];
        let result = vec_vec_mul(&v, &v);
        assert_eq!(result, 55.0f64);
    }
}
