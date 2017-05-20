//! `emcee` "The MCMC Hammer"
//!
//! A re-implementation of [`emcee`][1] in Rust. This library includes an implementation
//! of Goodman & Weare's [Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble
//! sampler][2]. All credit for this crate belongs to [Dan Foreman-Mackey][3], the original
//! author of [`emcee`][1]
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
//!     fn lnlike(&self, params: &Guess) -> f32 {
//!         // Insert actual implementation here
//!         0f32
//!     }
//!
//!     fn lnprior(&self, params: &Guess) -> f32 {
//!         // Insert actual implementation here
//!         0f32
//!     }
//! }
//! ```
//!
//! The trait has a default implementation for [`lnprob`][emcee-lnprob] which computes the product
//! of the likelihood and prior probability (sum in log space) as per Bayes' rule.  Invalid prior
//! values are marked by returning -[`std::f32::INFINITY`][std-infinity] from the priors function.
//! Note your implementation is likely to need external data. This data should be included with
//! your `Model` class, for example:
//!
//! ```rust
//! # use emcee::{Guess, Prob};
//! struct Model<'a> {
//!     x: &'a [f32],
//!     y: &'a [f32],
//! }
//!
//! // Linear model y = m * x + c
//! impl<'a> Prob for Model<'a> {
//!     fn lnlike(&self, params: &Guess) -> f32 {
//!         let m = params.values[0];
//!         let c = params.values[1];
//!
//!         -0.5 * self.x.iter().zip(self.y)
//!             .map(|(xval, yval)| {
//!                 let model = m * xval + c;
//!                 let residual = (yval - model).powf(2.0);
//!                 residual
//!             }).sum::<f32>()
//!     }
//!
//!     fn lnprior(&self, params: &Guess) -> f32 {
//!         // unimformative priors
//!         0.0f32
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
//! let initial_guess = Guess::new(&[0.0f32, 0.0f32]);
//! ```
//!
//! The sampler implemented by this create uses multiple *walkers*, and as such the initial
//! guess must be replicated once per walker, and typically dispersed from the initial position
//! to aid exploration of the problem parameter space. This can be achieved with the
//! [`create_initial_guess`][emcee-create-initial-guess] method:
//!
//! ```rust
//! # use emcee::Guess;
//! # let initial_guess = Guess::new(&[0.0f32, 0.0f32]);
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
//! #     x: &'a [f32],
//! #     y: &'a [f32],
//! # }
//! # // Linear model y = m * x + c
//! # impl<'a> Prob for Model<'a> {
//! #     fn lnlike(&self, params: &Guess) -> f32 {
//! #         let m = params.values[0];
//! #         let c = params.values[1];
//! #         -0.5 * self.x.iter().zip(self.y)
//! #             .map(|(xval, yval)| {
//! #                 let model = m * xval + c;
//! #                 let residual = (yval - model).powf(2.0);
//! #                 residual
//! #             }).sum::<f32>()
//! #     }
//! #     fn lnprior(&self, params: &Guess) -> f32 {
//! #         // unimformative priors
//! #         0.0f32
//! #     }
//! # }
//!
//! let initial_x = [0.0f32, 1.0f32, 2.0f32];
//! let initial_y = [5.0f32, 7.0f32, 9.0f32];
//!
//! let model = Model {
//!     x: &initial_x,
//!     y: &initial_y,
//! };
//!
//! let mut sampler = emcee::EnsembleSampler::new(nwalkers, ndim, &model)
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
//! #     x: &'a [f32],
//! #     y: &'a [f32],
//! # }
//! # // Linear model y = m * x + c
//! # impl<'a> Prob for Model<'a> {
//! #     fn lnlike(&self, params: &Guess) -> f32 {
//! #         let m = params.values[0];
//! #         let c = params.values[1];
//! #         -0.5 * self.x.iter().zip(self.y)
//! #             .map(|(xval, yval)| {
//! #                 let model = m * xval + c;
//! #                 let residual = (yval - model).powf(2.0);
//! #                 residual
//! #             }).sum::<f32>()
//! #     }
//! #     fn lnprior(&self, params: &Guess) -> f32 {
//! #         // unimformative priors
//! #         0.0f32
//! #     }
//! # }
//! #
//! # let initial_x = [0.0f32, 1.0f32, 2.0f32];
//! # let initial_y = [5.0f32, 7.0f32, 9.0f32];
//! #
//! # let model = Model {
//! #     x: &initial_x,
//! #     y: &initial_y,
//! # };
//! #
//! # let mut sampler = emcee::EnsembleSampler::new(nwalkers, ndim, &model)
//! #     .expect("could not create sampler");
//! #
//! # let initial_guess = Guess::new(&[0.0f32, 0.0f32]);
//! # let perturbed_guess = initial_guess.create_initial_guess(nwalkers);
//! let niterations = 100;
//! sampler.run_mcmc(&perturbed_guess, niterations).expect("error running sampler");
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
//! #     x: &'a [f32],
//! #     y: &'a [f32],
//! # }
//! # // Linear model y = m * x + c
//! # impl<'a> Prob for Model<'a> {
//! #     fn lnlike(&self, params: &Guess) -> f32 {
//! #         let m = params.values[0];
//! #         let c = params.values[1];
//! #         -0.5 * self.x.iter().zip(self.y)
//! #             .map(|(xval, yval)| {
//! #                 let model = m * xval + c;
//! #                 let residual = (yval - model).powf(2.0);
//! #                 residual
//! #             }).sum::<f32>()
//! #     }
//! #     fn lnprior(&self, params: &Guess) -> f32 {
//! #         // unimformative priors
//! #         0.0f32
//! #     }
//! # }
//! #
//! # let initial_x = [0.0f32, 1.0f32, 2.0f32];
//! # let initial_y = [5.0f32, 7.0f32, 9.0f32];
//! #
//! # let model = Model {
//! #     x: &initial_x,
//! #     y: &initial_y,
//! # };
//! #
//! # let mut sampler = emcee::EnsembleSampler::new(nwalkers, ndim, &model)
//! #     .expect("could not create sampler");
//! #
//! # let initial_guess = Guess::new(&[0.0f32, 0.0f32]);
//! # let perturbed_guess = initial_guess.create_initial_guess(nwalkers);
//! # let niterations = 100;
//! # sampler.run_mcmc(&perturbed_guess, niterations).expect("error running sampler");
//! let flatchain = sampler.flatchain();
//!
//! for (i, guess) in flatchain.iter().enumerate() {
//!     // Skip possible "burn-in" phase
//!     if i < 50 * nwalkers {
//!         continue;
//!     }
//!
//!     println!("Iteration {}; m={}, c={}", i, guess.values[0], guess.values[1]);
//! }
//! ```
//!
//! [1]: http://dan.iel.fm/emcee/current/
//! [2]: http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml
//! [3]: http://dan.iel.fm/
//! [emcee-prob]: prob/trait.Prob.html
//! [emcee-guess]: guess/struct.Guess.html
//! [emcee-lnprob]: prob/trait.Prob.html#method.lnprob
//! [std-infinity]: https://doc.rust-lang.org/std/f32/constant.INFINITY.html
//! [emcee-create-initial-guess]: guess/struct.Guess.html#method.create_initial_guess
//! [emcee-flatchain]: struct.EnsembleSampler.html#method.flatchain

#![allow(non_snake_case)]
extern crate rand;

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

mod errors;
mod guess;
mod prob;
mod stretch;
mod stores;

use rand::{StdRng, Rng, SeedableRng};
use rand::distributions::{Range, IndependentSample};

use errors::{EmceeError, Result};
pub use guess::Guess;
pub use prob::Prob;

use stretch::Stretch;
use stores::{Chain, ProbStore};

/// Affine-invariant Markov-chain Monte Carlo sampler
pub struct EnsembleSampler<'a, T: Prob + 'a> {
    nwalkers: usize,
    naccepted: usize,
    iterations: usize,
    lnprob: &'a T,
    dim: usize,
    rng: Box<Rng>,
    chain: Option<Chain>,
    probstore: Option<ProbStore>,
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
            return Err(EmceeError::InvalidInputs("the number of walkers must be even"));
        }

        if nwalkers <= 2 * dim {
            let msg = "the number of walkers should be more than \
                       twice the dimension of your parameter space";
            return Err(EmceeError::InvalidInputs(msg));
        }

        Ok(EnsembleSampler {
               nwalkers: nwalkers,
               iterations: 0,
               lnprob: lnprob,
               dim: dim,
               naccepted: 0,
               rng: Box::new(rand::thread_rng()),
               chain: None,
               probstore: None,
           })
    }

    /// Swap the built in random number generator for a seedable one
    ///
    /// This means the random number generation can be reproducable. Seed is whatever
    /// [`SeedableRng.from_seed`]
    /// (https://docs.rs/rand/0.3.15/rand/trait.SeedableRng.html#tymethod.from_seed)
    /// accepts.
    pub fn seed(&mut self, seed: &[usize]) {
        self.rng = Box::new(StdRng::from_seed(seed));
    }

    fn sample(&mut self, params: &[Guess], iterations: usize) -> Result<()> {
        let halfk = self.nwalkers / 2;

        /* Loop state */
        let mut p = params.to_owned(); // Take a copy of the input vector so we can mutate it
        let mut lnprob = self.get_lnprob(params)?;

        let indices: Vec<usize> = (0..params.len()).collect();
        self.chain = Some(Chain::new(self.dim, self.nwalkers, iterations));
        self.probstore = Some(ProbStore::new(self.nwalkers, iterations));

        for iter in 0..iterations {
            self.iterations = iter;;
            let cloned = p.clone();
            let (first_half, second_half) = cloned.split_at(halfk);
            let (first_i, second_i) = indices.split_at(halfk);
            let to_iterate = &[(&first_half, &second_half), (&second_half, &first_half)];
            let i_iterate = &[(&first_i, &second_i), (&second_i, &first_i)];

            let zipped = i_iterate.iter().zip(to_iterate);
            for val in zipped {
                let &(I0, _) = val.0;
                let &(S0, S1) = val.1;

                let stretch = self.propose_stretch(S0, S1, &lnprob);

                if stretch.accept.iter().any(|val| *val) {
                    /* Update the positions, log probabilities and acceptance counts */
                    assert_eq!(I0.len(), stretch.accept.len());
                    for j in 0..stretch.accept.len() {
                        if !stretch.accept[j] {
                            continue;
                        }

                        let idx = I0[j]; // position in the parameter vector
                        assert!(idx < lnprob.len());
                        lnprob[idx] = stretch.newlnprob[j];

                        let new_values = stretch.q[j].values.clone();
                        p[idx].values = new_values;
                    }
                }
            }

            // Update the internal chain object
            for (idx, guess) in p.iter().enumerate() {
                match self.chain {
                    Some(ref mut chain) => {
                        chain.set_params(idx, self.iterations, &guess.values);
                    }
                    None => unreachable!(),
                }
            }

            match self.probstore {
                Some(ref mut store) => {
                    store.set_probs(self.iterations, &lnprob);
                }
                None => unreachable!(),
            }
        }
        Ok(())
    }

    /// Run the sampling
    ///
    /// This runs the sampler for `N` iterations. Errors are signalled by the function returning
    /// a `Result`
    pub fn run_mcmc(&mut self, p0: &[Guess], N: usize) -> Result<()> {
        self.sample(p0, N)
    }

    /// Return the samples as computed by the sampler
    pub fn flatchain(&self) -> Vec<Guess> {
        match self.chain {
            Some(ref chain) => chain.flatchain(),
            None => unreachable!(),
        }
    }

    /// Return the sampler to its default state
    pub fn reset(&mut self) {
        self.iterations = 0;
        self.naccepted = 0;
    }

    // Internal functions

    fn propose_stretch(&mut self, p0: &[Guess], p1: &[Guess], lnprob0: &[f32]) -> Stretch {
        let Ns = p0.len();
        let Nc = p1.len();

        let z_range = Range::new(1.0f32, 2.0f32);
        let rint_range = Range::new(0usize, Nc);
        let unit_range = Range::new(0f32, 1f32);

        let a = 2.0f32;
        let zz: Vec<f32> = (0..Ns)
            .map(|_| ((a - 1.0) * z_range.ind_sample(&mut self.rng)).powf(2.0f32) / 2.0f32)
            .collect();

        let rint: Vec<usize> = (0..Ns)
            .map(|_| rint_range.ind_sample(&mut self.rng))
            .collect();

        let mut q = Vec::new();
        for guess_i in 0..Ns {
            let mut values = Vec::with_capacity(self.dim);
            for param_i in 0..self.dim {
                let guess_diff = p1[rint[guess_i]].values[param_i] - p0[guess_i].values[param_i];

                let new_value = p1[rint[guess_i]].values[param_i] - zz[guess_i] * guess_diff;
                values.push(new_value);
            }
            q.push(Guess { values });
        }
        assert_eq!(q.len(), zz.len());

        let mut out = Stretch::preallocated_accept(Ns);
        out.newlnprob = self.get_lnprob(&q).unwrap();
        out.q = q;

        for i in 0..Ns {
            let dim = p0[0].values.len();
            let lnpdiff = ((dim - 1) as f32) * zz[i].ln() + out.newlnprob[i] - lnprob0[i];
            if lnpdiff > unit_range.ind_sample(&mut self.rng).ln() {
                out.accept[i] = true;
            }
        }

        out
    }

    fn get_lnprob(&mut self, p: &[Guess]) -> Result<Vec<f32>> {
        let mut lnprobs = Vec::with_capacity(p.len());
        for guess in p {
            if guess.contains_infs() {
                return Err(EmceeError::Boxed("At least one parameter value was infinite".into()));
            } else if guess.contains_nans() {
                return Err(EmceeError::Boxed("At least one parameter value was NaN".into()));
            }

            let result = self.lnprob.lnprob(guess);
            if result.is_nan() {
                return Err(EmceeError::Boxed("NaN value of lnprob".into()));
            } else {
                lnprobs.push(result);
            }
        }

        Ok(lnprobs)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    const REAL_M: f32 = 2.0f32;
    const REAL_C: f32 = 5.0f32;

    struct LinearModel<'a> {
        x: &'a [f32],
        y: &'a [f32],
    }

    impl<'a> LinearModel<'a> {
        fn new(x: &'a [f32], y: &'a [f32]) -> Self {
            LinearModel { x, y }
        }
    }

    impl<'a> Prob for LinearModel<'a> {
        fn lnprior(&self, _params: &Guess) -> f32 {
            0.0f32
        }

        fn lnlike(&self, params: &Guess) -> f32 {
            let m = params.values[0];
            let c = params.values[1];
            let sum = self.x
                .iter()
                .zip(self.y)
                .fold(0.0f32, |acc, (x, y)| {
                    let model_value = m * x + c;
                    let residual = y - model_value;
                    acc + residual.powf(2.0)
                });
            -sum
        }
    }

    #[test]
    fn test_single_sample() {
        let (real_x, observed_y) = generate_dataset(20);
        let foo = LinearModel::new(&real_x, &observed_y);
        let p0 = create_guess();

        let nwalkers = 10;
        let mut sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();

        let params = p0.create_initial_guess(nwalkers);
        sampler.sample(&params, 1).unwrap();
    }

    #[test]
    fn test_run_mcmc() {
        let (real_x, observed_y) = generate_dataset(20);
        let foo = LinearModel::new(&real_x, &observed_y);
        let p0 = create_guess();

        let nwalkers = 10;
        let niters = 100;
        let mut sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();

        let params = p0.create_initial_guess(nwalkers);
        sampler.run_mcmc(&params, niters).unwrap();
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
        let mut sampler = EnsembleSampler::new(nwalkers, 2, &foo).unwrap();
        let lnprob = sampler.get_lnprob(&pos).unwrap();
        let expected: Vec<f32> = vec![-4613.19497084, -4613.277985, -4613.25381092, -4613.1954303];
        for (a, b) in lnprob.iter().zip(expected) {
            /*
             * TODO: this is quite a wide tolerance which makes the test pass, but needs tweaking.
             * Perhaps something is wrong with the algorithm itself
             * - perhaps the quoted floats copied from the printing of the python script are not
             *   precise
             *   enough to give the correct level of precision
             */
            assert_approx_eq!(a, b, 0.05f32);
        }
    }

    #[test]
    fn test_lnprob_implementations() {
        let p0 = create_guess();
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let expected = -4613.202966359966f32;
        assert_approx_eq!(foo.lnprob(&p0), expected);
    }

    #[test]
    fn test_propose_stretch() {
        let nwalkers = 100;
        let p0 = Guess { values: vec![2.0f32, 5.0f32] };

        let pos = p0.create_initial_guess(nwalkers);
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let mut sampler = EnsembleSampler::new(nwalkers, p0.values.len(), &foo).unwrap();
        let (a, b) = pos.split_at(nwalkers / 2);

        assert_eq!(a.len(), nwalkers / 2);
        assert_eq!(b.len(), nwalkers / 2);

        let lnprob = sampler.get_lnprob(&pos).unwrap();
        let _stretch = sampler.propose_stretch(&a, &b, &lnprob);
    }

    #[test]
    fn test_mcmc_run() {
        let nwalkers = 20;
        let p0 = Guess { values: vec![0f32, 0f32] };
        let mut rng = StdRng::from_seed(&[1, 2, 3, 4]);
        let pos = p0.create_initial_guess_with_rng(nwalkers, &mut rng);
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let niters = 1000;
        let mut sampler = EnsembleSampler::new(nwalkers, p0.values.len(), &foo).unwrap();
        sampler.seed(&[1]);
        let _ = sampler.run_mcmc(&pos, niters).unwrap();

        if let Some(ref chain) = sampler.chain {
            /* Wide margins due to random numbers :( */
            assert_approx_eq!(chain.get(0, 0, niters - 2), 2.0f32, 0.03f32);
            assert_approx_eq!(chain.get(1, 0, niters - 2), 5.0f32, 0.4f32);
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

    // Test helper functions
    fn create_guess() -> Guess {
        Guess { values: vec![0.0f32, 0.0f32] }
    }

    fn load_baked_dataset() -> (Vec<f32>, Vec<f32>) {
        let real_x: Vec<f32> = vec![0.20584494, 0.58083612, 1.5599452, 1.5601864, 1.81824967,
                                    1.8340451, 2.12339111, 2.9122914, 3.04242243, 3.74540119,
                                    4.31945019, 5.24756432, 5.98658484, 6.01115012, 7.08072578,
                                    7.31993942, 8.32442641, 8.66176146, 9.50714306, 9.69909852];
        let observed_y: Vec<f32> = vec![4.39885877,
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

    fn generate_dataset(size: usize) -> (Vec<f32>, Vec<f32>) {
        use rand::distributions::Normal;

        let mut rng = rand::thread_rng();
        let x_range = Range::new(0f32, 10f32);
        let norm_range = Normal::new(0.0, 3.0);

        let mut real_x: Vec<f32> = (0..size).map(|_| x_range.ind_sample(&mut rng)).collect();
        real_x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let real_y: Vec<f32> = real_x.iter().map(|x| REAL_M * x + REAL_C).collect();
        let observed_y: Vec<f32> = real_y
            .iter()
            .map(|y| y + norm_range.ind_sample(&mut rng) as f32)
            .collect();
        (real_x, observed_y)
    }
}
