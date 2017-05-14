extern crate rand;

use std::error::Error;

use rand::distributions::{Normal, IndependentSample};

enum EmceeError {
    Boxed(Box<Error>),
}

type Result<T> = ::std::result::Result<T, EmceeError>;
type GuessVector = Vec<Guess>;

#[derive(Debug)]
pub struct Guess {
    pub values: Vec<f32>,
}

impl Guess {
    fn new(values: &[f32]) -> Self {
        Guess {
            values: Vec::from(values),
        }
    }

    fn perturb(&self) -> Self {
        let mut new_values = self.values.clone();

        let normal = Normal::new(0.0, 1E-5);
        for i in 0..new_values.len() {
            new_values[i] = new_values[i] + normal.ind_sample(&mut rand::thread_rng()) as f32;
        }

        Guess {
            values: new_values,
        }
    }

    fn create_initial_guess(&self, nwalkers: usize) -> Vec<Self> {
        (0..nwalkers).map(|_| self.perturb()).collect()
    }

    fn contains_infs(&self) -> bool {
        self.values.iter().any(|val| val.is_infinite())
    }

    fn contains_nans(&self) -> bool {
        self.values.iter().any(|val| val.is_nan())
    }
}

pub struct RunResult {
}

pub trait Prob {
    fn lnlike(&self, params: &Guess) -> f32;
    fn lnprior(&self, params: &Guess) -> f32;

    fn lnprob(&self, params: &Guess) -> f32 {
        let lnp = self.lnprior(params);
        if lnp.is_finite() {
            lnp + self.lnlike(params)
        } else {
            std::f32::INFINITY
        }
    }
}

pub struct EnsembleSampler {
    nwalkers: usize,
    naccepted: usize,
    iterations: usize,
    lnprob: Box<Prob>,
    dim: usize,
    last_run_mcmc_result: Option<i32>, // TODO: this is not i32
}

impl EnsembleSampler {
    pub fn new(nwalkers: usize, dim: usize, lnprob: Box<Prob>) -> Self {
        EnsembleSampler {
            nwalkers: nwalkers,
            iterations: 0,
            lnprob: lnprob,
            dim: dim,
            naccepted: 0,
            last_run_mcmc_result: None,
        }
    }

    pub fn sample(&mut self, params: &GuessVector, iterations: usize) {
        let halfk = self.nwalkers / 2;
        for i in 0..iterations {}
    }

    pub fn run_mcmc(&mut self, p0: &GuessVector, N: usize) {}

    pub fn reset(&mut self) {
        self.iterations = 0;
        self.naccepted = 0;
        self.last_run_mcmc_result = None;
    }

    // Internal functions

    fn get_lnprob(&mut self, p: &GuessVector) -> Result<f32> {
        let mut run_results = Vec::with_capacity(p.len());
        for guess in p {
            if guess.contains_infs() {
                return Err(EmceeError::Boxed("At least one parameter value was infinite".into()));
            } else if guess.contains_nans() {
                return Err(EmceeError::Boxed("At least one parameter value was NaN".into()));
            }

            run_results.push(self.lnprob.lnprob(guess));
        }


        Ok(0.0f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pertubation() {
        let guess = Guess::new(&[1.0f32, 2.0f32]);
        let perturbed = guess.perturb();
        assert!(perturbed.values[0] != 1.0f32);
        assert!(perturbed.values[1] != 2.0f32);
    }

    #[test]
    fn test_initial_guess() {
        let guess = Guess::new(&[1.0f32, 2.0f32]);
        let initial = guess.create_initial_guess(10);
        assert_eq!(initial.len(), 10);
    }

    #[test]
    fn test_contains_infinites() {
        let guess = Guess::new(&[std::f32::INFINITY]);
        assert!(guess.contains_infs());

        let guess = Guess::new(&[0f32]);
        assert!(!guess.contains_infs());
    }

    #[test]
    fn test_contains_nans() {
        let guess = Guess::new(&[std::f32::NAN]);
        assert!(guess.contains_nans());

        let guess = Guess::new(&[0f32]);
        assert!(!guess.contains_nans());
    }
}
