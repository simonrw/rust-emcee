extern crate rand;

use std::error::Error;

use rand::distributions::{Range, Normal, IndependentSample};

#[derive(Debug)]
pub enum EmceeError {
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
        Guess { values: Vec::from(values) }
    }

    fn perturb(&self) -> Self {
        let mut new_values = self.values.clone();

        let normal = Normal::new(0.0, 1E-5);
        for i in 0..new_values.len() {
            new_values[i] = new_values[i] + normal.ind_sample(&mut rand::thread_rng()) as f32;
        }

        Guess { values: new_values }
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

pub struct RunResult {}

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

#[derive(Debug, Default)]
struct Stretch {
    q: GuessVector,
    newlnprob: Vec<f32>,
    accept: Vec<bool>,
}

pub struct EnsembleSampler {
    nwalkers: usize,
    naccepted: usize,
    iterations: usize,
    lnprob: Box<Prob>,
    chain: GuessVector,
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
            chain: Vec::new(),
            last_run_mcmc_result: None,
        }
    }

    pub fn sample(&mut self, params: &GuessVector, iterations: usize) -> Result<()> {
        let halfk = self.nwalkers / 2;
        let initial_lnprob = self.get_lnprob(params)?;
        let chain_len = self.chain.len();
        for i in 0..iterations {
            self.iterations += 1;
            let (first_half, second_half) = params.split_at(halfk);
            let to_iterate = &[(&first_half, &second_half), (&second_half, &first_half)];
            for &(S0, S1) in to_iterate {
                let stretch = self.propose_stretch(S0, S1, &initial_lnprob);
            }
        }
        Ok(())
    }

    pub fn run_mcmc(&mut self, p0: &GuessVector, N: usize) {}

    pub fn reset(&mut self) {
        self.iterations = 0;
        self.naccepted = 0;
        self.last_run_mcmc_result = None;
    }

    // Internal functions

    fn propose_stretch(&mut self,
                       p0: &[Guess],
                       p1: &[Guess],
                       lnprob0: &Vec<f32>)
                       -> Stretch {
        let Ns = p0.len();
        let Nc = p1.len();

        let z_range = Range::new(1.0f32, 2.0f32);
        let rint_range = Range::new(0usize, Nc);
        let unit_range = Range::new(0f32, 1f32);

        let a = 2.0f32;
        let mut q = Vec::new();
        let zz: Vec<f32> = (0..Ns)
            .map(|_| {
                     ((a - 1.0) * z_range.ind_sample(&mut rand::thread_rng()) + 1.0).powf(2.0f32) /
                     2.0f32
                 })
            .collect();
        for i in 0..Ns {
            let rint = rint_range.ind_sample(&mut rand::thread_rng());
            let ref s = p0[i].values;
            let ref c = p1[rint].values;

            let mut new_guess = Guess { values: Vec::with_capacity(Nc) };
            for j in 0..Nc {
                let val = c[j] - zz[i] * (c[j] * s[j]);
                new_guess.values.push(val);
            }
        }

        let mut out = Stretch::default();
        out.newlnprob = self.get_lnprob(&q).unwrap();
        out.q = q;
        out.accept.resize(out.newlnprob.len(), false);

        for i in 0..out.newlnprob.len() {
            let dim = p0[0].values.len();
            let lnpdiff = ((dim - 1) as f32) * zz[i].ln() + out.newlnprob[i] - lnprob0[i];
            if lnpdiff > unit_range.ind_sample(&mut rand::thread_rng()).ln() {
                out.accept[i] = true;
            }
        }

        out
    }

    fn get_lnprob(&mut self, p: &GuessVector) -> Result<Vec<f32>> {
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
