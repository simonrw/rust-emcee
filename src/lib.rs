#![allow(non_snake_case)]
extern crate rand;

use std::error::Error;

use rand::distributions::{Range, Normal, IndependentSample};

#[derive(Debug)]
pub enum EmceeError {
    Boxed(Box<Error>),
}

type Result<T> = ::std::result::Result<T, EmceeError>;

#[derive(Debug, Clone)]
pub struct Guess {
    pub values: Vec<f32>,
}

impl Guess {
    pub fn new(values: &[f32]) -> Self {
        Guess { values: Vec::from(values) }
    }

    pub fn perturb(&self) -> Self {
        let mut new_values = self.values.clone();

        let normal = Normal::new(0.0, 1E-5);
        for i in 0..new_values.len() {
            new_values[i] += normal.ind_sample(&mut rand::thread_rng()) as f32;
        }

        Guess { values: new_values }
    }

    pub fn create_initial_guess(&self, nwalkers: usize) -> Vec<Self> {
        (0..nwalkers).map(|_| self.perturb()).collect()
    }

    pub fn contains_infs(&self) -> bool {
        self.values.iter().any(|val| val.is_infinite())
    }

    pub fn contains_nans(&self) -> bool {
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
    q: Vec<Guess>,
    newlnprob: Vec<f32>,
    accept: Vec<bool>,
}

pub struct EnsembleSampler<'a, T: Prob + 'a> {
    nwalkers: usize,
    naccepted: usize,
    iterations: usize,
    lnprob: &'a T,
    dim: usize,
    last_run_mcmc_result: Option<i32>, // TODO: this is not i32
    rng: rand::ThreadRng,
}

impl<'a, T: Prob + 'a> EnsembleSampler<'a, T> {
    pub fn new(nwalkers: usize, dim: usize, lnprob: &'a T) -> Self {
        EnsembleSampler {
            nwalkers: nwalkers,
            iterations: 0,
            lnprob: lnprob,
            dim: dim,
            naccepted: 0,
            last_run_mcmc_result: None,
            rng: rand::thread_rng(),
        }
    }

    pub fn sample(&mut self, params: &[Guess], iterations: usize) -> Result<()> {
        let mut p = params.to_owned();
        let halfk = self.nwalkers / 2;
        let mut lnprob = self.get_lnprob(params)?;
        let indices: Vec<usize> = (0..params.len()).collect();

        for _ in 0..iterations {
            self.iterations += 1;
            let (first_half, second_half) = params.split_at(halfk);
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
                        lnprob[idx] = stretch.newlnprob[j];

                        p[idx].values = stretch.q[j].values.clone()
                    }
                }
            }
        }
        Ok(())
    }

    pub fn run_mcmc(&mut self, p0: &[Guess], N: usize) -> Result<()> {
        self.sample(p0, N)
    }

    pub fn reset(&mut self) {
        self.iterations = 0;
        self.naccepted = 0;
        self.last_run_mcmc_result = None;
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
            .map(|_| {
                     ((a - 1.0) * z_range.ind_sample(&mut self.rng)).powf(2.0f32) / 2.0f32
                 })
            .collect();


        let rint: Vec<usize> = (0..Ns).map(|_| rint_range.ind_sample(&mut self.rng)).collect();

        let mut q = Vec::new();
        for guess_i in 0..Ns {
            let mut values = Vec::with_capacity(self.dim);
            for param_i in 0..self.dim {
                let new_value = p1[rint[guess_i]].values[param_i] - zz[guess_i] * (p1[rint[guess_i]].values[param_i] - p0[guess_i].values[param_i]);
                values.push(new_value);
            }
            q.push(Guess { values });
        }
        assert_eq!(q.len(), zz.len());

        let mut out = Stretch::default();
        out.newlnprob = self.get_lnprob(&q).unwrap();
        out.q = q;
        out.accept.resize(out.newlnprob.len(), false);

        for i in 0..out.newlnprob.len() {
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
            let sum = self.x.iter()
                .zip(self.y)
                .fold(0.0f32, |acc, (x, y)| acc + (y - m * x + c).powf(2.0));
            -sum
        }
    }


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

    #[test]
    fn test_single_sample() {
        let (real_x, observed_y) = generate_dataset(20);
        let foo = LinearModel::new(&real_x, &observed_y);
        let p0 = create_guess();

        let nwalkers = 10;
        let mut sampler = EnsembleSampler::new(nwalkers, 2, &foo);

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
        let mut sampler = EnsembleSampler::new(nwalkers, 2, &foo);

        let params = p0.create_initial_guess(nwalkers);
        sampler.run_mcmc(&params, niters).unwrap();
    }

    // Test helper functions
    fn create_guess() -> Guess {
        Guess {
            values: vec![0.0f32, 0.0f32],
        }
    }

    fn generate_dataset(size: usize) -> (Vec<f32>, Vec<f32>) {
        let mut rng = rand::thread_rng();
        let x_range = Range::new(0f32, 10f32);
        let norm_range = Normal::new(0.0, 3.0);

        let mut real_x: Vec<f32> = (0..size)
            .map(|_| x_range.ind_sample(&mut rng))
            .collect();
        real_x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let real_y: Vec<f32> = real_x.iter().map(|x| REAL_M * x + REAL_C).collect();
        let observed_y: Vec<f32> = real_y
            .iter()
            .map(|y| y + norm_range.ind_sample(&mut rng) as f32)
            .collect();
        (real_x, observed_y)
    }
}
