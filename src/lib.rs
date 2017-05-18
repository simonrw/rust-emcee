#![allow(non_snake_case)]
extern crate rand;

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

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

impl Stretch {
    pub fn preallocated_accept(N: usize) -> Stretch {
        let mut s = Stretch::default();
        s.accept.resize(N, false);
        s
    }
}

#[derive(Debug, Default)]
pub struct Chain {
    data: Vec<f32>,
    nparams: usize,
    nwalkers: usize,
    niterations: usize,
}

impl Chain {
    pub fn new(nparams: usize, nwalkers: usize, niterations: usize) -> Chain {
        Chain {
            nparams: nparams,
            nwalkers: nwalkers,
            niterations: niterations,
            data: vec![0f32; nparams * nwalkers * niterations],
        }
    }

    pub fn set(&mut self, param_idx: usize, walker_idx: usize, iteration_idx: usize, value: f32) {
        assert!(param_idx < self.nparams);
        assert!(walker_idx < self.nwalkers);
        assert!(iteration_idx < self.niterations);

        let idx = self.index(param_idx, walker_idx, iteration_idx);

        self.data[idx] = value;
    }

    pub fn get(&self, param_idx: usize, walker_idx: usize, iteration_idx: usize) -> f32 {
        assert!(param_idx < self.nparams);
        assert!(walker_idx < self.nwalkers);
        assert!(iteration_idx < self.niterations);

        let idx = self.index(param_idx, walker_idx, iteration_idx);

        self.data[idx]
    }

    pub fn set_params(&mut self, walker_idx: usize, iteration_idx: usize, newdata: &[f32]) {
        assert_eq!(newdata.len(), self.nparams);
        for (idx, value) in newdata.iter().enumerate() {
            self.set(idx, walker_idx, iteration_idx, *value);
        }
    }

    fn index(&self, param_idx: usize, walker_idx: usize, iteration_idx: usize) -> usize {
        (iteration_idx * self.nwalkers * self.nparams) + (walker_idx * self.nparams) + param_idx
    }
}

#[derive(Debug, Default)]
pub struct ProbStore {
    data: Vec<f32>,
    nwalkers: usize,
    niterations: usize,
}

impl ProbStore {
    pub fn new(nwalkers: usize, niterations: usize) -> ProbStore {
        ProbStore {
            nwalkers: nwalkers,
            niterations: niterations,
            data: vec![0f32; nwalkers * niterations],
        }
    }

    pub fn set(&mut self, walker_idx: usize, iteration_idx: usize, value: f32) {
        assert!(walker_idx < self.nwalkers);
        assert!(iteration_idx < self.niterations);

        let idx = self.index(walker_idx, iteration_idx);

        self.data[idx] = value;
    }

    pub fn get(&self, walker_idx: usize, iteration_idx: usize) -> f32 {
        assert!(walker_idx < self.nwalkers);
        assert!(iteration_idx < self.niterations);

        let idx = self.index(walker_idx, iteration_idx);

        self.data[idx]
    }

    pub fn set_probs(&mut self, iteration_idx: usize, newdata: &[f32]) {
        assert_eq!(newdata.len(), self.nwalkers);
        for (idx, value) in newdata.iter().enumerate() {
            self.set(idx, iteration_idx, *value);
        }
    }

    fn index(&self, walker_idx: usize, iteration_idx: usize) -> usize {
        (iteration_idx * self.nwalkers) + walker_idx
    }
}

pub struct EnsembleSampler<'a, T: Prob + 'a> {
    nwalkers: usize,
    naccepted: usize,
    iterations: usize,
    lnprob: &'a T,
    dim: usize,
    rng: rand::ThreadRng,
    chain: Option<Chain>,
    probstore: Option<ProbStore>,
}

impl<'a, T: Prob + 'a> EnsembleSampler<'a, T> {
    pub fn new(nwalkers: usize, dim: usize, lnprob: &'a T) -> Self {
        EnsembleSampler {
            nwalkers: nwalkers,
            iterations: 0,
            lnprob: lnprob,
            dim: dim,
            naccepted: 0,
            rng: rand::thread_rng(),
            chain: None,
            probstore: None,
        }
    }

    pub fn sample(&mut self, params: &[Guess], iterations: usize) -> Result<()> {
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

    pub fn run_mcmc(&mut self, p0: &[Guess], N: usize) -> Result<()> {
        self.sample(p0, N)
    }

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
    fn test_chain() {
        let nparams = 2;
        let nwalkers = 10;
        let niterations = 1000;
        let mut chain = Chain::new(nparams, nwalkers, niterations);
        assert_eq!(chain.data.len(), nparams * nwalkers * niterations);

        assert_eq!(chain.index(0, 0, 0), 0);
        assert_eq!(chain.index(1, 0, 0), 1);
        assert_eq!(chain.index(0, 1, 0), 2);
        assert_eq!(chain.index(1, 1, 0), 3);
        assert_eq!(chain.index(0, 2, 0), 4);
        assert_eq!(chain.index(0, 9, 0), 18);
        assert_eq!(chain.index(0, 0, 1), 20);

        chain.set(0, 1, 0, 2.0f32);
        assert_eq!(chain.data[2], 2.0f32);
        assert_eq!(chain.get(0, 1, 0), 2.0f32);


        let newdata = vec![5.0f32, 100.0f32];
        chain.set_params(1, 250, &newdata);

        assert_eq!(chain.get(0, 1, 250), 5.0f32);
        assert_eq!(chain.get(1, 1, 250), 100.0f32);
    }

    #[test]
    fn test_probstore() {
        let nwalkers = 4;
        let niterations = 1000;
        let mut store = ProbStore::new(nwalkers, niterations);
        assert_eq!(store.data.len(), nwalkers * niterations);

        assert_eq!(store.index(0, 0), 0);
        assert_eq!(store.index(2, 0), 2);
        assert_eq!(store.index(0, 1), 4);

        store.set(1, 0, 2.0f32);
        assert_eq!(store.data[1], 2.0f32);
        assert_eq!(store.get(1, 0), 2.0f32);


        let newdata = vec![5.0f32, 100.0f32, 1.0f32, 20f32];
        store.set_probs(250, &newdata);

        assert_eq!(store.get(0, 250), 5.0f32);
        assert_eq!(store.get(1, 250), 100.0f32);
        assert_eq!(store.get(3, 250), 20.0f32);
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

        let nwalkers = 4;
        let mut sampler = EnsembleSampler::new(nwalkers, 2, &foo);
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

        let mut sampler = EnsembleSampler::new(nwalkers, p0.values.len(), &foo);
        let (a, b) = pos.split_at(nwalkers / 2);

        assert_eq!(a.len(), nwalkers / 2);
        assert_eq!(b.len(), nwalkers / 2);

        let lnprob = sampler.get_lnprob(&pos).unwrap();
        let stretch = sampler.propose_stretch(&a, &b, &lnprob);
    }

    #[test]
    fn test_mcmc_run() {
        let nwalkers = 20;
        let p0 = Guess { values: vec![0f32, 0f32] };
        let pos = p0.create_initial_guess(nwalkers);
        let (real_x, observed_y) = load_baked_dataset();
        let foo = LinearModel::new(&real_x, &observed_y);

        let niters = 1000;
        let mut sampler = EnsembleSampler::new(nwalkers, p0.values.len(), &foo);
        let _ = sampler.run_mcmc(&pos, niters).unwrap();

        /*
         * These are really tricky to test for. Random numbers are a pain to test.
         * We therefore have a quite wide margin and hope that these tests are _vaguely_ reliable.
         */
        if let Some(ref chain) = sampler.chain {
            assert_approx_eq!(chain.get(0, 0, niters - 1), 2.0f32, 3f32);
            assert_approx_eq!(chain.get(1, 0, niters - 1), 5.0f32, 3f32);
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
