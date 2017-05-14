extern crate rand;

pub trait Prob {
    fn lnlike(&self, params: &[f32]) -> f32;
    fn lnprior(&self, params: &[f32]) -> f32;

    fn lnprob(&self, params: &[f32]) -> f32 {
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
    rng: rand::ThreadRng,
    last_run_mcmc_result: Option<i32>,  // TODO: this is not i32
}

impl EnsembleSampler {
    pub fn new(nwalkers: usize, dim: usize, lnprob: Box<Prob>) -> Self {
        EnsembleSampler {
            nwalkers: nwalkers,
            iterations: 0,
            lnprob: lnprob,
            dim: dim,
            rng: rand::thread_rng(),
            naccepted: 0,
            last_run_mcmc_result: None,
        }
    }

    pub fn sample(&mut self, params: &Vec<f32>, p0: &Vec<Vec<f32>>, iterations: usize) {
        let halfk = self.nwalkers / 2;
        for i in 0..iterations {}
    }

    pub fn run_mcmc(&mut self, p0: &Vec<Vec<f32>>, N: usize) {}

    pub fn reset(&mut self) {
        self.iterations = 0;
        self.naccepted = 0;
        self.last_run_mcmc_result = None;
    }

    // Internal functions

    fn get_lnprob(&mut self, p: &[f32]) {
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
