/*
 * This example re-creates the "Fitting a Model to Data" example from the emcee documentation:
 *
 * http://dan.iel.fm/emcee/current/user/line/#example-fitting-a-model-to-data
 */

extern crate emcee;
extern crate rand;

use std::fs::File;
use std::io::{BufWriter, Write};
use rand::distributions::{Distribution, Normal, Uniform};
use rand::{SeedableRng, StdRng};

use emcee::{Guess, Prob};

fn sort(data: &mut Vec<f64>) {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
}

fn compute_quantiles(chain: &[Guess]) -> Vec<[f64; 3]> {
    let nparams = chain[0].values.len();
    let niterations = chain.len();
    let mut param_vecs: Vec<Vec<f64>> = vec![Vec::with_capacity(chain.len()); nparams];
    for guess in chain {
        for (param, value) in guess.values.iter().enumerate() {
            param_vecs[param].push(*value);
        }
    }

    let mut out = Vec::with_capacity(nparams);
    let lower_idx = (0.16 * niterations as f64) as usize;
    let med_idx = (0.5 * niterations as f64) as usize;
    let upper_idx = (0.84 * niterations as f64) as usize;

    for mut v in &mut param_vecs {
        sort(&mut v);

        let med = v[med_idx];
        let lower = v[lower_idx];
        let upper = v[upper_idx];
        let res = [lower, med, upper];
        out.push(res);
    }
    out
}

fn main() {
    /* Pre-generate rng and distributions */
    let mut rng = StdRng::seed_from_u64(42);
    let unit_range = Uniform::new(0f64, 1f64);
    let norm_gen = Normal::new(0.0, 1.0);

    // Choose the "true" parameters.
    let m_true = -0.9594f64;
    let b_true = 4.294f64;
    let f_true = 0.534f64;

    // Generate some synthetic data from the model.
    let npoints = 50usize;
    let x = {
        let mut unsorted: Vec<_> = (0..npoints)
            .map(|_| 10f64 * unit_range.sample(&mut rng))
            .collect();
        unsorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unsorted
    };
    let mut y = Vec::with_capacity(npoints);
    let mut yerr = Vec::with_capacity(npoints);

    for xval in &x {
        let yerr_val = 0.1 + 0.5 * unit_range.sample(&mut rng);
        let mut y_val = m_true * xval + b_true;
        y_val += (f_true * y_val).abs() * norm_gen.sample(&mut rng) as f64;
        y_val += yerr_val * norm_gen.sample(&mut rng) as f64;

        y.push(y_val);
        yerr.push(yerr_val);
    }

    /*
     * Shortcut the least squares minimisation by starting the sampling
     * from the values found in the documentation
     */
    let guess = Guess::new(&[-1.003, 4.528, 0.454f64.ln()]);

    /*
     * Define the equivalent of lnprior, lnlike and lnprob (note: lnprob is automatically
     * derived for you by the `Prob` trait, unless custom behaviour is required.
     */
    struct LinearWithUnderestimatedErrors<'a> {
        x: &'a [f64],
        y: &'a [f64],
        e: &'a [f64],
    };

    impl<'a> Prob for LinearWithUnderestimatedErrors<'a> {
        fn lnlike(&self, theta: &Guess) -> f64 {
            assert_eq!(theta.values.len(), 3);
            assert_eq!(self.x.len(), self.y.len());
            assert_eq!(self.y.len(), self.e.len());

            let m = theta[0];
            let b = theta[1];
            let lnf = theta[2];

            let mut result = 0.;
            for i in 0..self.x.len() {
                let model = m * self.x[i] + b;
                let inv_sigma2 = 1.0 / (self.e[i].powf(2.0) + model.powf(2.0) * (2.0 * lnf).exp());
                result += (self.y[i] - model).powf(2.) * inv_sigma2 - inv_sigma2.ln();
            }
            -0.5 * result
        }

        fn lnprior(&self, theta: &Guess) -> f64 {
            assert_eq!(theta.values.len(), 3);

            let m = theta[0];
            let b = theta[1];
            let lnf = theta[2];

            if (m > -5.0) && (m < 5.0) && (b > 0.0) && (b < 10.0) && (lnf > -10.0) && (lnf < 1.0) {
                0.
            } else {
                -std::f64::INFINITY
            }
        }
    }

    let model = LinearWithUnderestimatedErrors {
        x: &x,
        y: &y,
        e: &yerr,
    };

    /*
     * Now let's get to the MCMC sampling
     */
    let ndim = 3;
    let nwalkers = 100;
    let pos = guess.create_initial_guess_with_rng(nwalkers, &mut rng);

    let mut sampler =
        emcee::EnsembleSampler::new(nwalkers, ndim, &model).expect("creating sampler");
    sampler.seed(42);
    sampler.run_mcmc(&pos, 500).unwrap();

    let flatchain = sampler.flatchain().unwrap();

    let file = File::create("/tmp/emcee-results.txt").expect("opening output file");
    let mut writer = BufWriter::new(&file);

    for (i, guess) in flatchain.iter().enumerate() {
        if i < 50 * nwalkers {
            continue;
        }

        write!(&mut writer, "{} {} {}\n", guess[0], guess[1], guess[2])
            .expect("writing output line");
    }

    let marginalised_posteriors = compute_quantiles(&flatchain);

    print_marginalised("m", &marginalised_posteriors[0], m_true);
    print_marginalised("b", &marginalised_posteriors[1], b_true);
    print_marginalised("lnf", &marginalised_posteriors[2], f_true.ln());
}

fn print_marginalised(name: &str, values: &[f64], truth: f64) {
    println!(
        "{:3} = {:6.3} +{:.3} -{:.3} (truth: {:6.3})",
        name,
        values[1],
        values[1] - values[0],
        values[2] - values[1],
        truth
    );
}
