#![feature(test)]

extern crate emcee;
extern crate rand;
extern crate test;

#[cfg(test)]
mod benchmarks {
    use test::Bencher;
    use rand::distributions::{IndependentSample, Normal};
    use rand::{SeedableRng, StdRng};
    use emcee::{EnsembleSampler, Guess, Prob};

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
            let inv_prod = mat_vec_mul(self.icov, &values);
            -vec_vec_mul(&values, &inv_prod) / 2.0
        }
    }

    #[bench]
    fn benchmark_sample(b: &mut Bencher) {
        let nwalkers = 250;
        let ndim = 5;

        let icov = [
            [
                318.92634269,
                531.39511426,
                -136.10315845,
                154.17685545,
                552.308813,
            ],
            [
                531.39511426,
                899.91793286,
                -224.74333441,
                258.98686842,
                938.32014715,
            ],
            [
                -136.10315845,
                -224.74333441,
                60.61145495,
                -66.68898448,
                -232.52035701,
            ],
            [
                154.17685545,
                258.98686842,
                -66.68898448,
                83.9979827,
                266.44429402,
            ],
            [
                552.308813,
                938.32014715,
                -232.52035701,
                266.44429402,
                983.33032073,
            ],
        ];
        let model = MultivariateProb { icov: &icov };

        let norm_range = Normal::new(0.0f64, 1.0f64);
        let mut rng = StdRng::from_seed(&[1, 2, 3, 4]);
        let p0: Vec<_> = (0..nwalkers)
            .map(|_| Guess {
                values: (0..ndim)
                    .map(|_| 0.1f64 * norm_range.ind_sample(&mut rng) as f64)
                    .collect(),
            })
            .collect();

        let mut sampler = EnsembleSampler::new(nwalkers, ndim, &model).unwrap();
        sampler.seed(&[1]);

        // Now run the benchmark
        b.iter(|| sampler.sample(&p0, 1, |_| {}).unwrap());
    }

    fn mat_vec_mul(m: &[[f64; 5]; 5], v: &[f64; 5]) -> [f64; 5] {
        let mut out = [0.0f64; 5];

        for i in 0..5 {
            out[i] =
                v[0] * m[i][0] + v[1] * m[i][1] + v[2] * m[i][2] + v[3] * m[i][3] + v[4] * m[i][4];
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

}
