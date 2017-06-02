# emcee

[![Build Status](https://travis-ci.org/mindriot101/rust-emcee.svg?branch=master)](https://travis-ci.org/mindriot101/rust-emcee)
[![Crates version](https://img.shields.io/crates/v/emcee.svg)](https://crates.io/crates/emcee)
[![Docs](https://img.shields.io/badge/docs-emcee-brightgreen.svg)](https://docs.rs/emcee)

A re-implementation of [Dan Foreman-Mackey's][dfm] [emcee][emcee] in Rust.

See the [hosted documentation here][docs]

The [`fitting_a_model_to_data` example][fitting-model-to-data] is a re-creation of the ["fitting a model to
data"][fitting-model-to-data-python] example from the `emcee` documentation.

## Attribution

If you make use of emcee in your work, please cite Dan's paper ([arXiv](http://arxiv.org/abs/1202.3665), [ADS](http://adsabs.harvard.edu/abs/2013PASP..125..306F), [BibTeX](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2013PASP..125..306F&data_type=BIBTEX)).

A copy of the original MIT license is given under [DFM-LICENSE][dfm-license].

## Basic usage

### Implementing models

The sampler requires a struct that implements [`emcee::Prob`][emcee-prob], for example:

```rust
use emcee::{Guess, Prob};

struct Model;

impl Prob for Model {
    fn lnlike(&self, params: &Guess) -> f32 {
        // Insert actual implementation here
        0f32
    }

    fn lnprior(&self, params: &Guess) -> f32 {
        // Insert actual implementation here
        0f32
    }
}
```

The trait has a default implementation for [`lnprob`][emcee-lnprob] which computes the product
of the likelihood and prior probability (sum in log space) as per Bayes' rule.  Invalid prior
values are marked by returning -[`std::f32::INFINITY`][std-infinity] from the priors function.
Note your implementation is likely to need external data. This data should be included with
your `Model` class, for example:

```rust
struct Model<'a> {
    x: &'a [f32],
    y: &'a [f32],
}

// Linear model y = m * x + c
impl<'a> Prob for Model<'a> {
    fn lnlike(&self, params: &Guess) -> f32 {
        let m = params[0];
        let c = params[1];

        -0.5 * self.x.iter().zip(self.y)
            .map(|(xval, yval)| {
                let model = m * xval + c;
                let residual = (yval - model).powf(2.0);
                residual
            }).sum::<f32>()
    }

    fn lnprior(&self, params: &Guess) -> f32 {
        // unimformative priors
        0.0f32
    }
}

```

### Initial guess

Next, construct an initial guess. A [`Guess`][emcee-guess] represents a proposal parameter
vector:

```rust
use emcee::Guess;

let initial_guess = Guess::new(&[0.0f32, 0.0f32]);
```

The sampler implemented by this create uses multiple *walkers*, and as such the initial
guess must be replicated once per walker, and typically dispersed from the initial position
to aid exploration of the problem parameter space. This can be achieved with the
[`create_initial_guess`][emcee-create-initial-guess] method:

```rust
let nwalkers = 100;
let perturbed_guess = initial_guess.create_initial_guess(nwalkers);
assert_eq!(perturbed_guess.len(), nwalkers);
```

### Constructing a sampler

The sampler generates new parameter vectors, assess the probability using a user-supplied
probability model, accepts more likely parameter vectors and iterates for a number of
iterations.

The sampler needs to know the number of walkers to use, which must be an even number
and at least twice the size of your parameter vector. It also needs the size of your
parameter vector, and your probability struct (which implements [`Prob`][emcee-prob]):

```rust
let nwalkers = 100;
let ndim = 2;  // m and c

// Build a linear model y = m * x + c (see above)

let initial_x = [0.0f32, 1.0f32, 2.0f32];
let initial_y = [5.0f32, 7.0f32, 9.0f32];

let model = Model {
    x: &initial_x,
    y: &initial_y,
};

let mut sampler = emcee::EnsembleSampler::new(nwalkers, ndim, &model)
    .expect("could not create sampler");
```

Then run the sampler:

```rust
let niterations = 100;
sampler.run_mcmc(&perturbed_guess, niterations).expect("error running sampler");
```

#### Iterative sampling

It is sometimes useful to get the internal values proposed and evaluated
during each proposal step of the sampler. In the Python version, the
method `sample` is a generator which can be iterated over to evaluate
the sample steps.

In this Rust version, we provide this feature by exposing the
[`sample`][emcee-sample] method, which takes a callback, which is called
once per iteration with a single [`Step`][emcee-step] object. For
example:

```rust
sampler.sample(&perturbed_guess, niterations, |step| {
    println!("Current iteration: {}", step.iteration);
    println!("Current guess vectors: {:?}", step.pos);
    println!("Current log posterior probabilities: {:?}", step.lnprob);
});
```

### Studying the results

The samples are stored in the sampler's `flatchain` which is constructed through the
[`flatchain`][emcee-flatchain] method on the sampler:

```rust
let flatchain = sampler.flatchain().unwrap();

for (i, guess) in flatchain.iter().enumerate() {
    // Skip possible "burn-in" phase
    if i < 50 * nwalkers {
        continue;
    }

    println!("Iteration {}; m={}, c={}", i, guess[0], guess[1]);
}
```

[emcee]: http://dan.iel.fm/emcee/current/
[emcee-prob]: https://docs.rs/emcee/0.3.0/emcee/trait.Prob.html
[emcee-guess]: https://docs.rs/emcee/0.3.0/emcee/struct.Guess.html
[emcee-lnprob]: https://docs.rs/emcee/0.3.0/emcee/trait.Prob.html#method.lnprob
[std-infinity]: https://doc.rust-lang.org/std/f32/constant.INFINITY.html
[emcee-create-initial-guess]: https://docs.rs/emcee/0.3.0/emcee/struct.Guess.html#method.create_initial_guess
[emcee-flatchain]: https://docs.rs/emcee/0.3.0/emcee/struct.EnsembleSampler.html#method.flatchain
[docs]: https://docs.rs/emcee
[fitting-model-to-data]: https://github.com/mindriot101/rust-emcee/blob/master/examples/fitting_a_model_to_data.rs
[fitting-model-to-data-python]: http://dan.iel.fm/emcee/current/user/line/
[dfm]: http://dan.iel.fm/
[dfm-license]: https://github.com/mindriot101/rust-emcee/blob/master/DFM-LICENSE
[emcee-sample]: https://docs.rs/emcee/0.3.0/emcee/struct.EnsembleSampler.html#method.sample
[emcee-step]: https://docs.rs/emcee/0.3.0/emcee/struct.Step.html
