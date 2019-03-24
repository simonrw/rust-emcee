use rand::Rng;
use rand::distributions::{IndependentSample, Normal};

/// Represents an initial guess
///
/// This is the starting position for the sampling. All values are 64-bit floating point
/// numbers, and are contained in a [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html).
#[derive(Debug, Clone)]
pub struct Guess {
    /// A position in parameter space
    pub values: Vec<f64>,
}

impl ::std::ops::Index<usize> for Guess {
    type Output = f64;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.values[idx]
    }
}

impl ::std::ops::IndexMut<usize> for Guess {
    fn index_mut(&mut self, idx: usize) -> &mut f64 {
        &mut self.values[idx]
    }
}

impl Guess {
    /// Create a guess from a slice
    pub fn new(values: &[f64]) -> Self {
        Guess {
            values: Vec::from(values),
        }
    }

    /// Create a guess vector, perturbed from the starting position
    ///
    /// Use this to generate the starting guess for the sampling, where there is one guess
    /// slightly displaced from the staring location, per walker.
    pub fn create_initial_guess(&self, nwalkers: usize) -> Vec<Guess> {
        (0..nwalkers).map(|_| self.perturb()).collect()
    }

    /// Create a guess vector with custom random number generator
    ///
    /// For example, providing a random number generator that has been seeded causes re-creatable
    /// results. The random number generator must come from the [`rand`](https://docs.rs/rand)
    /// crate.
    pub fn create_initial_guess_with_rng<T: Rng>(
        &self,
        nwalkers: usize,
        mut rng: &mut T,
    ) -> Vec<Guess> {
        (0..nwalkers)
            .map(|_| self.perturb_with_rng(&mut rng))
            .collect()
    }

    /// Returns if the guess vector contains infinite values
    pub fn contains_infs(&self) -> bool {
        self.values.iter().any(|val| val.is_infinite())
    }

    /// Returns if the guess vector contains NaN values
    pub fn contains_nans(&self) -> bool {
        self.values.iter().any(|val| val.is_nan())
    }

    fn perturb(&self) -> Guess {
        let mut new_values = self.values.clone();

        let normal = Normal::new(0.0, 1E-5);
        for elem in &mut new_values {
            *elem += normal.ind_sample(&mut ::rand::thread_rng()) as f64;
        }

        Guess { values: new_values }
    }

    fn perturb_with_rng<T: Rng>(&self, mut rng: &mut T) -> Guess {
        let mut new_values = self.values.clone();

        let normal = Normal::new(0.0, 1E-5);
        for elem in &mut new_values {
            *elem += normal.ind_sample(&mut rng) as f64;
        }

        Guess { values: new_values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, StdRng};

    #[test]
    fn test_pertubation() {
        let guess = Guess::new(&[1.0f64, 2.0f64]);
        let mut rng = StdRng::from_seed(&[1, 2, 3, 4]);
        let perturbed = guess.perturb_with_rng(&mut rng);
        assert!(perturbed[0] != 1.0f64);
        assert!(perturbed[1] != 2.0f64);
    }

    #[test]
    fn test_initial_guess() {
        let guess = Guess::new(&[1.0f64, 2.0f64]);
        let initial = guess.create_initial_guess(10);
        assert_eq!(initial.len(), 10);
    }

    #[test]
    fn test_contains_infinites() {
        let guess = Guess::new(&[::std::f64::INFINITY]);
        assert!(guess.contains_infs());

        let guess = Guess::new(&[0f64]);
        assert!(!guess.contains_infs());
    }

    #[test]
    fn test_contains_nans() {
        let guess = Guess::new(&[::std::f64::NAN]);
        assert!(guess.contains_nans());

        let guess = Guess::new(&[0f64]);
        assert!(!guess.contains_nans());
    }

    #[test]
    fn test_indexing() {
        let mut guess = Guess::new(&[1., 2., 3., 4.]);
        assert_approx_eq!(guess[1], 2.0);

        guess[2] = 15.0;
        assert_eq!(guess[2], 15.0);
    }
}
