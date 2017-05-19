use rand::Rng;
use rand::distributions::{Normal, IndependentSample};

#[derive(Debug, Clone)]
pub struct Guess {
    pub values: Vec<f32>,
}

impl Guess {
    pub fn new(values: &[f32]) -> Self {
        Guess { values: Vec::from(values) }
    }

    pub fn perturb(&self) -> Guess {
        let mut new_values = self.values.clone();

        let normal = Normal::new(0.0, 1E-5);
        for elem in &mut new_values {
            *elem += normal.ind_sample(&mut ::rand::thread_rng()) as f32;
        }

        Guess { values: new_values }
    }

    pub fn perturb_with_rng<T: Rng>(&self, mut rng: &mut T) -> Guess {
        let mut new_values = self.values.clone();

        let normal = Normal::new(0.0, 1E-5);
        for elem in &mut new_values {
            *elem += normal.ind_sample(&mut rng) as f32;
        }

        Guess { values: new_values }
    }

    pub fn create_initial_guess(&self, nwalkers: usize) -> Vec<Guess> {
        (0..nwalkers).map(|_| self.perturb()).collect()
    }

    pub fn create_initial_guess_with_rng<T: Rng>(&self,
                                                 nwalkers: usize,
                                                 mut rng: &mut T)
                                                 -> Vec<Guess> {
        (0..nwalkers)
            .map(|_| self.perturb_with_rng(&mut rng))
            .collect()
    }

    pub fn contains_infs(&self) -> bool {
        self.values.iter().any(|val| val.is_infinite())
    }

    pub fn contains_nans(&self) -> bool {
        self.values.iter().any(|val| val.is_nan())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{StdRng, SeedableRng};

    #[test]
    fn test_pertubation() {
        let guess = Guess::new(&[1.0f32, 2.0f32]);
        let mut rng = StdRng::from_seed(&[1, 2, 3, 4]);
        let perturbed = guess.perturb_with_rng(&mut rng);
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
        let guess = Guess::new(&[::std::f32::INFINITY]);
        assert!(guess.contains_infs());

        let guess = Guess::new(&[0f32]);
        assert!(!guess.contains_infs());
    }

    #[test]
    fn test_contains_nans() {
        let guess = Guess::new(&[::std::f32::NAN]);
        assert!(guess.contains_nans());

        let guess = Guess::new(&[0f32]);
        assert!(!guess.contains_nans());
    }
}
