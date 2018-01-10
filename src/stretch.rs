use guess::Guess;

#[derive(Debug, Default)]
pub struct Stretch {
    pub q: Vec<Guess>,
    pub newlnprob: Vec<f64>,
    pub accept: Vec<bool>,
}

impl Stretch {
    pub fn preallocated_accept(size: usize) -> Stretch {
        let mut s = Stretch::default();
        s.accept.resize(size, false);
        s
    }
}
