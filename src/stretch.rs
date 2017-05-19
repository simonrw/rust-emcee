use guess::Guess;


#[derive(Debug, Default)]
pub struct Stretch {
    pub q: Vec<Guess>,
    pub newlnprob: Vec<f32>,
    pub accept: Vec<bool>,
}

impl Stretch {
    pub fn preallocated_accept(N: usize) -> Stretch {
        let mut s = Stretch::default();
        s.accept.resize(N, false);
        s
    }
}
