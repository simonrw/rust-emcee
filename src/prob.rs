use guess::Guess;


pub trait Prob {
    fn lnlike(&self, params: &Guess) -> f32;
    fn lnprior(&self, params: &Guess) -> f32;

    fn lnprob(&self, params: &Guess) -> f32 {
        let lnp = self.lnprior(params);
        if lnp.is_finite() {
            lnp + self.lnlike(params)
        } else {
            -::std::f32::INFINITY
        }
    }
}
