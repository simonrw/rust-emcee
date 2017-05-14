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
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
