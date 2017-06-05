use guess::Guess;


/// Encapsulate the model evaluation
///
/// The user must implement [`lnlike`](trait.Prob.html#tymethod.lnlike) and
/// [`lnprior`](trait.Prob.html#tymethod.lnprior) which correspond to the respective terms in
/// Bayes' equation. They must return the log probability which is then combined in the
/// [`lnprob`](trait.Prob.html#method.lnprob) function which has a default implementation.
///
/// Typically, access to any dataset required for fitting is handled through the struct which
/// implements this trait. By convention, invalid prior results are signaled by returning
/// [`-std::f64::INFINITY`](https://doc.rust-lang.org/std/f64/constant.INFINITY.html).
///
/// An example for a simple linear model to data with uncertainties:
///
/// ```rust
/// # use emcee::{Prob, Guess};
/// // lifetimes and slices used for efficiency reasons
/// struct Foo<'a> {
///     x: &'a [f64],
///     y: &'a [f64],
///     e: &'a [f64], // uncertainties
/// }
///
/// impl<'a> Prob for Foo<'a> {
///     fn lnlike(&self, params: &Guess) -> f64 {
///         let m = params[0];
///         let c = params[1];
///
///         let mut result = 0.;
///         for i in 9..self.x.len() {
///             let model = m * self.x[i] + c;
///             result += ((self.y[i] - model) / self.e[i]).powf(2.);
///         }
///         -0.5 * result
///
///     }
///
///     fn lnprior(&self, params: &Guess) -> f64 {
///         let m = params[0];
///         let c = params[1];
///
///         if (m > -5.) && (m < 5.) && (c > -10.) && (c < 10.) {
///             0.0
///         } else {
///             -std::f64::INFINITY
///         }
///     }
/// }
/// ```
///
/// The default implementation of
/// [`lnprob`](trait.Prob.html#method.lnprob) can be seen in the source code.
pub trait Prob {
    /// Computes the natural logarithm of the likelihood of a position in parameter space
    fn lnlike(&self, params: &Guess) -> f64;

    /// Computes the natural logarithm of the prior probability of a position in parameter space
    fn lnprior(&self, params: &Guess) -> f64;

    /// Computes the natural logarithm of the log posterior probabilities
    fn lnprob(&self, params: &Guess) -> f64 {
        let lnp = self.lnprior(params);
        if lnp.is_finite() {
            lnp + self.lnlike(params)
        } else {
            -::std::f64::INFINITY
        }
    }
}
