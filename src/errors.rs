//! Errors

/// General error class for any errors possible from emcee
#[derive(Debug)]
pub enum EmceeError {
    /// Encapsulates if invalid parameters are given when trying to create an EnsembleSampler
    InvalidInputs(String),

    /// General message type for ad-hoc messages
    Msg(String),
}

impl ::std::fmt::Display for EmceeError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Emcee Error")
    }
}

impl<'a> ::std::convert::From<&'a str> for EmceeError {
    fn from(msg: &'a str) -> EmceeError {
        EmceeError::Msg(msg.to_string())
    }
}

/// Result alias which wraps [`EmceeError`][emcee-error]
///
/// [emcee-error]: https://example.com
pub type Result<T> = ::std::result::Result<T, EmceeError>;

impl ::std::error::Error for EmceeError {
    fn description(&self) -> &str {
        use EmceeError::*;

        let details = match *self {
            InvalidInputs(ref msg) | Msg(ref msg) => msg,
        };

        details.as_str()
    }

    fn cause(&self) -> Option<&::std::error::Error> {
        // We are not wrapping other error types, and our types do not have an
        // underlying cause beyond the description passed via the creation
        None
    }
}
