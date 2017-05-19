use std::error::Error;

#[derive(Debug)]
pub enum EmceeError {
    Boxed(Box<Error>),
    InvalidInputs(&'static str),
}

pub type Result<T> = ::std::result::Result<T, EmceeError>;
