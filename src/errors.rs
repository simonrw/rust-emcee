use std::error::Error;

#[derive(Debug)]
pub enum EmceeError {
    Boxed(Box<Error>),
}

pub type Result<T> = ::std::result::Result<T, EmceeError>;
