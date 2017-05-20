error_chain! {
    errors {
        InvalidInputs(t: &'static str) {
            description("invalid inputs")
            display("invalid inputs: '{}'", t)
        }
    }
}
