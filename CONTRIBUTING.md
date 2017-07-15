# Contributing

Thank you for wanting to improve `emcee`. The most important thing for
this project is for this code to be used and trusted by users.

Contributions to `emcee` are always weldome, in any form. In particular,
usage examples, tests and improvements to the API are most encouraged.
In addition, bugfixes to the algorithm itself are appreciated.

At the moment, the project is small, with only [one developer][profile], so the
best way to get attention, or to ask questions is through github issues.
Just [file a new issue][new-issue] and you should hear back from me
within a few days - typically a few hours.

## Vision

I would like to see the library stick quite closely to the original
[`python` version][emcee]. If significant improvements are designed and
thought out for this library, I'd like to see a conversation with @dfm
about how to advance the implementations _together_.

## Development

Please make sure all tests pass before submitting pull requests, but
extra tests are always welcome.

As the library has very few external dependencies, development is quite
easy:

```sh
git clone https://github.com/mindriot101/rust-emcee.git
cd rust-emcee
cargo test
# start developing
```

All code is automatically formatted before each commit (with a git
pre-commit hook), and tested before each push and I try to keep all
commits passing the tests. If this is not the case, I'd ask you once
your new feature/PR is ready, to go back and edit your git history
accordingly.

I'd also like to see the code compile without warnings, and as such (due
to the `#![warn(missing_docs)]` line at the top of `src/lib.rs`) all
public methods should be documented.

[emcee]: http://dan.iel.fm/emcee/current/
[new-issue]: https://github.com/mindriot101/rust-emcee/issues/new 
[profile]: https://github.com/mindriot101
