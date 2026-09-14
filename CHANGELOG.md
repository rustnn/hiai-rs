# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- Initial workspace split into `hiai-rs-sys` (raw bindgen FFI) and `hiai-rs` (safe wrapper).
- Vendored pure-C CANN adapter (`adapter/`) over the HiAI DDK.
- `dispatch()` for running prebuilt offline models on Ascend NPUs.
- Borrowed `InputDesc`/`OutputDesc` tensor descriptors, with
  `OutputDesc.actual_len` reporting the produced byte count.
- `Session::prepare_io` and `IoCache` to pre-create and reuse DDK IO tensors
  across dispatches, reconciling only changed slots.
- `log`-based debug tracing for CANN timing, gated behind the `CANN_TIMING`
  environment variable.
