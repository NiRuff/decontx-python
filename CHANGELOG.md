# Changelog

## 0.3.0

### Changed
- Reported contamination now equals R decontX: the E-step contamination proportion
  instead of 1 - theta. Estimates are slightly lower, mainly with a fixed `delta`
  or for cells with few UMIs. Corrected counts are unchanged.
- Faster EM: fused, parallel sparse kernel and convergence checked every iteration
  as in R (PR #5, Stephen Williams, 10x Genomics). Results within ~0.001 of 0.2.0
  apart from the change above.

### Fixed
- Out-of-bounds writes when a batch lacks a cluster label; labels are now relabelled
  per batch as in R.
- Log-likelihood is recorded independent of `verbose`.

### Added
- Validation tests (simulated ground truth, batch edge cases, golden regression,
  thread invariance) and CI on Linux and Windows.
- LICENSE file (MIT, with attribution to celda).
