# NOTICE

This library is the standalone extraction of the spectral-telemetry components
of the paper repository
[optimizer-scaling-laws/spectral-scaling-laws](https://github.com/optimizer-scaling-laws/spectral-scaling-laws)
(MIT), companion to *Same Architecture, Different Capacity: Optimizer-Induced
Spectral Scaling Laws* (arXiv:2605.21803).

Ported (with API generalization) from upstream, preserving numerical
conventions so results are directly comparable:

- `core/ranks.py`      <- `optimizer_ssl/spectra/{covariance,effective_rank}.py`
- `core/fits.py`       <- `optimizer_ssl/analysis/scaling_fits.py`
- `core/schema.py`     <- `optimizer_ssl/analysis/log_schema.py`
- `core/frequency.py`  <- `optimizer_ssl/spectra/frequency_metrics.py`
- `torch_backend/*`    <- generalizes `optimizer_ssl/{probe.py,spectra/tracker.py}`

New in this library (not in upstream): the O(D^2) streaming covariance
accumulator, mergeable across chunks / buckets / ranks; predicate-based module
probes; tensor-parallel activation guards as a first-class API.
