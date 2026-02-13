# Changelog
## 1.3.2
- Step-memory fallback in `slurm-bulk-user-stats` now ignores `.extern` steps and still prefers live `sstat` memory for running jobs when parent RSS is missing.

## 1.3.1
- `slurm-bulk-user-stats` now falls back to step `MaxRSS` for memory metrics when the parent job row has empty RSS, preventing `memUsed`/`memEff` from showing `n/a` for such finished jobs.

## 1.3.
- Added `--finished-limit N` to `pbs-bulk-user-stats` and `slurm-bulk-user-stats`. Show only first N finished jobs.
- Added `--finished-limit-strategy {post,fetch}` to `pbs-bulk-user-stats` and `slurm-bulk-user-stats`.
  Default `post` keeps previous behavior (fetch all then trim), while `fetch` is a best-effort faster mode (fetch active jobs plus up to N finished jobs).
- For running jobs where both `sacct` and `sstat` provide no usable CPU/RSS values, `slurm-bulk-user-stats` now prints `n/a` (instead of misleading `0.00%` / `0.00`).

## 1.2
- Added GPU request counts (NGPUS) to PBS and Slurm bulk user stats outputs and CSVs.
- Added README note for installing from GitHub with plotting extras.
- Added optional GPU monitoring to psutil-monitor via `--gpu` (NVML/pynvml), including terminal, plot and CSV outputs.
- Added pip extras `gpu` and `all` (plot + GPU) for simpler installs.

## 1.1
- PBS and Slurm support
- Added job state counts (running/pending/finished/other) to PBS and Slurm bulk user stats summaries.

## 1.0
- Initial release of `hpc-stats-scripts` with PBS/Slurm job summaries and psutil-based monitoring.


