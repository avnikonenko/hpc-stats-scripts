# Changelog

## 1.0
- Initial release of `hpc-stats-scripts` with PBS/Slurm job summaries and psutil-based monitoring.

## 1.1
- PBS and Slurm support
- Added job state counts (running/pending/finished/other) to PBS and Slurm bulk user stats summaries.

## 1.2
- Added GPU request counts (NGPUS) to PBS and Slurm bulk user stats outputs and CSVs.
- Added README note for installing from GitHub with plotting extras.
- Added optional GPU monitoring to psutil-monitor via `--gpu` (NVML/pynvml), including terminal, plot and CSV outputs.
- Added pip extras `gpu` and `all` (plot + GPU) for simpler installs.

## the latest version from github**
- Added `--finished-limit N` to `pbs-bulk-user-stats` and `slurm-bulk-user-stats`. Show only first N finished jobs.
- Added `--finished-limit-strategy {post,fetch}` to `pbs-bulk-user-stats` and `slurm-bulk-user-stats`.
  Default `post` keeps previous behavior (fetch all then trim), while `fetch` is a best-effort faster mode (fetch active jobs plus up to N finished jobs).
- Improved `slurm-bulk-user-stats` CPU efficiency reporting by falling back to step-row CPU time and additional Slurm accounting fields (`UserCPU`/`SystemCPU`, `TRESUsageInTot`) when parent `TotalCPU` is missing or zero.
- Added a live fallback for running Slurm jobs: when `sacct` CPU/RSS fields are empty, `slurm-bulk-user-stats` now tries `sstat` (`.batch` first) to fill `CPUT`, `avgCPU`, `CPUeff`, and `memUsed`.
