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
- Improved `slurm-bulk-user-stats` CPU efficiency reporting by using step-row CPU time (for example `.batch`) when parent job `TotalCPU` is missing or zero.
