#!/usr/bin/env python3
"""Summarize Slurm job CPU and memory usage for a user or a single job."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


# ---------- Utilities ----------
def parse_slurm_time_to_seconds(val: str) -> Optional[int]:
    """Parse Slurm time strings like 1-02:03:04 or 02:03:04 to seconds."""
    if not val:
        return None
    val = val.strip()
    if val.lower() in {"unknown", "invalid"}:
        return None
    if val.isdigit():
        return int(val)
    days = 0
    if "-" in val:
        day_part, rest = val.split("-", 1)
        try:
            days = int(day_part)
            val = rest
        except ValueError:
            pass
    parts = val.split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    if len(parts) < 3:
        return None
    try:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    total_seconds = int(days * 86400 + hours * 3600 + minutes * 60 + seconds)
    return total_seconds


def parse_size_to_bytes(s: str) -> Optional[int]:
    """Parse memory strings like '4G', '4000M', '1.5T' to bytes (IEC 1024)."""
    if not s:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtTpP]?)", s.strip())
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    table = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    mult = table.get(unit)
    return int(val * mult) if mult else None


def parse_reqmem_to_bytes(mem: str, ncpus: Optional[int], nnodes: Optional[int]) -> Optional[int]:
    """Parse Slurm ReqMem strings (e.g., 4Gn, 4000Mc) into total bytes."""
    if not mem:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtTpP]?)([nc]?)\s*$", mem)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    scope = (m.group(3) or "").lower()  # c=per CPU, n=per node
    base = parse_size_to_bytes(f"{val}{unit}")
    if base is None:
        return None
    total = base
    if scope == "c" and ncpus:
        total = base * ncpus
    elif scope == "n" and nnodes:
        total = base * nnodes
    return int(total)


def fmt_bytes(n: Optional[int]) -> str:
    """Human-readable IEC format."""
    if not n or n <= 0:
        return "n/a"
    for unit, m in [("PiB", 1024**5), ("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)]:
        if n >= m:
            return f"{n/m:.2f} {unit}"
    return f"{n} B"


def secs_to_h(s: Optional[int]) -> str:
    return "n/a" if s is None else f"{s/3600:.2f}"


def pct_str(x: Optional[float]) -> str:
    return f"{x*100:.2f}%" if x is not None else "n/a"


def run(cmd: List[str]) -> str:
    """Run a command and capture stderr to avoid noisy sacct warnings."""
    return subprocess.check_output(cmd, text=True, errors="ignore", stderr=subprocess.PIPE)

def non_negative_int(val: str) -> int:
    n = int(val)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def parse_gpus_from_tres(tres: str) -> Optional[int]:
    if not tres:
        return None
    total = 0
    seen = False
    for part in tres.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("gres/gpu=") or part.startswith("gpu="):
            val = part.split("=", 1)[1]
            try:
                total += int(float(val))
                seen = True
            except ValueError:
                pass
    return total if seen else None


def parse_gpus_from_gres(gres: str) -> Optional[int]:
    if not gres:
        return None
    total = 0
    seen = False
    for part in gres.split(","):
        part = part.strip()
        if not part:
            continue
        toks = part.split(":")
        if len(toks) >= 2 and toks[0] == "gpu":
            try:
                count = int(toks[-1])
                total += count
                seen = True
            except ValueError:
                pass
    return total if seen else None


def parse_cpu_seconds_from_tres_usage(tres_usage: str) -> Optional[int]:
    """Extract CPU time from TRES usage strings like 'cpu=01:02:03,mem=...'."""
    if not tres_usage:
        return None
    for part in tres_usage.split(","):
        part = part.strip()
        if not part or not part.startswith("cpu="):
            continue
        val = part.split("=", 1)[1].strip()
        parsed = parse_slurm_time_to_seconds(val)
        if parsed is not None:
            return parsed
    return None


def parse_cpu_time_from_sacct_fields(data: Dict[str, str]) -> Optional[int]:
    """Best-effort CPU time extraction across Slurm/accounting variants."""
    total_cpu = parse_slurm_time_to_seconds(data.get("TotalCPU", ""))
    if total_cpu is not None and total_cpu > 0:
        return total_cpu

    user_cpu = parse_slurm_time_to_seconds(data.get("UserCPU", ""))
    system_cpu = parse_slurm_time_to_seconds(data.get("SystemCPU", ""))
    if user_cpu is not None or system_cpu is not None:
        return (user_cpu or 0) + (system_cpu or 0)

    tres_cpu = parse_cpu_seconds_from_tres_usage(data.get("TRESUsageInTot", ""))
    if tres_cpu is not None:
        return tres_cpu
    return total_cpu


def parse_sstat_live_usage(out: str) -> Optional[Dict[str, Optional[int]]]:
    """Parse sstat pipe output and pick the most informative step row."""
    best_score: Optional[tuple] = None
    best_row: Optional[Dict[str, Optional[int]]] = None

    def step_priority(step_id: str) -> int:
        sid = step_id.strip().lower().rstrip("+")
        if sid.endswith(".batch"):
            return 3
        if sid.endswith(".extern") or sid.endswith(".ext"):
            return 0
        if sid.endswith(".intern") or sid.endswith(".int"):
            return 2
        return 1

    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        step_id = parts[0].strip()
        cput_s = parse_slurm_time_to_seconds(parts[1].strip())
        ave_rss_b = parse_size_to_bytes(parts[2].strip())
        max_rss_b = parse_size_to_bytes(parts[3].strip())
        used_mem_b = max_rss_b or ave_rss_b
        cput_v = cput_s or 0
        mem_v = used_mem_b or 0
        has_usage = 1 if (cput_v > 0 or mem_v > 0) else 0
        score = (has_usage, step_priority(step_id), cput_v, mem_v)
        if best_score is None or score > best_score:
            best_score = score
            best_row = {"cput_s": cput_s, "used_mem_b": used_mem_b}
    return best_row


def get_live_usage_from_sstat(jobid: str) -> Optional[Dict[str, Optional[int]]]:
    """Best-effort live usage for running jobs when sacct fields are empty."""
    if not jobid or not shutil.which("sstat"):
        return None
    cmds = [
        ["sstat", "-n", "-P", "-j", jobid, "--allsteps", "--format=JobID,AveCPU,AveRSS,MaxRSS"],
        ["sstat", "-n", "-P", "-j", f"{jobid}.batch", "--format=JobID,AveCPU,AveRSS,MaxRSS"],
        ["sstat", "-n", "-P", "-j", jobid, "--format=JobID,AveCPU,AveRSS,MaxRSS"],
    ]
    for cmd in cmds:
        try:
            out = run(cmd)
        except subprocess.CalledProcessError:
            continue
        parsed = parse_sstat_live_usage(out)
        if parsed:
            return parsed
    return None


# ---------- sacct parsing ----------
SACCT_FIELDS_BASE = [
    "JobIDRaw",
    "JobName",
    "User",
    "State",
    "ElapsedRaw",
    "TotalCPU",
    "AllocCPUS",
    "ReqMem",
    "MaxRSS",
    "NNodes",
    "NodeList",
]  # minimal set that is widely supported
# Optional usage fields (availability depends on cluster/configuration)
SACCT_FIELDS_USAGE = ["UserCPU", "SystemCPU", "TRESUsageInTot"]
# Extended field sets (tried in order; sacct may reject unknown fields on some clusters)
SACCT_FIELDS_TRES = SACCT_FIELDS_BASE + ["AllocTRES", "ReqTRES"] + SACCT_FIELDS_USAGE
SACCT_FIELDS_GRES = SACCT_FIELDS_BASE + ["AllocGRES", "ReqGRES", "Gres"] + SACCT_FIELDS_USAGE
SACCT_FIELDS_FULL = SACCT_FIELDS_BASE + ["AllocTRES", "ReqTRES", "AllocGRES", "ReqGRES", "Gres"] + SACCT_FIELDS_USAGE
# Default/legacy alias for backward compatibility
SACCT_FIELDS = SACCT_FIELDS_FULL


def parse_state(raw_state: str) -> str:
    """Normalize State values like 'COMPLETED', 'CANCELLED by 0'."""
    if not raw_state:
        return "?"
    return raw_state.split()[0].upper()


def jobid_key(jobid: Optional[str]) -> str:
    if not jobid:
        return ""
    return jobid.split(".", 1)[0]


def summarize_from_sacct_line(line: str, fields: List[str]) -> Optional[Dict[str, Any]]:
    parts = line.split("|")
    if len(parts) < len(fields):
        return None
    data = {k: parts[i].strip() if i < len(parts) else "" for i, k in enumerate(fields)}
    jobid_raw = data["JobIDRaw"]
    # Skip step records like 123.batch or 123.0
    if not jobid_raw or "." in jobid_raw:
        return None

    jobid = jobid_raw
    name = data["JobName"]
    state = parse_state(data["State"])
    ncpus = int(data["AllocCPUS"]) if data["AllocCPUS"].isdigit() else None
    nnodes = int(data["NNodes"]) if data["NNodes"].isdigit() else None
    ngpus = (
        parse_gpus_from_tres(data.get("AllocTRES", ""))
        or parse_gpus_from_tres(data.get("ReqTRES", ""))
        or parse_gpus_from_gres(data.get("AllocGRES", ""))
        or parse_gpus_from_gres(data.get("ReqGRES", ""))
        or parse_gpus_from_gres(data.get("Gres", ""))
    )
    wall_s = int(data["ElapsedRaw"]) if data["ElapsedRaw"].isdigit() else None
    cput_s = parse_cpu_time_from_sacct_fields(data)
    used_mem_b = parse_size_to_bytes(data["MaxRSS"])
    req_mem_b = parse_reqmem_to_bytes(data["ReqMem"], ncpus=ncpus, nnodes=nnodes)
    avg_used_cpus = (cput_s / wall_s) if (cput_s is not None and wall_s and wall_s > 0) else None
    cpu_eff = (avg_used_cpus / ncpus) if (avg_used_cpus is not None and ncpus) else None
    mem_eff = (used_mem_b / req_mem_b) if (used_mem_b and req_mem_b and req_mem_b > 0) else None

    return {
        "jobid": jobid,
        "name": name,
        "state": state,
        "nodes": data.get("NodeList") or None,
        "ncpus": ncpus,
        "ngpus": ngpus,
        "wall_s": wall_s,
        "cput_s": cput_s,
        "avg_used_cpus": avg_used_cpus,
        "cpu_eff": cpu_eff,
        "used_mem_b": used_mem_b,
        "req_mem_b": req_mem_b,
        "mem_eff": mem_eff,
    }


def list_jobs_with_sacct(user: str, include_finished: bool, jobid: Optional[str]) -> List[Dict[str, Any]]:
    # Short state codes per sacct: PD=pending, R=running, CF=configuring, CG=completing, RQ=requeued, RS=resizing, S=suspended, SO=stageout
    states_active = ["PD", "R", "CF", "CG", "RQ", "RS", "S", "SO"]
    state_variants: List[Optional[str]] = [None] if include_finished else [
        "PD,R",                    # minimal, broadly supported set
        ",".join(states_active),   # full active set (may be unsupported on some clusters)
        None,                      # final fallback: no filter (include all)
    ]

    field_variants: List[List[str]] = [
        SACCT_FIELDS_TRES,  # preferred modern set (AllocGRES removed on newer Slurm)
        SACCT_FIELDS_FULL,
        SACCT_FIELDS_GRES,
        SACCT_FIELDS_BASE,
    ]

    last_err: Optional[Exception] = None
    out = ""
    fields_used: List[str] = SACCT_FIELDS
    for fields in field_variants:
        for state_arg in state_variants:
            cmd = [
                "sacct",
                "-n",
                "-P",
                f"--format={','.join(fields)}",
            ]
            if user:
                cmd += ["-u", user]
            if state_arg:
                cmd += ["-s", state_arg]
            if jobid:
                cmd += ["-j", jobid]
            try:
                out = run(cmd)
                last_err = None
                fields_used = fields
                break
            except subprocess.CalledProcessError as e:
                last_err = e
                continue
        if last_err is None:
            break

    if last_err:
        raise last_err

    rows = []
    rows_by_jobid: Dict[str, Dict[str, Any]] = {}
    step_cput_by_jobid: Dict[str, int] = {}
    step_mem_by_jobid: Dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= len(fields_used):
            data = {k: parts[i].strip() if i < len(parts) else "" for i, k in enumerate(fields_used)}
            jobid_raw = data.get("JobIDRaw", "")
            # Track step CPU time (e.g., 123.batch, 123.0) as fallback for clusters
            # where parent job TotalCPU is missing/zero while steps carry accounting data.
            if jobid_raw and "." in jobid_raw:
                base_jobid = jobid_raw.split(".", 1)[0]
                step_cput_s = parse_cpu_time_from_sacct_fields(data)
                if step_cput_s is not None:
                    step_cput_by_jobid[base_jobid] = step_cput_by_jobid.get(base_jobid, 0) + step_cput_s
                step_id = jobid_raw.lower().rstrip("+")
                is_extern = step_id.endswith(".extern") or step_id.endswith(".ext")
                if not is_extern:
                    step_mem_b = parse_size_to_bytes(data.get("MaxRSS", ""))
                    if step_mem_b is not None and step_mem_b > 0:
                        prev_mem_b = step_mem_by_jobid.get(base_jobid, 0)
                        if step_mem_b > prev_mem_b:
                            step_mem_by_jobid[base_jobid] = step_mem_b
        r = summarize_from_sacct_line(line, fields_used)
        if r:
            rows.append(r)
            rows_by_jobid[r["jobid"]] = r

    sstat_cache: Dict[str, Optional[Dict[str, Optional[int]]]] = {}
    for jobid_key, r in rows_by_jobid.items():
        parent_cput = r.get("cput_s")
        step_cput = step_cput_by_jobid.get(jobid_key)
        if (parent_cput is None or parent_cput <= 0) and step_cput is not None and step_cput > 0:
            wall_s = r.get("wall_s")
            ncpus = r.get("ncpus")
            avg_used_cpus = (step_cput / wall_s) if (wall_s and wall_s > 0) else None
            cpu_eff = (avg_used_cpus / ncpus) if (avg_used_cpus is not None and ncpus) else None
            r["cput_s"] = step_cput
            r["avg_used_cpus"] = avg_used_cpus
            r["cpu_eff"] = cpu_eff

        parent_mem = r.get("used_mem_b")
        step_mem = step_mem_by_jobid.get(jobid_key)
        if (parent_mem is None or parent_mem <= 0) and step_mem is not None and step_mem > 0:
            r["used_mem_b"] = step_mem
            req_mem_b = r.get("req_mem_b")
            r["mem_eff"] = (step_mem / req_mem_b) if (req_mem_b and req_mem_b > 0) else None

        state = (r.get("state") or "").upper()
        cpu_missing = r.get("cput_s") is None or r.get("cput_s", 0) <= 0
        mem_missing = r.get("used_mem_b") is None or r.get("used_mem_b", 0) <= 0
        parent_mem_missing = parent_mem is None or parent_mem <= 0
        need_live_mem = mem_missing or parent_mem_missing
        if state not in {"R", "RUNNING"} or (not cpu_missing and not need_live_mem):
            continue
        if jobid_key not in sstat_cache:
            sstat_cache[jobid_key] = get_live_usage_from_sstat(jobid_key)
        live = sstat_cache[jobid_key]
        if live:
            live_cput = live.get("cput_s")
            if cpu_missing and live_cput is not None and live_cput > 0:
                wall_s = r.get("wall_s")
                ncpus = r.get("ncpus")
                avg_used_cpus = (live_cput / wall_s) if (wall_s and wall_s > 0) else None
                cpu_eff = (avg_used_cpus / ncpus) if (avg_used_cpus is not None and ncpus) else None
                r["cput_s"] = live_cput
                r["avg_used_cpus"] = avg_used_cpus
                r["cpu_eff"] = cpu_eff

            live_mem = live.get("used_mem_b")
            if need_live_mem and live_mem is not None and live_mem > 0:
                r["used_mem_b"] = live_mem
                req_mem_b = r.get("req_mem_b")
                r["mem_eff"] = (live_mem / req_mem_b) if (req_mem_b and req_mem_b > 0) else None

        # If live reads failed and accounting still has no positive CPU signal, show n/a instead of 0.
        if (r.get("cput_s") is not None and r.get("cput_s", 0) <= 0) and (step_cput_by_jobid.get(jobid_key, 0) <= 0):
            r["cput_s"] = None
            r["avg_used_cpus"] = None
            r["cpu_eff"] = None
        if r.get("used_mem_b") is not None and r.get("used_mem_b", 0) <= 0:
            r["used_mem_b"] = None
            r["mem_eff"] = None
    return rows


# ---------- Output ----------
def render_table(rows: List[Dict[str, Any]], name_max: int) -> None:
    cols = ["JOBID", "STATE", "NAME", "NODES", "NCPUS", "NGPUS", "WALL(h)", "CPUT(h)", "avgCPU", "CPUeff", "memUsed", "memReq", "memEff"]
    w = {c: len(c) for c in cols}
    table = []
    for r in rows:
        name = r.get("name") or ""
        if name_max > 0 and len(name) > name_max:
            name = name[: max(0, name_max - 1)] + "…"
        row = {
            "JOBID": r["jobid"],
            "STATE": r["state"],
            "NAME": name,
            "NODES": r.get("nodes") or "n/a",
            "NCPUS": str(r["ncpus"] if r["ncpus"] is not None else "n/a"),
            "NGPUS": str(r["ngpus"] if r.get("ngpus") is not None else "n/a"),
            "WALL(h)": secs_to_h(r["wall_s"]),
            "CPUT(h)": secs_to_h(r["cput_s"]),
            "avgCPU": f"{r['avg_used_cpus']:.2f}" if r["avg_used_cpus"] is not None else "n/a",
            "CPUeff": pct_str(r["cpu_eff"]),
            "memUsed": fmt_bytes(r["used_mem_b"]),
            "memReq": fmt_bytes(r["req_mem_b"]),
            "memEff": pct_str(r["mem_eff"]),
        }
        for k, v in row.items():
            w[k] = max(w[k], len(str(v)))
        table.append(row)
    hdr = "  ".join(f"{c:<{w[c]}}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for row in table:
        print("  ".join(f"{row[c]:<{w[c]}}" for c in cols))


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv

    fields = [
        "jobid",
        "name",
        "state",
        "nodes",
        "ncpus",
        "ngpus",
        "wall_s",
        "cput_s",
        "avg_used_cpus",
        "cpu_eff",
        "used_mem_b",
        "used_mem_gb",
        "req_mem_b",
        "req_mem_gb",
        "mem_eff",
    ]
    f = sys.stdout if path == "-" else open(path, "w", newline="")
    with f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            row = {
                "jobid": r.get("jobid"),
                "name": r.get("name"),
                "state": r.get("state"),
                "nodes": r.get("nodes"),
                "ncpus": r.get("ncpus"),
                "ngpus": r.get("ngpus"),
                "wall_s": r.get("wall_s"),
                "cput_s": r.get("cput_s"),
                "avg_used_cpus": r.get("avg_used_cpus"),
                "cpu_eff": r.get("cpu_eff"),
                "used_mem_b": r.get("used_mem_b"),
                "req_mem_b": r.get("req_mem_b"),
                "mem_eff": r.get("mem_eff"),
            }

            if row["avg_used_cpus"] is not None:
                row["avg_used_cpus"] = round(row["avg_used_cpus"], 2)

            for field in ["cpu_eff", "mem_eff"]:
                if row.get(field) is not None:
                    row[field] = round(row[field], 2)

            for src, dest in [
                ("used_mem_b", "used_mem_gb"),
                ("req_mem_b", "req_mem_gb"),
            ]:
                val = row.get(src)
                row[dest] = round((val / (1024**3)), 2) if val is not None else None

            w.writerow({k: row.get(k) for k in fields})


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    import math

    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
        return sum(xs) / len(xs) if xs else float("nan")

    peak_mem = None
    mem_vals = [r.get("used_mem_b") for r in rows if r.get("used_mem_b")]
    if mem_vals:
        peak_mem = max(mem_vals)
    unique_nodes: set[str] = set()
    state_counts: Dict[str, int] = {}
    for r in rows:
        nodes_field = r.get("nodes")
        if not nodes_field:
            pass
        else:
            for node in nodes_field.split(","):
                node = node.strip()
                if node:
                    unique_nodes.add(node)

        state = (r.get("state") or "").strip().upper() or "?"
        state_counts[state] = state_counts.get(state, 0) + 1

    other_states = {
        k: v
        for k, v in state_counts.items()
        if k not in {"RUNNING", "R", "PENDING", "PD", "COMPLETED", "CD"}
    }
    return {
        "jobs": len(rows),
        "avg_CPUeff_%": mean([r["cpu_eff"] * 100 for r in rows if r.get("cpu_eff") is not None]),
        "avg_avgCPU": mean([r["avg_used_cpus"] for r in rows if r.get("avg_used_cpus") is not None]),
        "avg_memEff_%": mean([r["mem_eff"] * 100 for r in rows if r.get("mem_eff") is not None]),
        "max_mem_b": peak_mem,
        "unique_nodes": len(unique_nodes),
        "state_counts": state_counts,
        "state_other_total": sum(other_states.values()),
        "state_other_breakdown": other_states,
    }


ACTIVE_STATES = {
    "PD", "PENDING",
    "R", "RUNNING",
    "CF", "CONFIGURING",
    "CG", "COMPLETING",
    "RQ", "REQUEUED",
    "RS", "RESIZING",
    "S", "SUSPENDED",
    "SO", "STAGE_OUT",
}


def is_active_state(state: str) -> bool:
    return state.strip().upper() in ACTIVE_STATES


def collect_finished_jobids_with_sacct(user: str, finished_limit: int, active_keys: set[str]) -> List[str]:
    if finished_limit <= 0:
        return []
    cmd = [
        "sacct",
        "-n",
        "-P",
        "--format=JobIDRaw,State",
    ]
    if user:
        cmd += ["-u", user]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        errors="ignore",
    )
    if proc.stdout is None:
        return []

    selected: List[str] = []
    seen: set[str] = set()
    terminated_early = False
    try:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) < 2:
                continue
            jobid_raw = parts[0].strip()
            state = parse_state(parts[1].strip())
            if not jobid_raw or "." in jobid_raw:
                continue
            key = jobid_key(jobid_raw)
            if key in seen or key in active_keys:
                continue
            if is_active_state(state):
                continue
            seen.add(key)
            selected.append(jobid_raw)
            if len(selected) >= finished_limit:
                terminated_early = True
                proc.terminate()
                break
    finally:
        proc.stdout.close()

    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    if not terminated_early and proc.returncode not in (0, None):
        raise subprocess.CalledProcessError(proc.returncode or 1, cmd)
    return selected


def list_jobs_with_finished_limit_fetch(user: str, finished_limit: int) -> List[Dict[str, Any]]:
    active_rows = list_jobs_with_sacct(user=user, include_finished=False, jobid=None)
    if finished_limit <= 0:
        return active_rows

    active_keys = {jobid_key(r.get("jobid")) for r in active_rows}
    finished_jobids = collect_finished_jobids_with_sacct(
        user=user,
        finished_limit=finished_limit,
        active_keys=active_keys,
    )
    if not finished_jobids:
        return active_rows

    finished_rows = list_jobs_with_sacct(
        user="",
        include_finished=True,
        jobid=",".join(finished_jobids),
    )
    trimmed_finished = limit_finished_rows(
        [r for r in finished_rows if not is_active_state(r.get("state") or "")],
        finished_limit,
    )

    out = list(active_rows)
    seen_keys = set(active_keys)
    for row in trimmed_finished:
        key = jobid_key(row.get("jobid"))
        if key and key not in seen_keys:
            out.append(row)
            seen_keys.add(key)
    return out


def limit_finished_rows(rows: List[Dict[str, Any]], finished_limit: Optional[int]) -> List[Dict[str, Any]]:
    if finished_limit is None:
        return rows
    kept_finished = 0
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        state = (row.get("state") or "").strip().upper()
        is_finished = bool(state) and not is_active_state(state)
        if is_finished:
            if kept_finished >= finished_limit:
                continue
            kept_finished += 1
        filtered.append(row)
    return filtered


# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Slurm job CPU and memory statistics (single job or all jobs for a user)."
    )
    ap.add_argument(
        "--job",
        help="Job ID to summarize (default: $SLURM_JOB_ID). Returns the current job statistics if valid $SLURM_JOB_ID is found and --user is not set.",
    )
    ap.add_argument(
        "--user",
        help="Summarize all jobs of USER (default: current user). The --user argument takes precedence over --job.",
    )
    ap.add_argument(
        "--include-finished",
        action="store_true",
        help="Include finished jobs (sacct --state=all). Without this flag only pending/running jobs are shown.",
    )
    ap.add_argument(
        "--finished-limit",
        type=non_negative_int,
        metavar="N",
        help="With --include-finished, show at most N finished jobs. Active jobs are always shown.",
    )
    ap.add_argument(
        "--finished-limit-strategy",
        choices=["post", "fetch"],
        default="post",
        help="How to apply --finished-limit: post=fetch all then trim (default), fetch=fetch active + up to N finished jobs.",
    )
    ap.add_argument(
        "--csv",
        metavar="PATH",
        help='Write CSV to PATH (use "-" for stdout)',
    )
    ap.add_argument(
        "--name-max",
        type=int,
        default=30,
        help="Max width for job name column; 0=disable truncation (default: 30)",
    )
    args = ap.parse_args()

    if args.finished_limit is not None and not args.include_finished:
        ap.error("--finished-limit requires --include-finished")

    if not shutil.which("sacct"):
        print("ERROR: sacct not found in PATH.", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []

    jobid = args.job or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    default_user = (
        os.environ.get("SLURM_JOB_USER") or os.environ.get("USER") or getpass.getuser()
    )

    if args.user is None and jobid:
        rows = list_jobs_with_sacct(user="", include_finished=True, jobid=jobid)
    else:
        user = args.user or default_user
        if not user:
            print(
                "Provide --job JOBID or run inside Slurm with $SLURM_JOB_ID set, or use --user USER.",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.include_finished and args.finished_limit is not None and args.finished_limit_strategy == "fetch":
            try:
                rows = list_jobs_with_finished_limit_fetch(user=user, finished_limit=args.finished_limit)
            except subprocess.CalledProcessError:
                rows = list_jobs_with_sacct(user=user, include_finished=True, jobid=None)
            rows = limit_finished_rows(rows, args.finished_limit)
        else:
            rows = list_jobs_with_sacct(user=user, include_finished=args.include_finished, jobid=None)
            rows = limit_finished_rows(rows, args.finished_limit)

    render_table(rows, name_max=args.name_max)
    agg = aggregate(rows)
    print("\nSummary:")
    print(f"  jobs:         {agg['jobs']}")
    print(f"  unique nodes: {agg['unique_nodes']}")
    state_counts = agg.get("state_counts", {})
    r_count = sum(state_counts.get(k, 0) for k in ("RUNNING", "R"))
    pd_count = sum(state_counts.get(k, 0) for k in ("PENDING", "PD"))
    cd_count = sum(state_counts.get(k, 0) for k in ("COMPLETED", "CD"))
    other_total = agg.get("state_other_total", 0)
    print(f"  states:       R={r_count}  PD={pd_count}  CD={cd_count}  other={other_total}")
    other_breakdown = agg.get("state_other_breakdown") or {}
    if other_breakdown:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(other_breakdown.items()))
        print(f"    other breakdown: {detail}")
    if agg["avg_CPUeff_%"] == agg["avg_CPUeff_%"]:
        print(f"  mean CPUeff:  {agg['avg_CPUeff_%']:.2f}%")
    if agg["avg_avgCPU"] == agg["avg_avgCPU"]:
        print(f"  mean avgCPU:  {agg['avg_avgCPU']:.2f}")
    if agg["avg_memEff_%"] == agg["avg_memEff_%"]:
        print(f"  mean memEff:  {agg['avg_memEff_%']:.2f}%")
    if agg.get("max_mem_b"):
        print(f"  max memUsed:  {fmt_bytes(agg['max_mem_b'])}")

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
