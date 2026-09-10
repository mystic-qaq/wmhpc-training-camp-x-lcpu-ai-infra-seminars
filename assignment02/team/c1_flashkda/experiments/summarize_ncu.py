#!/usr/bin/env python3
"""Print the useful rows of ``ncu --page raw --csv`` without huge names.

Usage::

    ncu --import report.ncu-rep --page raw --csv | \
      python experiments/summarize_ncu.py

The raw CSV contains a very long C++ kernel name and hundreds of columns.  A
small parser makes the result suitable for putting in a report while keeping
the original ``.ncu-rep`` as the source of truth.
"""

from __future__ import annotations

import csv
import re
import sys


PREFERRED_METRICS = (
    # The first two are the same concept exposed by different Nsight Compute
    # metric sets/versions.  A full profile can contain either one.
    ("dram throughput (% peak)", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("dram throughput (% peak)", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L1/TEX throughput (% peak, active)", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("L1/TEX throughput (% peak, elapsed)", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L2 throughput (% peak)", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("SM throughput (% peak)", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("tensor pipe active (% peak)", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
    ("tensor pipe elapsed (% peak)", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("HMMA pipe active (% peak)", "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active"),
    ("HMMA pipe elapsed (% peak)", "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("TMA pipe active (% peak)", "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active"),
    ("TMA pipe elapsed (% peak)", "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("TMA load instructions", "smsp__inst_executed_op_tma_ld.sum"),
    ("TMA store instructions", "smsp__inst_executed_op_tma_st.sum"),
    ("UMMA matrix instructions", "smsp__mem_tensor_reads_op_utcmma_matrix_c.sum"),
    ("SM active warps (% peak)", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("registers/thread", "launch__registers_per_thread"),
    ("registers/thread allocated", "launch__registers_per_thread_allocated"),
    ("shared memory/block (KiB)", "launch__shared_mem_per_block"),
    ("shared memory/block allocated (KiB)", "launch__shared_mem_per_block_allocated"),
    ("theoretical occupancy: registers", "launch__occupancy_limit_registers"),
    ("theoretical occupancy: shared memory", "launch__occupancy_limit_shared_mem"),
    ("theoretical occupancy: warps", "launch__occupancy_limit_warps"),
    ("theoretical occupancy: blocks", "launch__occupancy_limit_blocks"),
    ("theoretical occupancy: barriers", "launch__occupancy_limit_barriers"),
    ("occupancy per register count", "launch__occupancy_per_register_count"),
    ("occupancy per shared-memory size", "launch__occupancy_per_shared_mem_size"),
    ("cluster occupancy (%)", "launch__occupancy_cluster_pct"),
    ("kernel duration", "gpu__time_duration"),
)


def short_metric(name: str) -> str:
    # Nsight sometimes prefixes a metric with a subunit/metric-set name.  The
    # report should retain the actual metric name but not the verbose suffix.
    name = name.rsplit(".", 1)[-1] if name.startswith("unit.") else name
    return re.sub(r"\.(avg|sum|pct_of_peak_sustained_(?:active|elapsed)|per_second)$", "", name)


def choose_columns(header: list[str]) -> list[tuple[int, str]]:
    """Choose one available column for each useful metric.

    NCU 2026 profiles have names such as
    ``FBSP.TriageCompute.dram__throughput...``.  Matching by substring keeps
    this parser usable across NCU metric-set prefixes while the preference
    order avoids printing both active and elapsed variants unnecessarily.
    """
    columns: list[tuple[int, str]] = []
    used: set[int] = set()
    used_labels: set[str] = set()
    for label, wanted in PREFERRED_METRICS:
        # Some NCU versions expose both a derived gpu__ metric and a raw
        # subunit metric for DRAM.  The second entry is a fallback, not a
        # second column to print.
        if label in used_labels:
            continue
        for i, name in enumerate(header):
            lowered = name.lower()
            if i in used or wanted.lower() not in lowered:
                continue
            # Don't accidentally choose a .max/.min/.sum column when the
            # requested metric is an .avg or a launch attribute.
            if ".max." in lowered or ".min." in lowered:
                continue
            columns.append((i, label))
            used.add(i)
            used_labels.add(label)
            break
    return columns


def main() -> int:
    rows = csv.reader(sys.stdin)
    try:
        header = next(rows)
        units = next(rows)
    except StopIteration:
        print("empty ncu CSV", file=sys.stderr)
        return 2

    columns = choose_columns(header)
    print("metric columns: " + ", ".join(label for _, label in columns))

    seen: set[str] = set()
    print("values (first invocation of each kernel):")
    for row in rows:
        if not row or not row[0].isdigit() or len(row) <= 4:
            continue
        kernel = row[4]
        kind = "prepare" if "_flash_kda_fwd_prepare" in kernel else (
            "recurrence" if "_flash_kda_fwd_recurrence" in kernel else "other"
        )
        if kind == "other" or kind in seen:
            continue
        seen.add(kind)
        print(f"[{kind}] id={row[0]} block={row[7]} grid={row[8]}")
        for i, label in columns:
            if i < len(row) and row[i] not in ("", "n/a"):
                unit = units[i] if i < len(units) else ""
                print(f"  {label} = {row[i]} {unit}".rstrip())
        if len(seen) == 2:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
