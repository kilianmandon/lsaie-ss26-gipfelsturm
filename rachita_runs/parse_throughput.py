"""Compute steady-state throughput from a Megatron training log.

Parses per-iteration Megatron stdout / log lines (with the
``tokens/sec/GPU`` field added by ``patches/0001-log-tokens-per-sec-to-wandb.patch``),
drops warmup / compile iterations, and prints a JSON summary suitable for
pasting into W&B run notes.

Usage::

    python rachita_runs/parse_throughput.py logs/<job>.log
    python rachita_runs/parse_throughput.py logs/<job>.log \\
        --torch-compile-skip 3 --cuda-graph-warmup 1

Pure stdlib - no extra deps.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ITER_RE = re.compile(r"iteration\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)")
ELAPSED_RE = re.compile(r"elapsed time per iteration \(ms\):\s*(?P<ms>[0-9.]+)")
TFLOPS_RE = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*(?P<tflops>[0-9.]+)")
TOKENS_RE = re.compile(r"tokens/sec/GPU:\s*(?P<tps>[0-9]+(?:\.[0-9]+)?)")
LOSS_RE = re.compile(r"lm loss:\s*(?P<loss>[0-9.eE+\-]+)")

# Memory bookkeeping (matches lines like
# "[Rank 0] (after 2 iterations) memory (MB) | allocated: 46727.23 | "
# "max allocated: 85667.84 | reserved: 89304.00 | max reserved: 89304.00").
# Megatron prints these only a handful of times early in training, but the
# "max reserved" / "max allocated" are high-watermarks across the whole run.
MEM_RE = re.compile(
    r"\(after\s+(?P<step>\d+)\s+iterations\)\s+memory\s+\(MB\)\s*\|"
    r"\s*allocated:\s*(?P<alloc>[0-9.]+)\s*\|"
    r"\s*max allocated:\s*(?P<max_alloc>[0-9.]+)\s*\|"
    r"\s*reserved:\s*(?P<resv>[0-9.]+)\s*\|"
    r"\s*max reserved:\s*(?P<max_resv>[0-9.]+)"
)
# Theoretical memory estimate that Megatron prints at startup, e.g.
# "Theoretical memory footprints: weight and optimizer=63443.29 MB, "
# "activation=37482.97 MB, total=100926.25 MB"
THEORETICAL_MEM_RE = re.compile(
    r"Theoretical memory footprints:\s*weight and optimizer=\s*(?P<weight>[0-9.]+)\s*MB"
    r"\s*,\s*activation=\s*(?P<act>[0-9.]+)\s*MB"
    r"\s*,\s*total=\s*(?P<total>[0-9.]+)\s*MB"
)


def parse_log(path):
    """Returns (per_step_metrics, memory_summary)."""
    steps = []
    mem_samples = []
    theoretical = None
    for line in path.read_text(errors="ignore").splitlines():
        mm = MEM_RE.search(line)
        if mm:
            mem_samples.append({
                "step": int(mm.group("step")),
                "allocated_mb": float(mm.group("alloc")),
                "max_allocated_mb": float(mm.group("max_alloc")),
                "reserved_mb": float(mm.group("resv")),
                "max_reserved_mb": float(mm.group("max_resv")),
            })
            continue
        if theoretical is None:
            tm = THEORETICAL_MEM_RE.search(line)
            if tm:
                theoretical = {
                    "weight_and_optimizer_mb": float(tm.group("weight")),
                    "activation_mb": float(tm.group("act")),
                    "total_mb": float(tm.group("total")),
                }
        m = ITER_RE.search(line)
        if not m:
            continue
        ms = ELAPSED_RE.search(line)
        tflops = TFLOPS_RE.search(line)
        tps = TOKENS_RE.search(line)
        loss = LOSS_RE.search(line)
        steps.append({
            "step": int(m.group("step")),
            "elapsed_ms": float(ms.group("ms")) if ms else None,
            "tflops": float(tflops.group("tflops")) if tflops else None,
            "tps": float(tps.group("tps")) if tps else None,
            "loss": float(loss.group("loss")) if loss else None,
        })
    memory = None
    if mem_samples:
        memory = {
            "max_reserved_mb": max(s["max_reserved_mb"] for s in mem_samples),
            "max_allocated_mb": max(s["max_allocated_mb"] for s in mem_samples),
            "n_samples": len(mem_samples),
        }
    if theoretical is not None:
        memory = (memory or {})
        memory["theoretical_total_mb"] = theoretical["total_mb"]
        memory["theoretical_weight_optimizer_mb"] = theoretical["weight_and_optimizer_mb"]
        memory["theoretical_activation_mb"] = theoretical["activation_mb"]
    return steps, memory


def _pct(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round((len(ys) - 1) * q))))
    return ys[k]


def summarise(steps, skip, memory=None):
    body = steps[skip:]
    if not body:
        return None
    tps = [s["tps"] for s in body if s["tps"] is not None]
    tfl = [s["tflops"] for s in body if s["tflops"] is not None]
    losses = [s["loss"] for s in body if s["loss"] is not None]
    elapsed = [s["elapsed_ms"] for s in body if s["elapsed_ms"] is not None]

    first_ms = next((s["elapsed_ms"] for s in steps if s["elapsed_ms"] is not None), None)
    median_ms = statistics.median(elapsed) if elapsed else None
    overhead = None
    if first_ms is not None and median_ms is not None:
        overhead = max(0.0, first_ms - median_ms)

    return {
        "n_steps_total": len(steps),
        "n_steps_used": len(body),
        "skipped": skip,
        "first_step_elapsed_ms": first_ms,
        "median_step_elapsed_ms": median_ms,
        "compile_overhead_ms": overhead,
        "tokens_per_sec_per_gpu": {
            "median": statistics.median(tps) if tps else None,
            "p10": _pct(tps, 0.10),
            "p90": _pct(tps, 0.90),
        },
        "tflops_per_gpu": {
            "median": statistics.median(tfl) if tfl else None,
            "p10": _pct(tfl, 0.10),
            "p90": _pct(tfl, 0.90),
        },
        "lm_loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            # statistics.fmean is Python >= 3.8; the login node ships 3.6,
            # so stick to statistics.mean here.
            "mean": statistics.mean(losses) if losses else None,
        },
        "memory_mb": memory,
    }


def _summarise_log(path, skip):
    if not path.is_file():
        return None, "log not found: {}".format(path)
    steps, memory = parse_log(path)
    if not steps:
        return None, "no iteration lines parsed"
    summary = summarise(steps, skip=skip, memory=memory)
    if summary is None:
        return None, "only {} steps parsed; nothing left after skip={}".format(len(steps), skip)
    return summary, None


def _label_for(path):
    return path.stem


def _print_compare_table(rows):
    headers = ["run", "n_used", "tokens/s/GPU median", "p10", "p90",
               "TFLOPS median", "step ms median", "first ms", "compile ms",
               "max reserved MB", "max alloc MB",
               "loss first", "loss last", "loss mean"]
    widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))

    base = rows[0]
    base_med = base[2]
    if isinstance(base_med, (int, float)) and base_med > 0:
        print()
        print("Delta vs baseline ({}) for tokens/s/GPU median:".format(base[0]))
        for r in rows[1:]:
            med = r[2]
            if isinstance(med, (int, float)):
                pct = (med - base_med) / base_med * 100.0
                sign = "+" if pct >= 0 else ""
                print("  {:30s}  {:>10}  ({}{:.2f}%)".format(r[0], med, sign, pct))

    # Memory delta vs baseline (negative = saving)
    base_mem = rows[0][9]
    if isinstance(base_mem, (int, float)) and base_mem > 0:
        print()
        print("Delta vs baseline ({}) for max reserved memory (MB/GPU):".format(rows[0][0]))
        for r in rows[1:]:
            mem = r[9]
            if isinstance(mem, (int, float)):
                delta = mem - base_mem
                pct = delta / base_mem * 100.0
                sign = "+" if delta >= 0 else ""
                print("  {:30s}  {:>10.1f}  ({}{:.0f} MB, {}{:.2f}%)".format(
                    r[0], mem, sign, delta, sign, pct))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", type=Path, nargs="+",
                   help="One or more Megatron log files. The first is treated as the baseline.")
    p.add_argument("--torch-compile-skip", type=int, default=0,
                   help="Extra steps to drop for torch.compile recompiles")
    p.add_argument("--cuda-graph-warmup", type=int, default=0,
                   help="Extra steps to drop for CUDA-graph warmup")
    p.add_argument("--base-skip", type=int, default=1,
                   help="Steps to drop unconditionally as iteration-1 warmup")
    p.add_argument("--compare", action="store_true",
                   help="Print a side-by-side table across all logs (delta vs first log)")
    args = p.parse_args()

    skip = args.base_skip + args.torch_compile_skip + args.cuda_graph_warmup

    if not args.compare and len(args.logs) == 1:
        summary, err = _summarise_log(args.logs[0], skip)
        if err:
            print(err, file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    rows = []
    for path in args.logs:
        summary, err = _summarise_log(path, skip)
        if err:
            print("{}: {}".format(path, err), file=sys.stderr)
            continue
        mem = summary.get("memory_mb") or {}
        rows.append([
            _label_for(path),
            summary["n_steps_used"],
            summary["tokens_per_sec_per_gpu"]["median"],
            summary["tokens_per_sec_per_gpu"]["p10"],
            summary["tokens_per_sec_per_gpu"]["p90"],
            summary["tflops_per_gpu"]["median"],
            summary["median_step_elapsed_ms"],
            summary["first_step_elapsed_ms"],
            summary["compile_overhead_ms"],
            mem.get("max_reserved_mb"),
            mem.get("max_allocated_mb"),
            summary["lm_loss"]["first"],
            summary["lm_loss"]["last"],
            round(summary["lm_loss"]["mean"], 4) if summary["lm_loss"]["mean"] is not None else None,
        ])
    if not rows:
        return 2
    _print_compare_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
