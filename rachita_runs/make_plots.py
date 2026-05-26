"""Generate headline plots from Megatron training logs.

Produces three figures, mirroring Lingfeng's attention-sweep figures:

  1. Throughput bar chart (sorted by median tokens/sec/GPU, p10-p90 error bars).
  2. Loss-curve correctness overlay (one line per variant; all should overlap
     pre-divergence, indicating no kernel-fusion technique alters numerics).
  3. Memory bar chart (max-reserved + max-allocated MB/GPU per variant).

Each log is automatically classified into one of:
    baseline, torch_compile_default, torch_compile_fullmodel,
    torch_compile_reduce_overhead, torch_compile_max_autotune,
    torch_compile_plus_cuda_graphs, cuda_graphs_local, cuda_graphs_te,
    cuda_graphs_te_attn, quack, context_parallel_cp2
by parsing the `[rachita] torch_compile=... target=... mode=... quack=...` line
and the `CMD: torchrun ... pretrain_gpt.py ... --cuda-graph-impl ... --context-parallel-size ...`
line. Runs that produced fewer than `--min-steps` completed iterations are
listed as "failed" in the JSON sidecar and dropped from the throughput / memory
plots (still drawn on the loss plot if they have at least 1 datapoint).

Usage::

    python rachita_runs/make_plots.py logs/gipfel-throughput-8b-*.log \\
        --out-dir rachita_runs/results/figures \\
        --skip 10

This script needs matplotlib; everything else is stdlib. Run it inside the
alps3 container (or on the login node after `pip install --user matplotlib`).
Pure post-processing; no GPU required.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import parse_throughput as pt


VARIANT_ORDER = [
    "baseline",
    "torch_compile_default",
    "torch_compile_fullmodel",
    "torch_compile_reduce_overhead",
    "torch_compile_max_autotune",
    "torch_compile_plus_cuda_graphs",
    "cuda_graphs_local",
    "cuda_graphs_te",
    "cuda_graphs_te_attn",
    "quack",
    "context_parallel_cp2",
]

PRETTY = {
    "baseline": "baseline",
    "torch_compile_default": "torch.compile\n(layers)",
    "torch_compile_fullmodel": "torch.compile\n(full model)",
    "torch_compile_reduce_overhead": "torch.compile\nreduce-overhead",
    "torch_compile_max_autotune": "torch.compile\nmax-autotune",
    "torch_compile_plus_cuda_graphs": "torch.compile\n+ CUDA graphs",
    "cuda_graphs_local": "CUDA graphs\n(local)",
    "cuda_graphs_te": "CUDA graphs\n(TE, full layer)",
    "cuda_graphs_te_attn": "CUDA graphs\n(TE, attn only)",
    "quack": "Quack\nRMSNorm",
    "context_parallel_cp2": "CP=2\n(seq=8192)",
}

RACHITA_LINE_RE = re.compile(
    r"\[rachita\] torch_compile=(?P<tc>\d+) quack=(?P<q>\d+) mode=(?P<mode>\S+) target=(?P<target>\S+)"
)
CMD_LINE_RE = re.compile(r"^CMD: torchrun.*pretrain_gpt(?:_rachita)?\.py\b(?P<args>.*)$")
CG_IMPL_RE = re.compile(r"--cuda-graph-impl\s+(\S+)")
CG_SCOPE_RE = re.compile(r"--cuda-graph-scope\s+(\S+)")
CP_RE = re.compile(r"--context-parallel-size\s+(\d+)")
SEQLEN_RE = re.compile(r"--seq-length\s+(\d+)")


def classify_log(path):
    """Return one of VARIANT_ORDER or 'unknown'."""
    rachita = None
    cmd_args = None
    for line in path.read_text(errors="ignore").splitlines():
        if rachita is None:
            m = RACHITA_LINE_RE.search(line)
            if m:
                rachita = m.groupdict()
        if cmd_args is None:
            m = CMD_LINE_RE.search(line)
            if m:
                cmd_args = m.group("args")
        if rachita is not None and cmd_args is not None:
            break

    if cmd_args:
        if CP_RE.search(cmd_args):
            return "context_parallel_cp2"
        cg = CG_IMPL_RE.search(cmd_args)
        scope = CG_SCOPE_RE.search(cmd_args)
    else:
        cg = scope = None

    tc_on = rachita is not None and rachita["tc"] == "1"
    quack_on = rachita is not None and rachita["q"] == "1"

    if tc_on and cg:
        return "torch_compile_plus_cuda_graphs"
    if tc_on:
        mode = rachita["mode"]
        target = rachita["target"]
        if mode == "reduce-overhead":
            return "torch_compile_reduce_overhead"
        if mode == "max-autotune":
            return "torch_compile_max_autotune"
        if target == "model":
            return "torch_compile_fullmodel"
        return "torch_compile_default"
    if quack_on:
        return "quack"
    if cg:
        impl = cg.group(1)
        if impl == "local":
            return "cuda_graphs_local"
        if impl == "transformer_engine":
            return "cuda_graphs_te_attn" if scope and "attn" in scope.group(1) else "cuda_graphs_te"
    if rachita is None and cmd_args is not None:
        return "baseline"
    if rachita is not None and not tc_on and not quack_on and not cg:
        return "baseline"
    return "unknown"


def collect(log_paths, skip):
    """Returns (rows, failed) where rows is a dict {variant: payload}.

    Failed runs (no parseable iterations after skip) go in `failed` and are
    excluded from the throughput / memory plots.
    """
    rows = {}
    failed = []
    for path in log_paths:
        variant = classify_log(path)
        steps, memory = pt.parse_log(path)
        summary = pt.summarise(steps, skip=skip, memory=memory) if steps else None
        if summary is None or summary["n_steps_used"] < 3:
            failed.append({
                "log": str(path), "variant": variant,
                "n_steps_total": len(steps),
                "reason": "no usable iterations after skip={}".format(skip),
            })
            continue
        if variant in rows:
            sys.stderr.write(
                "warning: duplicate variant '{}' (kept {}, skipping {})\n".format(
                    variant, rows[variant]["log"], path)
            )
            continue
        rows[variant] = {
            "log": str(path),
            "variant": variant,
            "summary": summary,
            "loss_series": [(s["step"], s["loss"]) for s in steps if s["loss"] is not None],
        }
    return rows, failed


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_throughput(rows, out_path):
    plt = _matplotlib()
    items = []
    for v in VARIANT_ORDER:
        if v not in rows:
            continue
        if v == "context_parallel_cp2":
            continue
        tps = rows[v]["summary"]["tokens_per_sec_per_gpu"]
        if tps["median"] is None:
            continue
        items.append((v, tps["median"], tps["p10"], tps["p90"]))
    items.sort(key=lambda x: x[1], reverse=True)
    if not items:
        return None

    labels = [PRETTY[v] for v, _, _, _ in items]
    med = [x[1] for x in items]
    err_lo = [x[1] - x[2] for x in items]
    err_hi = [x[3] - x[1] for x in items]

    baseline_med = rows["baseline"]["summary"]["tokens_per_sec_per_gpu"]["median"] if "baseline" in rows else None

    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(items) + 2), 4.5))
    colors = ["#4c72b0" if v != "baseline" else "#888888" for v, *_ in items]
    bars = ax.bar(range(len(items)), med, yerr=[err_lo, err_hi],
                  capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("tokens / sec / GPU")
    ax.set_title("8B steady-state throughput (median; whiskers show p10\u2013p90 iteration range)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    if baseline_med is not None:
        ax.axhline(baseline_med, color="#888888", linestyle="--", linewidth=0.8,
                   label="baseline median")
        ax.legend(loc="lower right", fontsize=8)
    ymin = min(med) - max(err_lo)
    ymax = max(med) + max(err_hi)
    ax.set_ylim(ymin * 0.97, ymax * 1.03)

    for i, (m, lo, hi) in enumerate(zip(med, err_lo, err_hi)):
        label = "{:.0f}".format(m)
        if baseline_med and abs(m - baseline_med) / baseline_med > 0.005:
            pct = (m - baseline_med) / baseline_med * 100.0
            sign = "+" if pct > 0 else ""
            label += "\n({}{:.1f}%)".format(sign, pct)
        ax.text(i, m + hi + (ymax - ymin) * 0.01, label,
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_loss(rows, out_path):
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    n = 0
    for v in VARIANT_ORDER:
        if v not in rows:
            continue
        if v == "context_parallel_cp2":
            continue
        series = rows[v]["loss_series"]
        if not series:
            continue
        xs, ys = zip(*series)
        ax.plot(xs, ys, "-", label=PRETTY[v].replace("\n", " "),
                color=cmap(n % 10), linewidth=1.2, alpha=0.85)
        n += 1
    ax.set_xlabel("iteration")
    ax.set_ylabel("lm loss")
    ax.set_title("Correctness check: 8B training loss per variant\n"
                 "(identical seed at seq=4096; curves should overlap pre-divergence)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_memory(rows, out_path):
    plt = _matplotlib()
    items = []
    for v in VARIANT_ORDER:
        if v not in rows:
            continue
        if v == "context_parallel_cp2":
            continue
        mem = rows[v]["summary"].get("memory_mb") or {}
        max_resv = mem.get("max_reserved_mb")
        max_alloc = mem.get("max_allocated_mb")
        if max_resv is None and max_alloc is None:
            continue
        items.append((v, max_resv, max_alloc))
    if not items:
        return None
    items.sort(key=lambda x: (x[1] if x[1] is not None else 0))

    labels = [PRETTY[v] for v, _, _ in items]
    resv = [x[1] for x in items]
    alloc = [x[2] for x in items]
    x = list(range(len(items)))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(items) + 2), 4.5))
    ax.bar([i - w / 2 for i in x], resv, w, color="#4c72b0",
           edgecolor="black", linewidth=0.6, label="max reserved")
    ax.bar([i + w / 2 for i in x], alloc, w, color="#dd8452",
           edgecolor="black", linewidth=0.6, label="max allocated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MB / GPU")
    ax.set_title("8B peak GPU memory per variant")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    for i, (r, a) in enumerate(zip(resv, alloc)):
        if r is not None:
            ax.text(i - w / 2, r, "{:.0f}".format(r),
                    ha="center", va="bottom", fontsize=7)
        if a is not None:
            ax.text(i + w / 2, a, "{:.0f}".format(a),
                    ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", type=Path, nargs="+", help="Megatron log files (.log).")
    p.add_argument("--out-dir", type=Path, default=Path("rachita_runs/results/figures"))
    p.add_argument("--skip", type=int, default=10,
                   help="Steps to drop as warmup before computing statistics.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, failed = collect(args.logs, skip=args.skip)

    print("Detected variants:")
    for v in VARIANT_ORDER:
        if v in rows:
            s = rows[v]["summary"]
            tps = s["tokens_per_sec_per_gpu"]
            mem = (s.get("memory_mb") or {}).get("max_reserved_mb")
            print("  {:35s}  n={:3d}  tps={:>7.1f} (p10={:.0f}/p90={:.0f})  max_resv={}".format(
                v, s["n_steps_used"], tps["median"] or float("nan"),
                tps["p10"] or float("nan"), tps["p90"] or float("nan"),
                "{:.0f} MB".format(mem) if mem else "n/a"))
    if failed:
        print("\nFailed / incomplete runs:")
        for f in failed:
            print("  {:35s}  log={}".format(f["variant"], f["log"]))

    out = {}
    out["throughput"] = str(plot_throughput(rows, args.out_dir / "8b_throughput_bar.png"))
    out["loss"] = str(plot_loss(rows, args.out_dir / "8b_loss_curves.png"))
    out["memory"] = str(plot_memory(rows, args.out_dir / "8b_memory_bar.png"))

    sidecar = args.out_dir / "8b_summary.json"
    sidecar.write_text(json.dumps({
        "skip": args.skip,
        "variants": {v: rows[v]["summary"] for v in rows},
        "logs": {v: rows[v]["log"] for v in rows},
        "failed": failed,
    }, indent=2, sort_keys=True))

    print("\nWrote:")
    for k, v in out.items():
        print("  {:10s} {}".format(k + ":", v))
    print("  json:      {}".format(sidecar))


if __name__ == "__main__":
    main()
