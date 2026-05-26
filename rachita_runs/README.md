# Rachita's Experiments

Kernel-fusion (`torch.compile`, CUDA Graphs, Quack) and long-context (TransformerEngine context parallelism) benchmarks. Everything is gated by config keys that default to off, so the upstream training path is unchanged when no Rachita feature is enabled.

## Layout

```
rachita_runs/
├── pretrain_gpt_rachita.py        # thin wrapper around Megatron's pretrain_gpt.py
├── rachita_extensions/
│   ├── torch_compile_wrap.py      # gated torch.compile wrapping
│   └── quack_kernels.py           # gated Quack RMSNorm monkey-patches
├── configs/                       # one YAML per benchmark variant (table below)
├── parse_throughput.py            # log → JSON summary (median tps, p10/p90, memory)
├── make_plots.py / make_plots.sbatch   # logs → throughput / loss / memory PNGs
├── run_benchmarks.sh              # submit/dry-run the whole matrix
├── install_quack.sbatch           # one-shot Quack pip install inside alps3
└── results/figures/               # generated PNGs + JSON sidecar
```

## How the wrapper works

`pretrain_gpt_rachita.py` mirrors upstream's `__main__` block:

1. If `MEGATRON_USE_QUACK=1`, install Quack monkey-patches **before** importing `pretrain_gpt`.
2. If `MEGATRON_TORCH_COMPILE=1`, wrap each `TransformerLayer` (or the whole model) with `torch.compile()` inside `model_provider`.
3. Call upstream `pretrain()` with the original `extra_args_provider`, `get_embedding_ranks`, and `inprocess_restart` machinery preserved.

`launch_with_config.py` automatically selects this wrapper when either `torch_compile` or `use_quack` is enabled; otherwise it uses upstream `pretrain_gpt.py` directly. CUDA-graph and context-parallel configs go through upstream unchanged — they just add the appropriate Megatron CLI flags.

## Config keys (added to `default_config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `torch_compile` | `false` | Master switch for `torch.compile`. |
| `torch_compile_mode` | `default` | `default`, `reduce-overhead`, `max-autotune`. |
| `torch_compile_backend` | `inductor` | |
| `torch_compile_target` | `layers` | `layers` (per-`TransformerLayer`) or `model` (top-level). |
| `torch_compile_fullgraph` | `false` | Forbid graph breaks if true. |
| `torch_compile_dynamic` | `false` | Megatron uses static shapes; leave false. |
| `cuda_graph_impl` | `null` | `null`, `local`, or `transformer_engine`. |
| `cuda_graph_warmup_steps` | `1` | |
| `cuda_graph_scope` | `null` | Only valid with `transformer_engine`, e.g. `[attn]`. |
| `use_quack` | `false` | Falls back gracefully if Quack is not installed. |
| `quack_ops` | `[rmsnorm, cross_entropy]` | We only patch `rmsnorm` in practice. |
| `context_parallel_size` | `1` | Maps to `--context-parallel-size`. |
| `pytorch_cuda_alloc_conf` | `null` | e.g. `expandable_segments:True`. Exported as `PYTORCH_CUDA_ALLOC_CONF`. |

## Experiment matrix

| YAML | What it tests |
|---|---|
| `config_baseline.yaml` | Reference; no fusion. |
| `config_torch_compile.yaml` | `torch.compile` per-layer, `mode=default`. |
| `config_torch_compile_fullmodel.yaml` | `torch.compile` whole model, `mode=default` (scope ablation). |
| `config_torch_compile_reduce_overhead.yaml` | `torch.compile`, `mode=reduce-overhead` (Inductor auto-wraps CUDA graphs). |
| `config_torch_compile_max_autotune.yaml` | `torch.compile`, `mode=max-autotune`. |
| `config_cuda_graphs_local.yaml` | Megatron native `local` impl (1 graph per layer). |
| `config_cuda_graphs_te.yaml` | TE `make_graphed_callables` over the full layer. |
| `config_cuda_graphs_te_attn.yaml` | TE graph capture scoped to attention only. |
| `config_quack.yaml` | Quack RMSNorm monkey-patch. |
| `config_context_parallel_cp2.yaml` | Long-context: CP=2, `seq_len=8192`. |

All configs are overlays on `default_config.yaml`; they set only what they need to override.

## Running

Submit the whole matrix at the 8B headline scale:

```bash
bash rachita_runs/run_benchmarks.sh --size 8b --steps 80 --submit
```

Or just one variant (short names, comma-separated):

```bash
bash rachita_runs/run_benchmarks.sh --size 8b --steps 80 \
     --only baseline,quack,cuda_graphs_te_attn --submit
```

`--list` prints the registered short names; omit `--submit` for dry-run (sbatch files only, no submission).

The launcher tags every output sbatch as `logs/rachita-<short_name>-...sbatch` so job-to-config mapping is unambiguous.

## Parsing logs

Single log → JSON summary:

```bash
python3 rachita_runs/parse_throughput.py --base-skip 10 \
    logs/gipfel-throughput-8b-*-<JOBID>.log
```

Side-by-side table across many logs (first is treated as baseline; reports delta vs baseline for both throughput and memory):

```bash
python3 rachita_runs/parse_throughput.py --base-skip 10 --compare \
    logs/<baseline>.log logs/<variant1>.log logs/<variant2>.log ...
```

## Generating plots

Pure post-processing; produces three PNGs + a JSON sidecar in `rachita_runs/results/figures/`:

```bash
sbatch rachita_runs/make_plots.sbatch
```

The sbatch wraps `make_plots.py logs/gipfel-throughput-8b-*.log --skip 10`. The script auto-classifies each log into its variant from the `[rachita]` log line and the `--cuda-graph-impl` / `--context-parallel-size` CLI args, so the order of the glob doesn't matter.

Outputs:

| File | Contents |
|---|---|
| `8b_throughput_bar.png` | Bars sorted by median tokens/s/GPU; whiskers show p10–p90 iteration range. |
| `8b_loss_curves.png` | Correctness check: lm-loss vs iteration, one line per variant. |
| `8b_memory_bar.png` | Max-reserved + max-allocated MB per variant. |
| `8b_summary.json` | All numbers behind the plots (so the report can cite them directly). |

## Installing Quack

Quack is not in the alps3 image by default. Submit:

```bash
sbatch rachita_runs/install_quack.sbatch
```

This runs `pip install --user quack-kernels` inside an alps3 allocation. The wrapper falls back to upstream RMSNorm with a `[rachita]` warning if the import fails at training time, so a missing install never crashes a run.

## Validation checklist

- [ ] Every variant uses `--base-skip 10` so the median is computed over iterations 11–80.
- [ ] `attention_preset`, `precision`, `global_batch_size`, and `seq_len` are held constant across the comparison set (otherwise tokens/s/GPU is not apples-to-apples).
