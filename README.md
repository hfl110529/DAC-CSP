# CSP: Continual Profiled Side-Channel Analysis Across Heterogeneous Devices

Anonymous code artifact for the paper. The framework connects temporal leakage traces to a shared Large Language Model (LLM) through a Trace Tokenizer; Class-Conditioned Subspace Projection (CSP) learns new tasks with protected, projected low-rank updates.

This repository contains the model implementation, continual-learning methods, experiment configurations, evaluation, plotting, and tests. Raw traces, pretrained weights, trained checkpoints, paper sources, and historical experiment logs are not included. The command-line interfaces use portable paths instead of the original research-workspace layout.

## Paper-to-code map

| Paper component | Implementation / configuration | Entry point or output |
| --- | --- | --- |
| Section III-A: trace tokenizer, full LLM backbone, mean pooling and shared classifier; Figures 1–2 | [`models.py`](src/csp_sca/models.py) | `MLPQwen3SCA`, `TraceLLM`, `build_model` |
| Section III-B: projected residual and the zero response on stored directions | [`protected_lowrank.py`](src/csp_sca/protected_lowrank.py) | `ProtectedLowRankLinear`, `install_qv_adapters` |
| Section III-C: class/token means and cumulative memory | [`memory.py`](src/csp_sca/memory.py), [`multidevice_memory.py`](src/csp_sca/multidevice_memory.py) | `class_means`, `basis_from_class_token_means`, `union_basis` |
| Algorithm 1: source fitting, sequential updates and memory extraction | [`experiments.py`](src/csp_sca/experiments.py), [`engine.py`](src/csp_sca/engine.py) | `train`, `fit`, `update_memory` |
| Table I: eleven tasks, architectures and label rules | [`configs/main.json`](configs/main.json), [`data.py`](src/csp_sca/data.py) | `pack`, `validate-data` |
| Figure 3: dataset NICV | [`analysis.py`](src/csp_sca/analysis.py), [`plots.py`](src/csp_sca/plots.py) | `plot nicv` |
| Table II: ablation comparison (recovery, inflation and ACC) | `table3` in [`configs/suites.json`](configs/suites.json) | `suite --name table3`, `report`; `summary.csv` |
| Figure 4: Qwen3 with/without CSP, Wide MLP and Transformer | `backbones` in [`configs/suites.json`](configs/suites.json), [`models.py`](src/csp_sca/models.py) | `suite --name backbones`, `plot ge` |
| Figure 5: final recovery heatmap and attack-trace costs | `main` in [`configs/suites.json`](configs/suites.json), [`metrics.py`](src/csp_sca/metrics.py) | `suite --name main`, `report`; `task_costs.csv` |
| Figure 6(a,b): task ACC and mean ACC after updates | [`experiments.py`](src/csp_sca/experiments.py), [`plots.py`](src/csp_sca/plots.py) | `evaluate`, `plot retention` |
| Figure 6(c): relative Q/V input drift | [`cli.py`](src/csp_sca/cli.py), [`analysis.py`](src/csp_sca/analysis.py) | `features`, `drift`, `plot retention --drift` |
| Table III: six-task token sweep | [`configs/six_tasks.json`](configs/six_tasks.json), `tokens` in [`configs/suites.json`](configs/suites.json) | `suite --name tokens`, `report` |
| Section VI-B: changing the position of XMEGA | `order` in [`configs/suites.json`](configs/suites.json) | `suite --name order` |
| Additional LLM backbones in Section VI | [`configs/small_qwen2.json`](configs/small_qwen2.json), [`configs/small_minicpm.json`](configs/small_minicpm.json) | `train` and `evaluate` with the corresponding local model |

The implementations of GPM and O-LoRA are in [`baselines.py`](src/csp_sca/baselines.py); InfLoRA is in [`inflora.py`](src/csp_sca/inflora.py). [`controls.py`](src/csp_sca/controls.py) contains class-balanced replay, effective-Q/V Fisher regularization, and the parameter-matched InfLoRA variant. Figures 1–2 are conceptual illustrations of the corresponding model and update code.

## Installation

Python 3.12 is recommended. Install a suitable PyTorch build for your CUDA environment, then install this repository:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m csp_sca --help
```

The Qwen3 interface targets the complete Qwen3-1.7B backbone with 28 blocks and hidden size 2048. Supply an existing local model directory through `--model`; model loading does not automatically download weights. The dependency specification uses Transformers 4.57.6. MLP and Transformer comparisons do not require pretrained LLM weights.

## Data interface

Prepare one directory per task:

```text
data/
  F0/
    train.npz
    validation.npz
    attack.npz
  F1/
    ...
  XMEGA/
    ...
```

The full schema, sample-ID checks, labels, and preprocessing are described in [`docs/data.md`](docs/data.md). Use `pack` to validate disjoint splits. Training opens only profiling and validation files; attack files are opened by the separate evaluation command after a run is complete.

```bash
csp-sca pack --input prepared/F0.npz --output data/F0 --label-rule first_sbox
csp-sca validate-data --config configs/main.json --data data
```

The current manuscript configuration uses the task order `F0, F1, F2, F3, F4, XMEGA, ASCAD-0, ASCAD-2, DPA-v4, DPA-v2, AES_HD`. The six-task study uses `F0, XMEGA, F1, F2, F3, F4`. Public-dataset windows must be chosen using profiling data before packing. The loader applies per-trace standardization and the specified 500-to-700 interpolation for MCU traces.

## Main experiments

Inspect the jobs before running the complete suite:

```bash
csp-sca suite --name main --dry-run
csp-sca suite --name main --data data --model models/Qwen3-1.7B --output runs --evaluate
csp-sca report --runs runs/main --output outputs/main
```

Each suite runs seeds **13, 42, and 2026**. The source stage trains the tokenizer and classifier at `1e-3` and the first LLM block at `1e-5`. Later CSP stages train rank-8 Q/V residuals at `1e-4`. The default budget is 10 epochs, batch size 64, clipping at 1, and 5% warmup followed by cosine decay. Checkpoint selection uses the earliest minimum of current-task validation NLL.

A single run can be invoked directly:

```bash
csp-sca train --config configs/main.json --method csp --seed 13 \
  --data data --model models/Qwen3-1.7B --output runs/single-csp
csp-sca evaluate --run runs/single-csp --data data --model models/Qwen3-1.7B
```

Use `--resume` with `train` or `suite` to reuse completed stage boundaries with the same configuration. An interrupted stage is restarted from its preceding completed checkpoint. Source and increment checkpoints store the selected trainable tensors; the original pretrained model directory is needed when reconstructing LLM runs.

The Joint reference freezes the fitted F0 source, adds one unprotected rank-8 residual, and fits pooled training/validation splits. Its report contains source and final evaluation cells; sequential forgetting metrics are not assigned to Joint.

## Ablations and other backbones

```bash
# Table II: replay, GPM/O-LoRA settings, EWC, factor/memory variants, caps and backbones
csp-sca suite --name table3 --data data --model models/Qwen3-1.7B --output runs --evaluate
csp-sca report --runs runs/table3 --output outputs/table3

# Figure 4: four model configurations
csp-sca suite --name backbones --data data --model models/Qwen3-1.7B --output runs --evaluate
csp-sca plot ge --runs runs/backbones --output outputs/fig4-ge.pdf

# Table III and Section VI-B: use the native 500-sample MCU bundles
csp-sca suite --name tokens --data data --model models/Qwen3-1.7B --output runs --evaluate
csp-sca suite --name order --data data --model models/Qwen3-1.7B --output runs --evaluate
csp-sca report --runs runs/tokens --output outputs/table4

# Additional LLMs: repeat for the configured seeds with the corresponding weights
csp-sca train --config configs/small_qwen2.json --seed 13 --data data \
  --model models/Qwen2-0.5B --output runs/qwen2-seed13
csp-sca train --config configs/small_minicpm.json --seed 13 --data data \
  --model models/MiniCPM5-1B --output runs/minicpm-seed13
```

The Wide MLP has eight residual blocks of width 2048. The independent Transformer has eight blocks, 16 attention heads, width 2048, and feed-forward width 8192. Production configurations preserve these dimensions; CPU integration tests use smaller networks to check the execution path.

## Figures and metrics

```bash
csp-sca plot nicv --config configs/main.json --data data --output outputs/fig3-nicv.pdf
csp-sca plot retention --runs runs/main --output outputs/fig5-accuracy.pdf
```

For the XMEGA feature-stability comparison, stages are zero-based: stage 4 is after F4, and stage 5 is after XMEGA. Export the same validation samples before and after that update:

```bash
csp-sca features --run runs/main/csp-seed13 --stage 4 --task F0 --samples 512 \
  --data data --model models/Qwen3-1.7B --output outputs/F0-before.npz
csp-sca features --run runs/main/csp-seed13 --stage 5 --task F0 --samples 512 \
  --data data --model models/Qwen3-1.7B --output outputs/F0-after.npz
csp-sca drift --before outputs/F0-before.npz --after outputs/F0-after.npz \
  --label CSP --output outputs/drift.json --append
csp-sca plot retention --runs runs/main --drift outputs/drift.json --output outputs/fig5-retention.pdf
```

Repeat the feature export and append step for F1–F4 and the compared methods. The drift panel takes the median across task/Q/V entries at each block. The paired feature archives must have matching task names and trace IDs. Exporting all Q/V inputs of a full LLM can require substantial host memory and disk space.

Evaluation uses 200 random attack orders and at most 5000 traces. `n_attack` is the first budget from which the mean zero-based correct-key rank stays zero to the horizon, corresponding to the paper's recovery trace count. Ties use average rank; success requires a unique correct winner. Different key groups are ranked independently.

`report` produces:

- `task_costs.csv`: mean recovery cost over recovered cases, with recovered/attempted counts for each task.
- `summary.csv`: recovery rate, final ACC, geometric-mean attack-cost inflation on retained cases, ACC loss in percentage points, tokenizer parameter count, and the six-task maximum recovery cost.
- `summary.json`: the same values with configuration and task details.

Missing recovery is represented by `null`/empty CSV fields. Summaries require the configured seed cohort unless `--allow-incomplete` is explicitly selected. The supplied recipes use the current paper settings; recovering the published measurements requires the corresponding prepared data, model weights and experiment runs.

## Tests and source scope

Tests cover projected-update invariants, gradients in both low-rank factors, frozen history, cumulative memory, baseline constraints, key-rank ties, split separation, checkpoint reconstruction, stage-boundary resume and report aggregation. The integration fixtures are synthetic correctness checks, not paper results. Full LLM training is not part of the test suite.

Release validation passed **131 CPU tests**, including the CLI training/evaluation/reporting and PDF plotting paths, with Python 3.12.3, PyTorch 2.14.1, NumPy 2.5.4, and Matplotlib 3.11.2. Archived training metadata records PyTorch 2.9.0 with CUDA 12.8 and Transformers 4.57.6; the complete pretrained-model experiments were not rerun during this source cleanup.

This release has no inherited research-workspace Git history, author metadata, institution links, or machine-specific data/model paths. Generated runs, data, weights and figures are excluded by `.gitignore`. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for dependencies and method attribution.

Before publishing a changed source tree, run `python tools/check_anonymity.py`. Additional names or identifiers can be checked with repeated `--forbid` arguments.
