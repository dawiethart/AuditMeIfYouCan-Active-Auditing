# BAFA Experimental Suite

This directory contains scripts to reproduce all experiments from the paper **"Audit Me If You Can: Query-Efficient Active Fairness Auditing of Black-Box LLMs"**.

## 📋 Overview

The experimental suite consists of:

1. **Main Experiments (Table 1)**: Compare BAFA against baselines across 20 seeds
2. **Surrogate Ablations (Appendix A.6.1)**: Test different surrogate architectures
3. **Hyperparameter Sweeps (Tables 5-6, Appendix A.4.3)**: Systematic parameter ablations

## 🚀 Quick Start

### Running Everything

```bash
# Complete experimental suite (WARNING: ~400 runs, days of compute)
python run_all_experiments.py --mode all

# Main experiments only (Table 1 results)
python run_all_experiments.py --mode main

# Quick test with 1 seed per configuration
python run_all_experiments.py --mode test
```

### Running Individual Experiments

#### 1. Main Experiments (Table 1)

**Case Study A: CivilComments (Hate Speech Detection)**
```bash
python experiment_main_bafa.py \
  --dataset jigsaw \
  --blackbox hatebert \
  --strategies bo disagreement stratified \
  --num_seeds 20
```

**Case Study B: Bias-in-Bios (Occupation Prediction)**
```bash
python experiment_main_bafa.py \
  --dataset bios \
  --blackbox bios_csv \
  --bios_scores_csv blackbox_bios.csv \
  --bios_max_rows 50000 \
  --strategies bo disagreement stratified \
  --num_seeds 20
```

#### 2. Surrogate Ablations (Appendix A.6.1)

Test BAFA with different surrogate models:

```bash
python experiment_surrogate_ablations.py \
  --dataset jigsaw \
  --blackbox hatebert \
  --surrogates bert-base-uncased distilbert-base-uncased roberta-large \
  --strategies disagreement bo
```

Models tested:
- `bert-base-uncased` (baseline, 110M params)
- `distilbert-base-uncased` (smaller, 66M params)
- `roberta-large` (larger, 355M params)

#### 3. Hyperparameter Sweeps (Tables 5-6)

**C-ERM Optimization Epochs (Table 5)**
```bash
python experiment_hyperparameter_sweeps.py \
  --sweep epochs_opt \
  --dataset jigsaw \
  --blackbox hatebert
```

**Active Batch Size k (Table 6)**
```bash
python experiment_hyperparameter_sweeps.py \
  --sweep active_batch \
  --dataset jigsaw \
  --blackbox hatebert
```

**C-ERM Batch Size B_cerm (Table 6)**
```bash
python experiment_hyperparameter_sweeps.py \
  --sweep cerm_batch \
  --dataset jigsaw \
  --blackbox hatebert
```

**BO Exploration Parameter β**
```bash
python experiment_hyperparameter_sweeps.py \
  --sweep bo_beta \
  --dataset jigsaw \
  --blackbox hatebert \
  --strategies bo
```

**BO Diversity Penalty γ**
```bash
python experiment_hyperparameter_sweeps.py \
  --sweep bo_diversity \
  --dataset jigsaw \
  --blackbox hatebert \
  --strategies bo
```

**Run all sweeps**
```bash
python experiment_hyperparameter_sweeps.py --sweep all
```

## 📊 Experiment Structure

### Main Experiments

| Script | Purpose | Seeds | Configurations | Total Runs |
|--------|---------|-------|----------------|------------|
| `experiment_main_bafa.py` | Table 1 results | 20 | 3 strategies × 2 datasets | 120 |

**Strategies:**
- `bo`: BAFA with Bayesian Optimization
- `disagreement`: BAFA with Disagreement sampling
- `stratified`: Stratified sampling baseline (C-ERM only)

### Surrogate Ablations

| Script | Purpose | Seeds | Configurations | Total Runs |
|--------|---------|-------|----------------|------------|
| `experiment_surrogate_ablations.py` | Appendix A.6.1 | 5 | 3 models × 2 strategies | 30 |

### Hyperparameter Sweeps

| Sweep | Parameter | Values | Purpose |
|-------|-----------|--------|---------|
| `epochs_opt` | C-ERM epochs | [3, 6, 8, 10] | Table 5 |
| `active_batch` | Queries/round (k) | [8, 16, 32] | Table 6 |
| `cerm_batch` | C-ERM batch (B) | [256, 512, 1024, 2048] | Table 6 |
| `bo_beta` | BO exploration (β) | [0.5, 1.0, 2.0] | BO tuning |
| `bo_diversity` | BO diversity (γ) | [0.0, 0.1, 0.2, 0.5] | BO tuning |
| `reg_alpha` | Distribution reg (α) | [0.5, 1.0, 2.0, 4.0] | Regularization |

## ⚙️ Hyperparameters (Table 7)

### Default Settings

**CivilComments:**
```python
--size_T 16              # Queries per round
--iterations 75          # Total audit rounds
--epochs_sur 4           # Surrogate fine-tuning epochs
--epochs_opt 10          # C-ERM optimization epochs
--batch_size 512         # C-ERM batch size
--lambda_penalty 1e-2    # Constraint tolerance λ
--epsilon 1e-2           # Target precision
--reg_alpha 2.0          # Distribution matching penalty
--bo_beta 1.0            # BO UCB parameter
--bo_diversity_gamma 0.2 # BO diversity weight
```

**Bias-in-Bios:**
Same as CivilComments except:
```python
--epochs_opt 8           # Reduced for stability
```

## 🔬 Reproducing Paper Results

### Table 1: Query Efficiency and Performance

```bash
# Run main experiments for both case studies
python run_all_experiments.py --mode main

# Analyze results (example)
python analyze_results.py \
  --metric queries_to_epsilon \
  --epsilon 0.02 0.05 \
  --strategies bo disagreement stratified
```

### Table 5: C-ERM Optimization Epochs

```bash
python experiment_hyperparameter_sweeps.py --sweep epochs_opt
```

### Table 6: Batch Size Ablations

```bash
python experiment_hyperparameter_sweeps.py --sweep active_batch
python experiment_hyperparameter_sweeps.py --sweep cerm_batch
```

### Figure 3 & 4: Convergence Curves

Results are automatically logged to W&B (Weights & Biases). Plot using:

```python
import wandb
api = wandb.Api()
runs = api.runs("your-project/bafa")
# Plot convergence curves
```

## 💾 Data Requirements

### CivilComments (Case Study A)
- Downloaded automatically via datasets library
- ~50k comments after filtering
- No pre-processing required

### Bias-in-Bios (Case Study B)
- Requires cached black-box scores: `blackbox_bios.csv`
- Generated using GPT-4.1-mini with deterministic prompting
- Format: `id, gold_occupation, gender, pred_occupation, score_0, score_1, ..., score_27`

**Generate scores:**
```bash
python generate_bios_scores.py \
  --output blackbox_bios.csv \
  --model gpt-4.1-mini-2025-04-14 \
  --max_rows 50000
```

## 📈 Expected Runtimes

Based on RTX 4090 / A6000 / A100 (40-48GB):

| Experiment | Per Run | Total (20 seeds) |
|------------|---------|------------------|
| BAFA-BO (Jigsaw) | 4-6h | 80-120h |
| BAFA-Disagreement (Jigsaw) | 5-7h | 100-140h |
| BAFA-BO (Bios) | 6-8h | 120-160h |
| Surrogate ablation | 4-8h | 20-40h (5 seeds) |
| Hyperparam sweep | 3-5h | Varies by sweep |

**Total for complete suite:** ~400-600 GPU-hours

## 🔧 Customization

### Custom Seeds
```bash
python experiment_main_bafa.py --seeds 0 1 2 3 4
```

### Custom Strategies
```bash
python experiment_main_bafa.py --strategies bo
```

### Dry Run (preview commands)
```bash
python experiment_main_bafa.py --dry_run
```

### Custom W&B Project
```bash
export WANDB_PROJECT=my-bafa-experiments
python experiment_main_bafa.py
```

## 📝 Script Descriptions

### Core Experiment Runners

- **`run_all_experiments.py`**: Master coordinator for complete experimental suite
- **`experiment_main_bafa.py`**: Main experiments (Table 1) with all strategies
- **`experiment_surrogate_ablations.py`**: Test different surrogate architectures
- **`experiment_hyperparameter_sweeps.py`**: Systematic parameter ablations

### Legacy Scripts (for reference)

- `run_20_seeds.py`: Simple 20-seed runner (replaced by `experiment_main_bafa.py`)
- `experiment_disagreement_reg_grid.py`: Reg-alpha grid (replaced by hyperparameter sweeps)
- `experiment_BO_BAFA_updated.py`: BO-specific configs (merged into main)
- `experiment_5x3_*.py`: Small-scale experiments (superseded)

## 🐛 Troubleshooting

### Out of Memory

**For large surrogates (RoBERTa-large):**
```bash
# Reduce batch size
python experiment_surrogate_ablations.py --batch_size 256
```

**For BO with large candidate pools:**
```bash
# Reduce candidate pool size in main.py
--candidate_pool_size 500  # default: 1000
```

### Slow C-ERM Convergence

```bash
# Reduce epochs_opt for faster runs (trade-off: wider bounds)
--epochs_opt 6  # instead of 10
```

### W&B Quota Issues

```bash
# Run offline
export WANDB_MODE=offline
python experiment_main_bafa.py
```


