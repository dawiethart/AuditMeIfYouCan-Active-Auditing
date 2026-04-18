# Audit Me If You Can

Code for the paper **"Audit Me If You Can: Query-Efficient Active Fairness 
Auditing of Black-Box LLMs"** (ACL 2026 Findings).

BAFA audits black-box LLM fairness by maintaining a version space of surrogate 
models consistent with queried scores, computing uncertainty intervals for 
fairness metrics via constrained empirical risk minimisation, and actively 
selecting queries that maximally shrink those intervals. On CivilComments and 
Bias-in-Bios, BAFA reaches target error thresholds with up to 40× fewer queries 
than stratified sampling.

## Installation

```bash
git clone https://github.com/dawiethart/audit-me-if-you-can.git
cd audit-me-if-you-can
pip install -r requirements.txt
cp .env.example .env
```

## Quickstart

```bash
# Audit CivilComments/HateBERT with disagreement-based selection
python main.py \
    --dataset jigsaw \
    --blackbox hatebert \
    --strategy disagreement \
    --iterations 75 \
    --seed 42 \
    --output_dir ./results

# Audit Bias-in-Bios with BO-based selection
python main.py \
    --dataset bios \
    --blackbox bios_csv \
    --bios_scores_csv blackboxes/blackbox_bios.csv \
    --strategy bo \
    --iterations 75 \
    --seed 42
```

## Reproducing paper results

```bash
# Run main experiments (20 seeds, both case studies)
python experiments/experiment_main_bafa.py

# Hyperparameter sweeps
python experiments/experiment_hyperparameter_sweeps.py

# Surrogate ablations
python experiments/experiment_surrogate_ablations.py
```

See `experiments/EXPERIMENTS_README.md` for full details.

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--strategy` | `bo` | Selection strategy: `random`, `stratified`, `disagreement`, `bo` |
| `--iterations` | `50` | Number of audit rounds |
| `--size_T` | `4` | Batch size per round |
| `--epochs_opt` | `3` | C-ERM gradient steps per round |
| `--seed` | `0` | Random seed |

Full argument reference: `python main.py --help`

## Repository structure

├── main.py                     # Entry point
├── audit_run.py                # Audit loop
├── selection.py                # Selection strategies
├── surrogate_model.py          # BERT surrogate training
├── optimization.py             # C-ERM via Cooper
├── evaluation.py               # AUC bounds and metrics
├── blackboxes/                 # Black-box model wrappers
└── experiments/                # Scripts to reproduce paper results

## Citation

```bibtex
@inproceedings{hartmann2026auditme,
  title     = {Audit Me If You Can: Query-Efficient Active Fairness Auditing 
               of Black-Box {LLMs}},
  author    = {Hartmann, David and Pohlmann, Lena and Hanslik, Lelia and 
               Gie{\ss}ing, Noah and Berendt, Bettina and Delobelle, Pieter},
  booktitle = {Findings of the Association for Computational Linguistics: 
               ACL 2026},
  year      = {2026}
}
```

## License

MIT License

Copyright (c) 2026 David Hartmann, Lena Pohlmann, Lelia Hanslik, 
Noah Gießing, Bettina Berendt, Pieter Delobelle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.