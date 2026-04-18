# config.py
"""
Configuration module for audit experiments.
Supports environment variables and dataclass defaults.
"""
from dataclasses import dataclass
from typing import Optional
import os


def get_env(key: str, default: str) -> str:
    """Get environment variable or return default."""
    return os.getenv(key, default)


@dataclass
class AuditConfig:
    """Configuration for audit runs. All values can be overridden by CLI args or env vars."""
    
    # ===== Model & Surrogate =====
    model: str = "lora"  # Surrogate model architecture
    
    # ===== Query Budget & Iteration =====
    size_T: int = 4  # Top-k batch size (queries per round)
    iterations: int = 50  # Total audit rounds
    k_batch: int = 8  # Same as size_T for consistency
    
    # ===== Surrogate Training =====
    use_surrogate: bool = True
    surrogate_epochs: int = 2  # Per-round surrogate fine-tuning
    surrogate_lr: float = 2e-5
    surrogate_batch_size: int = 16
    surrogate_max_len: int = 128
    candidate_pool_M: int = 1000
    
    # ===== C-ERM Optimization =====
    epochs_opt: int = 3  # C-ERM gradient steps per round
    epochs_sur: int = 3  # Surrogate training epochs
    batch_size: int = 32  # C-ERM batch size
    lambda_penalty: float = 0.5  # Constraint tolerance λ
    epsilon: float = 1e-2  # Target precision / stopping threshold
    reg_alpha: float = 1.0  # Distribution matching penalty
    
    # ===== Bayesian Optimization =====
    bo_beta: float = 1.0  # UCB exploration parameter
    bo_acq: str = "ucb"  # Acquisition function (ucb, ei, poi)
    bo_diversity_gamma: float = 0.2  # Diversity penalty weight
    
    # ===== Selection Strategy =====
    strategy: str = "bo"  # Selection strategy: random, stratified, disagreement, bo
    
    # ===== Dataset & Blackbox =====
    dataset: str = "jigsaw"  # Dataset choice: jigsaw, bios, sbic
    blackbox: str = "hatebert"  # Blackbox model: hatebert, bios_csv, perspective
    
    # ===== Reproducibility =====
    seed: int = 0  # Random seed
    
    # ===== Logging & Output =====
    output_dir: str = "./outputs"  # Where to save results
    use_wandb: bool = False  # Enable Weights & Biases logging
    wandb_project: Optional[str] = None  # W&B project name (uses env var if None)
    wandb_entity: Optional[str] = None  # W&B entity name (uses env var if None)
    wandb_prefix: str = "audit"  # Prefix for metric names in W&B
    title: Optional[str] = None  # Run title/name
    
    # ===== Dataset-Specific =====
    bios_scores_csv: Optional[str] = None  # Path to precomputed bias-in-bios scores
    bios_max_rows: Optional[int] = None  # Limit bias-in-bios dataset size
    bios_targets: Optional[list] = None  # Target protected attributes for bios
    
    def __post_init__(self):
        """Populate from environment variables if not set."""
        if self.wandb_project is None:
            self.wandb_project = get_env("WANDB_PROJECT", "audit-repo")
        if self.wandb_entity is None:
            self.wandb_entity = get_env("WANDB_ENTITY", "anonymous")