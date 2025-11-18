#!/usr/bin/env python
"""
W&B Sweep Integration for 3-Stage World Model Training

This script wraps the training pipeline to integrate with Weights & Biases
hyperparameter sweeps. It handles W&B configuration, logging, and metrics
tracking.

USAGE:
    python train_with_sweep.py  # Normal training
    wandb agent <sweep_id>      # When run as part of a W&B sweep
"""

import os
import sys
import wandb
import hydra
import logging
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

log = logging.getLogger(__name__)


def initialize_wandb_sweep():
    """
    Initialize W&B integration for sweeps.

    This function:
    1. Initializes W&B run (required before accessing config)
    2. Gets sweep parameters from W&B if in sweep mode
    3. Returns sweep config to be merged with Hydra config
    """
    # Initialize W&B (required for sweep integration)
    wandb.init()

    # Get sweep configuration from W&B
    sweep_config = {}
    if wandb.config:
        # Convert W&B config to dict for merging with Hydra config
        sweep_config = dict(wandb.config)
        log.info(f"Loaded sweep config from W&B: {sweep_config}")

    return sweep_config


def merge_configs(hydra_cfg: DictConfig, sweep_cfg: dict) -> DictConfig:
    """
    Merge W&B sweep config with Hydra config.

    W&B sweep parameters override Hydra defaults.

    Args:
        hydra_cfg: Hydra configuration
        sweep_cfg: W&B sweep configuration

    Returns:
        Merged configuration
    """
    # Create a copy to avoid modifying original
    merged_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)

    # Merge sweep config (uses dot notation for nested keys)
    for key, value in sweep_cfg.items():
        # Handle nested keys like "model.quantization_loss_weight"
        if "." in key:
            parts = key.split(".")
            current = merged_cfg

            # Navigate to nested dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value
            current[parts[-1]] = value
        else:
            # Top-level key
            merged_cfg[key] = value

    # Convert back to DictConfig
    merged_cfg = OmegaConf.create(merged_cfg)
    return merged_cfg


def log_sweep_info(cfg: DictConfig):
    """
    Log sweep configuration to W&B and console.

    Args:
        cfg: Configuration being used for training
    """
    # Log key hyperparameters
    log.info("=" * 70)
    log.info("HYPERPARAMETER SWEEP CONFIGURATION")
    log.info("=" * 70)
    log.info(f"Quantization Loss Weight: {cfg.model.quantization_loss_weight}")
    log.info(f"VCReg Loss Weight: {cfg.model.vcreg_loss_weight}")
    log.info(f"State Quantizer LR: {cfg.training.state_quantizer_lr}")
    log.info(f"Predictor LR: {cfg.training.predictor_lr}")
    log.info(f"Encoder LR: {cfg.training.encoder_lr}")
    log.info("=" * 70)

    # Log to W&B
    wandb.log({
        "sweep/quantization_loss_weight": cfg.model.quantization_loss_weight,
        "sweep/vcreg_loss_weight": cfg.model.vcreg_loss_weight,
        "sweep/state_quantizer_lr": cfg.training.state_quantizer_lr,
        "sweep/predictor_lr": cfg.training.predictor_lr,
        "sweep/encoder_lr": cfg.training.encoder_lr,
    })


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> float:
    """
    Main training function for W&B sweeps.

    This function:
    1. Initializes W&B sweep integration
    2. Merges sweep parameters with Hydra config
    3. Runs training
    4. Returns validation loss for sweep optimization

    Args:
        cfg: Hydra configuration

    Returns:
        Final validation loss (for sweep optimization)
    """
    # Initialize W&B and get sweep config
    sweep_cfg = initialize_wandb_sweep()

    # Merge configurations
    if sweep_cfg:
        cfg = merge_configs(cfg, sweep_cfg)
        log.info("Merged W&B sweep config with Hydra config")

    # Log sweep configuration
    log_sweep_info(cfg)

    # Now import and run the actual training
    # Import here to avoid circular imports
    from train import TrainingPipeline

    # Create training pipeline with merged config
    trainer = TrainingPipeline(cfg)

    # Run training
    final_val_loss = trainer.train()

    # Log final metrics to W&B for sweep optimization
    wandb.log({
        "sweep/final_validation_loss": final_val_loss,
    })

    return final_val_loss


if __name__ == "__main__":
    main()
