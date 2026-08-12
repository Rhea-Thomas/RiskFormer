"""
Main training script for RiskFormer.

Usage:
    python train.py --config configs/base.yaml

Workflow:
    1. Load configuration
    2. Download market data
    3. Clean and engineer features
    4. Create train/val/test splits (chronologically)
    5. Fit normalization on train only
    6. Create rolling-window dataset
    7. Initialize model, optimizer, loss function
    8. Train with validation
    9. Save checkpoint
"""

import argparse
import yaml
import torch


def main(config_path):
    """
    Main training entry point.

    Args:
        config_path: path to YAML configuration file
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # TODO: Implement full pipeline
    print(f"Config loaded: {config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    main(args.config)
