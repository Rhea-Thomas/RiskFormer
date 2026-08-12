"""
Main evaluation script for RiskFormer.

Usage:
    python evaluate.py --checkpoint checkpoints/model.pt --config configs/base.yaml

Workflow:
    1. Load trained model and configuration
    2. Load test data (using train normalization statistics)
    3. Generate predictions
    4. Compute forecasting metrics
    5. Run walk-forward validation
    6. Compare against baselines
    7. Generate report
"""

import argparse
import yaml
import torch


def main(checkpoint_path, config_path):
    """
    Main evaluation entry point.

    Args:
        checkpoint_path: path to trained model checkpoint
        config_path: path to configuration file
    """
    # Load config and checkpoint
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # TODO: Implement full evaluation pipeline
    print(f"Config: {config}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.config)
