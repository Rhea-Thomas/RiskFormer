"""
Main training script for RiskFormer.

Usage:
    python train.py --config configs/base.yaml

Workflow:
    1. Load configuration
    2. Download market data (cached under data/cache/)
    3. Clean and engineer features
    4. Create train/val/test splits (chronologically, with embargo)
    5. Fit normalization on train only
    6. Create rolling-window dataset
    7. Initialize model, optimizer, loss function
    8. Train with validation + early stopping
    9. Save checkpoint and training curves
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from data.pipeline import build_splits, load_panel_from_config
from models.riskformer import count_parameters
from training.fit import fit_riskformer, set_seed

logger = logging.getLogger("riskformer.train")


def plot_history(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    comps = ["loss", "return", "volatility", "quantile", "direction"]
    fig, axes = plt.subplots(1, len(comps), figsize=(4 * len(comps), 3.2))
    for ax, c in zip(axes, comps):
        ax.plot(epochs, [h[f"train_{c}"] for h in history], label="train")
        if f"val_{c}" in history[0]:
            ax.plot(epochs, [h[f"val_{c}"] for h in history], label="val")
        ax.set_title(c if c != "loss" else "total loss")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(config_path, checkpoint="checkpoints/riskformer.pt", results_dir="results"):
    """
    Main training entry point.

    Args:
        config_path: path to YAML configuration file
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    set_seed(config["system"]["seed"])

    panel = load_panel_from_config(config)
    train, val, test, _ = build_splits(config, panel)
    logger.info(
        f"train {len(train)} ({train.dates[0].date()}→{train.dates[-1].date()}), "
        f"val {len(val)}, test {len(test)}"
    )

    model, trainer = fit_riskformer(config, train, val)
    logger.info(f"Parameters: {count_parameters(model):,}  best val loss {trainer.best_val_loss:.4f}")

    trainer.save_checkpoint(checkpoint, config=config)
    logger.info(f"Checkpoint saved to {checkpoint}")

    out = Path(results_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    with open(out / "training_history.json", "w") as f:
        json.dump(trainer.history, f, indent=2)
    plot_history(trainer.history, out / "figures" / "training_curves.png")
    return model, trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ("models", "data"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="Path to configuration file")
    parser.add_argument("--checkpoint", default="checkpoints/riskformer.pt")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.results)
