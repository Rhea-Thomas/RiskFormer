"""
High-level helpers: build a RiskFormer + Trainer from config and fit it.

Shared by train.py (one fit) and evaluate.py (walk-forward refits).
"""

import logging
import random

import numpy as np
import torch

from data.pipeline import SplitData, to_loader
from losses.multitask import RiskFormerLoss
from models.riskformer import RiskFormer
from training.trainer import Trainer, warmup_cosine_schedule

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loss(config: dict) -> RiskFormerLoss:
    w = dict(config["losses"])
    return RiskFormerLoss(quantiles=config["heads"]["quantiles"], **w)


def fit_riskformer(
    config: dict,
    train: SplitData,
    val: SplitData,
    max_epochs: int = None,
    init_state: dict = None,
    lr_scale: float = 1.0,
    log_every: int = 1,
):
    """
    Train a RiskFormer on `train`, early-stopping on `val`.

    Args:
        init_state: optional state_dict to warm-start from (walk-forward refits)
        lr_scale: multiply the configured LR (smaller for fine-tuning)

    Returns:
        (model, trainer)
    """
    tc = config["training"]
    device = config["system"]["device"]
    max_epochs = max_epochs or tc["max_epochs"]

    n_assets, n_features = train.X.shape[2], train.X.shape[3]
    model = RiskFormer.from_config(config, n_features=n_features, n_assets=n_assets)
    if init_state is not None:
        model.load_state_dict(init_state)

    loss_fn = build_loss(config).to(device)
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=tc["learning_rate"] * lr_scale, weight_decay=tc["weight_decay"]
    )

    train_loader = to_loader(train, tc["batch_size"], shuffle=True)
    val_loader = to_loader(val, 256) if val is not None else None
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine_schedule(
        optimizer,
        warmup_steps=tc.get("warmup_epochs", 0) * steps_per_epoch if init_state is None else 0,
        total_steps=max_epochs * steps_per_epoch,
    )

    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        early_stopping_patience=tc["early_stopping_patience"],
        grad_clip=tc.get("grad_clip", 1.0),
        scheduler=scheduler,
        log_every=log_every,
    )
    trainer.train(train_loader, val_loader, max_epochs=max_epochs)
    return model, trainer


@torch.no_grad()
def predict(model, split: SplitData, device="cpu", batch_size=256) -> dict:
    """Run the model over a split; returns numpy arrays keyed like the model output."""
    model.eval()
    outs = []
    X = torch.from_numpy(split.X)
    for i in range(0, len(X), batch_size):
        o = model(X[i : i + batch_size].to(device))
        outs.append({k: v.cpu().numpy() for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
