"""
Tests for training loop.

Focus:
  - Training convergence on toy dataset
  - Validation loss computation
  - Checkpoint saving/loading
  - Early stopping behavior
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from losses.multitask import RiskFormerLoss
from models.riskformer import RiskFormer
from training.trainer import Trainer, warmup_cosine_schedule


def _toy_regression(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 4, generator=g)
    y = X @ torch.tensor([1.0, -2.0, 0.5, 0.0]) + 0.05 * torch.randn(n, generator=g)
    return TensorDataset(X, y.unsqueeze(-1))


def test_training_convergence_toy_data():
    """Verify training loop converges on simple dataset."""
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, opt, lambda out, y: nn.functional.mse_loss(out, y), early_stopping_patience=50)
    history = trainer.train(DataLoader(_toy_regression(), batch_size=32, shuffle=True),
                            DataLoader(_toy_regression(seed=1), batch_size=64), max_epochs=40)
    assert history[-1]["val_loss"] < 0.01
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_riskformer_overfits_tiny_batch():
    """The full model + multi-task loss can drive training loss down."""
    torch.manual_seed(0)
    B, L, A, F = 16, 10, 2, 5
    X = torch.randn(B, L, A, F)
    ret = X[:, -1, :, 0]  # target is a visible feature -> learnable
    y = torch.stack([ret, ret.abs() + 0.5, (ret > 0).float()], -1)
    model = RiskFormer(F, A, d_model=32, n_heads=4, n_layers=1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    trainer = Trainer(model, opt, RiskFormerLoss(direction_weight=0.1))
    loader = DataLoader(TensorDataset(X, y), batch_size=B)
    hist = trainer.train(loader, None, max_epochs=150)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"] - 1.0


def test_early_stopping_restores_best():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # nothing changes -> val never improves after epoch 1
    trainer = Trainer(model, opt, lambda o, y: nn.functional.mse_loss(o, y), early_stopping_patience=3)
    hist = trainer.train(DataLoader(_toy_regression()), DataLoader(_toy_regression(seed=1)), max_epochs=50)
    assert len(hist) == 4  # 1 best epoch + 3 without improvement
    assert trainer.best_state is not None


def test_checkpoint_save_load(tmp_path):
    """Verify checkpoint saving and loading works."""
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, lambda o, y: nn.functional.mse_loss(o, y))
    trainer.train(DataLoader(_toy_regression()), None, max_epochs=2)
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path, note="hello")

    model2 = nn.Linear(4, 1)
    trainer2 = Trainer(model2, torch.optim.Adam(model2.parameters()), trainer.loss_fn)
    ckpt = trainer2.load_checkpoint(path)
    assert ckpt["note"] == "hello"
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
    assert len(trainer2.history) == 2


def test_non_finite_loss_raises():
    model = nn.Linear(4, 1)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), lambda o, y: o.sum() * float("nan"))
    with pytest.raises(FloatingPointError):
        trainer.train_epoch(DataLoader(_toy_regression()))


def test_warmup_cosine_schedule():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = warmup_cosine_schedule(opt, warmup_steps=10, total_steps=110, min_lr_ratio=0.0)
    lrs = []
    for _ in range(110):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1)
    assert max(lrs) == pytest.approx(1.0)
    assert lrs[-1] < 0.01
