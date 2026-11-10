"""
Tests for loss functions.

Focus:
  - Loss computation correctness
  - Numerical stability (no NaN/Inf)
  - Gradient flow
  - Multi-task weighting
"""

import math

import pytest
import torch

from losses.gaussian_nll import GaussianNLL
from losses.multitask import MultiTaskLoss, RiskFormerLoss
from losses.quantile import QuantileLoss


def test_gaussian_nll_basic():
    """Verify Gaussian NLL computation."""
    nll = GaussianNLL()
    mu = torch.tensor([[0.0]])
    y = torch.tensor([[2.0]])
    log_sigma = torch.tensor([[math.log(2.0)]])
    # log σ + 0.5 * ((y-μ)/σ)^2 = log 2 + 0.5
    assert nll(mu, log_sigma, y).item() == pytest.approx(math.log(2.0) + 0.5, rel=1e-6)


def test_gaussian_nll_matches_torch():
    torch.manual_seed(0)
    mu, y = torch.randn(64, 3), torch.randn(64, 3)
    sigma = torch.rand(64, 3) + 0.2
    ours = GaussianNLL()(mu, torch.log(sigma), y)
    ref = torch.nn.functional.gaussian_nll_loss(mu, y, sigma**2, full=False)
    assert ours.item() == pytest.approx(ref.item(), rel=1e-5)


def test_gaussian_nll_minimized_at_true_sigma():
    """For fixed error e, NLL in σ is minimized at σ = |e|."""
    nll = GaussianNLL(reduction="none")
    e = torch.tensor(1.5)
    sigmas = torch.linspace(0.5, 3.0, 251)
    vals = torch.stack([nll(torch.zeros(()), torch.log(s), e) for s in sigmas])
    assert sigmas[vals.argmin()].item() == pytest.approx(1.5, abs=0.02)


def test_gaussian_nll_stable_for_extreme_sigma():
    nll = GaussianNLL()
    out = nll(torch.zeros(4), torch.tensor([-50.0, -10.0, 10.0, 50.0]), torch.ones(4))
    assert torch.isfinite(out)


def test_quantile_loss_basic():
    """Verify pinball loss computation."""
    ql = QuantileLoss([0.1, 0.9], reduction="none")
    y = torch.tensor([[1.0]])
    q = torch.tensor([[[0.0, 0.0]]])  # both quantiles below y by 1.0
    out = ql(q, y)
    # under-prediction: τ * (y - q)
    assert out[0, 0, 0].item() == pytest.approx(0.1)
    assert out[0, 0, 1].item() == pytest.approx(0.9)


def test_quantile_loss_recovers_empirical_quantile():
    """Minimizing pinball loss over a constant recovers the sample quantile."""
    torch.manual_seed(0)
    y = torch.randn(4000, 1)
    q = torch.zeros(1, 1, 3, requires_grad=True)
    ql = QuantileLoss([0.05, 0.5, 0.95])
    opt = torch.optim.Adam([q], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        ql(q.expand(4000, 1, 3), y).backward()
        opt.step()
    expected = torch.quantile(y, torch.tensor([0.05, 0.5, 0.95]))
    assert torch.allclose(q.detach().squeeze(), expected, atol=0.08)


def test_quantile_loss_rejects_wrong_shape():
    with pytest.raises(ValueError):
        QuantileLoss([0.05, 0.95])(torch.zeros(2, 1, 3), torch.zeros(2, 1))


def test_multitask_loss_weighting():
    """Verify multi-task loss weights are applied correctly."""
    mt = MultiTaskLoss(return_weight=1.0, volatility_weight=0.5, quantile_weight=0.25, regime_weight=0.0)
    parts = {
        "return": torch.tensor(2.0),
        "volatility": torch.tensor(4.0),
        "quantile": torch.tensor(8.0),
        "regime": torch.tensor(100.0),  # weight 0 -> ignored
    }
    assert mt(parts).item() == pytest.approx(2.0 + 2.0 + 2.0)


def test_multitask_learned_weights_have_gradients():
    mt = MultiTaskLoss(learn_weights=True)
    loss = mt({"return": torch.tensor(2.0), "volatility": torch.tensor(1.0)})
    loss.backward()
    assert mt.log_vars["return"].grad is not None
    # d/ds [exp(-s)·w·L + s] at s=0 = 1 - w·L
    assert mt.log_vars["return"].grad.item() == pytest.approx(1 - 1.0 * 2.0)


def test_multitask_rejects_unknown_task():
    with pytest.raises(KeyError):
        MultiTaskLoss()({"typo": torch.tensor(1.0)})


def test_riskformer_loss_gradients_flow():
    torch.manual_seed(0)
    B, A = 8, 3
    raw = torch.randn(B, A, 7, requires_grad=True)
    out = {
        "mu": raw[..., 0],
        "sigma": torch.nn.functional.softplus(raw[..., 1]) + 1e-3,
        "quantiles": torch.sort(raw[..., 2:5], dim=-1).values,
        "direction_logit": raw[..., 5],
    }
    y = torch.stack([torch.randn(B, A), torch.rand(B, A) + 0.5, (torch.rand(B, A) > 0.5).float()], -1)
    total, parts = RiskFormerLoss(direction_weight=0.1)(out, y)
    total.backward()
    assert torch.isfinite(total)
    assert set(parts) == {"return", "volatility", "quantile", "direction"}
    assert raw.grad[..., :6].abs().sum() > 0
