# RiskFormer: Advanced Financial Time-Series Intelligence

RiskFormer is a modular Transformer-based system for financial time-series analysis and multi-horizon probabilistic forecasting. This project demonstrates rigorous ML engineering, quantitative finance, and Transformer architecture design applied to real financial data.

**Project Type:** Advanced capstone + portfolio project + learning platform

## Core Philosophy

- **Teach First:** Each component is explained conceptually before implementation
- **Incremental Development:** Build one phase at a time; understand each layer
- **Production Quality:** Strong engineering practices, testing, clean architecture
- **No Artificial Simplification:** Use sophisticated models where warranted
- **CPU-First:** Designed to run on standard hardware without GPU
- **Reproducible:** All experiments logged, seeded, and replicable

## Project Goals

Do **not** claim this system can reliably predict markets or generate risk-free profits.

**Do** build a technically rigorous system that eventually supports:

- Multi-asset financial time-series modeling
- Multi-horizon return forecasting
- Volatility forecasting
- Probabilistic forecasting with uncertainty estimation
- Market regime detection
- Anomaly detection
- Risk-aware portfolio analysis
- Backtesting with proper leakage prevention
- Model calibration and uncertainty quantification
- Explainability and ablation studies

## Core Modeling Problem

Given historical market information **X_(t-L:t)**, estimate the future distribution:

```
P(y_(t+h) | X_(t-L:t))
```

where:
- **y** may be: future return, volatility, quantiles, regime probability, or directional sign
- **h** is the forecast horizon (days/weeks/months)
- **L** is the lookback window (historical context)
- The model learns representations of financial markets, not just next-tick prediction

## Data Architecture

**Conceptual tensor shape:**
```
X ∈ ℝ^(T × N × F)

T = time steps (days)
N = assets (stocks, indices, etc.)
F = features (OHLCV, technical indicators, macro data)
```

**Initial dataset:** US equities (S&P 500-style universe)
- Free accessible data from yfinance or similar
- Daily OHLCV (open, high, low, close, volume)
- Multiple years of historical data
- Macro indicators (VIX, broad indices as context)

**Raw features:**
- Open, High, Low, Close, Adjusted Close, Volume

**Derived features:**
- Simple and log returns
- Rolling volatility (20-day, 60-day)
- Moving averages (5-day, 20-day, 60-day)
- Momentum indicators (RSI, MACD)
- Volume changes and ratios
- Rolling correlation / covariance

**Future extensibility:** CRSP institutional data, Compustat fundamentals, news embeddings, alternative data, intraday tick data, order book microstructure.

## Temporal Leakage Prevention

Financial ML has severe leakage and evaluation pitfalls. This system **explicitly prevents** temporal leakage:

✅ **Correct:**
- Chronological train/val/test split
- Fit normalization (scaling) on training data only
- Roll that normalization forward to val/test
- Feature engineering computed on training data only
- Rolling windows respect temporal order
- Walk-forward validation for evaluation

❌ **Forbidden:**
- Random train/test split
- Normalization using full dataset
- Future values in feature computation
- Information leakage through preprocessing
- Look-ahead bias in backtesting

**Pipeline:**
```
Raw data
  → Cleaning
  → Feature Engineering
  → Chronological split (train | val | test)
  → Fit preprocessing on train only
  → Rolling window creation (respecting time)
  → Model training
  → Walk-forward validation
  → Out-of-sample evaluation
  → Backtesting
```

## Model Architecture (Long-Term Vision)

```
Market Data
    ↓
Data Cleaning & QA
    ↓
Feature Engineering
    ↓
Leakage-Safe Preprocessing
    ↓
Temporal / Asset Embeddings
    ↓
RiskFormer Transformer Backbone
    ├─ Causal self-attention
    ├─ Multi-head attention
    ├─ Feed-forward networks
    ├─ Residual connections
    └─ Dropout / normalization
    ↓
Multi-Task Prediction Heads
    ├─ Return prediction
    ├─ Volatility forecasting
    ├─ Quantile regression
    ├─ Regime classification
    └─ Uncertainty estimation
    ↓
Probabilistic Forecasts P(y | X)
    ↓
Evaluation & Calibration
    ├─ Forecasting metrics (MAE, RMSE, directional accuracy, correlation)
    ├─ Probabilistic metrics (calibration, coverage)
    └─ Risk metrics (VaR, CVaR, max drawdown)
    ↓
Backtesting & Risk Analysis
    ↓
Advanced Extensions
    ├─ Multi-scale temporal modeling
    ├─ Cross-asset attention
    ├─ Market regime conditioning
    ├─ Explainability / ablation
    └─ Ensemble methods
```

## Development Roadmap

### Phase 0: Repository Architecture ✓ (Current)
- [x] Directory structure
- [x] Configuration system (YAML)
- [x] Testing infrastructure
- [x] Documentation (README, roadmap)
- [x] Git initialization and first commit

### Phase 1: Data Ingestion
- [ ] Market data downloaders (yfinance, etc.)
- [ ] Data validation and QA
- [ ] Caching and versioning
- [ ] Tests for data consistency

### Phase 2: Data Cleaning & Feature Engineering
- [ ] Missing value handling
- [ ] Outlier detection and handling
- [ ] Technical indicator computation
- [ ] Feature engineering pipeline
- [ ] Tests for leakage prevention

### Phase 3: Leakage-Safe Normalization
- [ ] Fit on training data only
- [ ] Roll forward to val/test
- [ ] Handle missing values safely
- [ ] Numerical stability checks
- [ ] Tests for temporal correctness

### Phase 4: Rolling Window Dataset
- [ ] Chronological window creation
- [ ] Asset/feature alignment
- [ ] Batch construction
- [ ] PyTorch Dataset/DataLoader integration
- [ ] Tests for temporal order

### Phase 5: Embeddings
- [ ] Temporal positional encoding
- [ ] Asset embeddings
- [ ] Feature embeddings
- [ ] Initialization and training
- [ ] Tests for embedding properties

### Phase 6: Causal Self-Attention
- [ ] Attention mechanism from scratch
- [ ] Query, Key, Value projections
- [ ] Scaling and softmax
- [ ] Causal masking (no future information)
- [ ] Tests for masking correctness

### Phase 7: Transformer Backbone
- [ ] Multi-head attention blocks
- [ ] Feed-forward networks
- [ ] Residual connections
- [ ] Layer normalization
- [ ] Dropout and regularization
- [ ] Tests for shape correctness

### Phase 8: Prediction Heads
- [ ] Return prediction head
- [ ] Volatility forecasting head
- [ ] Quantile regression head
- [ ] Regime classification head
- [ ] Tests for head correctness

### Phase 9: Probabilistic & Multi-Task Losses
- [ ] Gaussian NLL for return/volatility
- [ ] Pinball loss for quantile regression
- [ ] Cross-entropy for regime classification
- [ ] Multi-task weighted loss
- [ ] Tests for loss computation

### Phase 10: Training Pipeline
- [ ] Optimizer and learning rate scheduler
- [ ] Training loop with validation
- [ ] Checkpointing and early stopping
- [ ] Gradient clipping and warmup
- [ ] Tests for convergence

### Phase 11: Walk-Forward Validation
- [ ] Sequential split logic
- [ ] Model retraining
- [ ] Performance aggregation
- [ ] Proper backtest semantics
- [ ] Tests for leakage prevention

### Phase 12: Baseline Models
- [ ] Naive return prediction
- [ ] Historical mean/volatility
- [ ] Moving average forecast
- [ ] Linear regression
- [ ] XGBoost-style ensemble
- [ ] Tests for baseline correctness

### Phase 13: Evaluation & Calibration
- [ ] Forecasting metrics (MAE, RMSE, directional accuracy)
- [ ] Probabilistic calibration
- [ ] Correlation and Sharpe ratio
- [ ] Quantile coverage analysis
- [ ] Comparison against baselines

### Phase 14: Backtesting & Risk Analysis
- [ ] Portfolio construction from signals
- [ ] Transaction cost simulation
- [ ] Cumulative return and Sharpe ratio
- [ ] Maximum drawdown and Sortino
- [ ] Risk attribution

### Phase 15: Advanced Architecture
- [ ] Multi-scale temporal modeling
- [ ] Cross-asset attention
- [ ] Regime-conditional attention
- [ ] Uncertainty calibration
- [ ] Feature/attention visualization
- [ ] Robustness testing

### Phase 16: Optimization & Documentation
- [ ] Ablation studies
- [ ] Hyperparameter search
- [ ] Model compression (where appropriate)
- [ ] Final documentation
- [ ] Reproducibility checklist

## Key Design Decisions

### Why Transformers for Finance?

1. **Temporal dependencies:** Attention learns variable-length dependencies
2. **Interpretability:** Attention weights reveal what the model considers
3. **Parallelization:** Can process long sequences efficiently
4. **Flexibility:** Extensible to multi-asset, multi-horizon, multi-task
5. **Causal masking:** Natural way to prevent look-ahead bias

### Why Multi-Task Learning?

Financial properties are interconnected:
- High volatility correlates with different return distributions
- Market regimes affect both returns and risk
- Shared representations improve generalization
- Uncertainty is a first-class prediction target

### Why Probabilistic Forecasts?

- Risk management requires uncertainty quantification
- Point predictions are insufficient for portfolio optimization
- Quantile regression captures tail risk
- NLL losses enable calibration analysis
- Aleatoric and epistemic uncertainty are both important

### Why CPU-First Development?

1. Reproducibility: Everyone has a CPU
2. Focus: Encourages algorithm efficiency over raw compute
3. Accessibility: Can run locally without cloud costs
4. Learning: Understand bottlenecks, not hide them in hardware

## Project Structure

```
riskformer/
├── configs/
│   └── base.yaml                    # Hyperparameters and settings
├── data/
│   ├── __init__.py
│   ├── ingestion.py                 # Download and cache market data
│   ├── cleaning.py                  # QA and outlier handling
│   ├── features.py                  # Technical indicators
│   ├── normalization.py             # Leakage-safe scaling
│   └── dataset.py                   # PyTorch Dataset/DataLoader
├── models/
│   ├── __init__.py
│   ├── embeddings.py                # Temporal, asset, feature embeddings
│   ├── attention.py                 # Causal self-attention
│   ├── transformer.py               # Transformer blocks and backbone
│   ├── regime.py                    # Regime detection (future)
│   └── heads.py                     # Prediction heads (return, vol, quantile, regime)
├── losses/
│   ├── __init__.py
│   ├── gaussian_nll.py              # Gaussian negative log-likelihood
│   ├── quantile.py                  # Pinball loss for quantiles
│   └── multitask.py                 # Multi-task weighted loss
├── training/
│   ├── __init__.py
│   ├── trainer.py                   # Training loop
│   └── scheduler.py                 # Learning rate scheduling
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # Forecasting and risk metrics
│   ├── walk_forward.py              # Walk-forward validation
│   └── calibration.py               # Probability calibration
├── baselines/
│   ├── __init__.py
│   ├── naive.py                     # Baseline models
│   ├── linear.py
│   └── xgboost.py
├── backtesting/
│   ├── __init__.py
│   └── portfolio.py                 # Portfolio construction and analysis
├── experiments/                      # Experimental notebooks and scripts
├── notebooks/                        # Jupyter notebooks for exploration
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_training.py
│   └── test_evaluation.py
├── train.py                         # Main training script
├── evaluate.py                      # Main evaluation script
├── .gitignore
├── README.md                        # This file
└── pyproject.toml                   # Project metadata and dependencies
```

## Dependencies

Minimal, intentional dependencies:

- **numpy:** Numerical computing
- **pandas:** Time-series data manipulation
- **torch:** Deep learning
- **scikit-learn:** Preprocessing, classical baselines
- **pyyaml:** Configuration management

**Development:**
- **pytest:** Testing
- **pytest-cov:** Coverage reporting

No high-level frameworks that hide implementation details (yet). All major components (attention, Transformer, losses, datasets) are written from scratch to maximize learning.

## Getting Started

### Installation

```bash
git clone <repository-url>
cd riskformer
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Configuration

Edit `configs/base.yaml` to adjust:
- Assets, features, dates
- Normalization method
- Transformer architecture (d_model, n_heads, n_layers)
- Training hyperparameters
- Loss weights for multi-task learning

### Development

Each phase follows this process:

1. **Conceptual explanation:** Why does the component exist? What's the mental model?
2. **Mathematical foundation:** Key equations and tensor shapes
3. **Design decisions:** Why build it this way?
4. **Implementation:** Clean, readable code with inline documentation
5. **Testing:** Shape tests, edge cases, leakage prevention tests
6. **Verification:** Ensure correctness before moving forward
7. **Documentation:** Update README and inline comments
8. **Commit:** "phase X: <component>"

## Testing Philosophy

Tests focus on:
- **Shape correctness:** Output tensors have expected dimensions
- **Leakage prevention:** No future information leaks backward
- **Temporal order:** Chronological behavior is preserved
- **Numerical stability:** No NaN/Inf in normal operation
- **Determinism:** Same seed = same behavior (where applicable)

Example test structure:
```python
def test_attention_shape():
    # Verify attention output shape matches input
    pass

def test_no_future_leakage():
    # Verify causal mask prevents attending to future
    pass

def test_dataset_chronological():
    # Verify rolling windows respect time
    pass
```

## Evaluation Standards

Never present results without baselines.

**Forecasting metrics:**
- MAE, RMSE (point estimate error)
- Directional accuracy (signs correct)
- Correlation with realized values
- Calibration (predicted uncertainty matches actual error)

**Risk metrics:**
- Volatility forecast error
- VaR and CVaR
- Maximum drawdown

**Portfolio metrics (backtesting):**
- Cumulative return
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Transaction cost impact
- Turnover and rebalancing frequency

## Future Extensions

Once the core is solid, consider:

1. **Multi-scale temporal modeling:** Different attention windows for different frequencies
2. **Cross-asset attention:** Learn correlations between assets
3. **Market regime conditioning:** Separate forecasts by market regime
4. **Asset embeddings:** Learned representations of different stocks
5. **Intraday data:** Minute-level predictions with longer history
6. **Sentiment embeddings:** News and social media integration
7. **Explainability:** Attention visualization, SHAP values, feature importance
8. **Ensembles:** Combine multiple models for robustness
9. **Adaptive learning:** Non-stationary market adaptation
10. **Portfolio-aware objectives:** Optimize for portfolio-level metrics

## Learning Objectives

By completing this project, you should understand:

- **Transformer architecture:** Attention, multi-head attention, causal masking, feedforward, residual connections
- **Financial ML:** Leakage prevention, walk-forward validation, probability calibration
- **Quantitative finance:** Returns, volatility, VaR, portfolio construction, backtesting
- **Production ML:** Configuration, testing, reproducibility, baseline comparison, ablation studies
- **PyTorch:** Custom models, datasets, loss functions, training loops
- **Software engineering:** Modular design, testing, documentation, version control

## Disclaimer

This project demonstrates ML engineering applied to finance. It does **not**:
- Guarantee market returns
- Claim to have discovered a profitable strategy
- Suggest that past performance predicts future results
- Replace professional financial advice

All backtesting results are **ex-post** (fitted to historical data) and subject to:
- Overfitting
- Regime change
- Transaction cost underestimation
- Survivorship bias
- Look-ahead bias (if implemented incorrectly)

Use this system for **learning** and **research**, not production trading.

## License

MIT

## Contact

Thomas H. (thomas.rh@northeastern.edu)
