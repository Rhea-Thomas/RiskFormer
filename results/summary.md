# RiskFormer — out-of-sample results (Jan 2023 – Dec 2025, SPY, QQQ, IWM)

Walk-forward: every model is refit on all data before each quarter, then forecasts that quarter.

## Volatility & probabilistic forecasts

| Model | Vol QLIKE ↓ | Vol RMSE ↓ | Vol corr ↑ | NLL ↓ | z-std (→1) | 90% cover | VaR95 breach (→5%) |
|---|---|---|---|---|---|---|---|
| RiskFormer | 0.501 | **0.551** | 0.495 | 1.514 | 1.14 | 89.3% | 4.4% |
| RiskFormer (static) | **0.436** | 0.562 | **0.497** | **1.491** | 1.03 | 91.0% | 3.5% |
| zero+rolling20 | 0.542 | 0.646 | 0.350 | 1.554 | 1.10 | 88.8% | 5.5% |
| zero+ewma | 0.487 | 0.619 | 0.372 | 1.532 | 1.05 | 89.8% | 5.5% |
| zero+garch | 0.446 | 0.594 | 0.438 | 1.516 | 1.02 | 90.2% | 5.2% |
| zero+har | 0.596 | 0.579 | 0.412 | 1.572 | 1.24 | 84.3% | 7.7% |
| persistence | 0.487 | 0.619 | 0.372 | 1.979 | 1.41 | 76.2% | 10.9% |
| hist_mean | 0.487 | 0.619 | 0.372 | 1.530 | 1.05 | 89.9% | 5.7% |
| moving_avg | 0.487 | 0.619 | 0.372 | 1.553 | 1.07 | 89.0% | 5.9% |
| ridge | 0.487 | 0.619 | 0.372 | 1.543 | 1.06 | 89.3% | 6.0% |
| gbm | 0.487 | 0.619 | 0.372 | 1.526 | 1.04 | 90.0% | 5.9% |

## Return forecasts

| Model | MAE ↓ | RMSE ↓ | Directional acc. | Corr | Rank IC |
|---|---|---|---|---|---|
| RiskFormer | 0.851 | 1.192 | 55.3% | 0.084 | 0.049 |
| RiskFormer (static) | 0.848 | 1.190 | 55.7% | 0.098 | 0.073 |
| zero+rolling20 | 0.856 | 1.197 | 0.0% | 0.000 | 0.029 |
| zero+ewma | 0.856 | 1.197 | 0.0% | 0.000 | 0.029 |
| zero+garch | 0.856 | 1.197 | 0.0% | 0.000 | 0.029 |
| zero+har | 0.856 | 1.197 | 0.0% | 0.000 | 0.029 |
| persistence | 1.232 | 1.717 | 50.1% | -0.035 | -0.018 |
| hist_mean | 0.851 | 1.195 | 55.7% | 0.005 | 0.017 |
| moving_avg | 0.879 | 1.226 | 51.6% | -0.044 | -0.036 |
| ridge | 0.863 | 1.200 | 51.5% | 0.052 | 0.015 |
| gbm | 0.847 | 1.177 | 54.0% | 0.177 | 0.041 |

## Backtests (net of 10 bps costs, 10% vol target)

| Forecaster | Strategy | Sharpe | CAGR | Ann. vol | Max DD | Turnover/day |
|---|---|---|---|---|---|---|
| Benchmark | buy_and_hold | 1.27 | 23.2% | 17.6% | -21.8% | 0.00 |
| RiskFormer | vol_target | 1.00 | 17.3% | 17.5% | -21.2% | 0.13 |
| RiskFormer | signal | 0.43 | 5.3% | 14.6% | -21.4% | 0.48 |
| RiskFormer | sign | 0.41 | 5.0% | 14.4% | -19.9% | 0.39 |
| RiskFormer (static) | vol_target | 1.00 | 16.0% | 16.2% | -19.1% | 0.12 |
| RiskFormer (static) | signal | 0.51 | 6.9% | 15.3% | -18.9% | 0.34 |
| RiskFormer (static) | sign | 0.64 | 9.2% | 15.9% | -20.2% | 0.23 |
| zero+rolling20 | vol_target | 1.18 | 20.5% | 17.1% | -20.9% | 0.04 |
| zero+ewma | vol_target | 1.14 | 19.3% | 16.6% | -20.0% | 0.03 |
| zero+garch | vol_target | 1.11 | 18.3% | 16.4% | -19.6% | 0.07 |
| zero+har | vol_target | 1.13 | 22.0% | 19.2% | -23.1% | 0.08 |
| persistence | vol_target | 1.14 | 19.3% | 16.6% | -20.0% | 0.03 |
| persistence | signal | -2.25 | -28.1% | 14.2% | -64.5% | 1.20 |
| persistence | sign | -1.75 | -23.2% | 14.4% | -56.8% | 1.08 |
| hist_mean | vol_target | 1.14 | 19.3% | 16.6% | -20.0% | 0.03 |
| hist_mean | signal | 1.25 | 20.6% | 16.1% | -18.9% | 0.04 |
| hist_mean | sign | 1.14 | 19.3% | 16.6% | -20.0% | 0.03 |
| moving_avg | vol_target | 1.14 | 19.3% | 16.6% | -20.0% | 0.03 |
| moving_avg | signal | 0.18 | 1.6% | 14.6% | -16.5% | 0.23 |
| moving_avg | sign | 0.26 | 2.8% | 15.0% | -19.8% | 0.19 |
| ridge | vol_target | 1.14 | 19.3% | 16.6% | -20.0% | 0.03 |
| ridge | signal | -1.63 | -21.1% | 13.9% | -51.7% | 0.95 |
| ridge | sign | -1.26 | -17.1% | 14.1% | -44.0% | 0.83 |
| gbm | vol_target | 1.14 | 19.3% | 16.6% | -20.0% | 0.03 |
| gbm | signal | 0.41 | 5.0% | 14.4% | -20.7% | 0.55 |
| gbm | sign | 0.63 | 8.5% | 14.8% | -23.3% | 0.42 |
