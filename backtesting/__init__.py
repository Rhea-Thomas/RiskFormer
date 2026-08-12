"""
Backtesting and portfolio analysis.

Converts model predictions into trading signals and simulates P&L.

Pipeline:
  Predictions → Signal generation → Portfolio construction
             → Transaction costs → Realized P&L → Metrics

Key:
  - Do NOT separate model evaluation from strategy evaluation
  - A model can have great forecasting metrics but bad portfolio metrics
  - Account for transaction costs, slippage, and market impact
  - Proper backtesting is complex; keep it simple to avoid pitfalls
"""
