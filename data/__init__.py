"""
Data module for RiskFormer.

Handles market data ingestion, cleaning, feature engineering,
normalization (with leakage prevention), and dataset construction.

Pipeline:
  Raw data → Cleaning → Features → Normalization → Dataset → DataLoader
"""
