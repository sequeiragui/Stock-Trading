# S&P 500 Directional Forecasting

Trading algorithm for a S&P 500 using DL and probabilistic models on macroeconomic and technical indicators.
Do not use it if you want to make money :)
---

## Project Overview

This project explores whether machine learning models can predict weekly S&P 500 direction and whether those predictions translate into profitable trading strategies.

It compares several architectures:

- Linear Regression (baseline)
- LSTM
- Transformer Encoder
- Mixture of Experts (MoE) combining LSTM, Transformer, and Kalman expert
- Learnable Kalman Filter (neural-augmented)

The models are evaluated both on:

- Directional accuracy (up/down correctly predicted)
- Risk-adjusted performance via Sharpe ratio of a simple trading strategy

---

## Data

The dataset combines:

- **Technical indicators** computed from the S&P 500 index (Yahoo Finance):
  - SMA/EMA (10, 50)
  - ROC (Rate of Change)
  - RSI (14)
  - Bollinger Bands
  - 20-day volatility

- **Macroeconomic indicators** (FRED):
  - Federal funds rate
  - Unemployment rate
  - 10Y Treasury yield
  - USD broad index
  - Crude oil price
  - Gold price
  - MOVE index (bond volatility)
  - VIX (equity volatility)

Key preprocessing steps:

- ~8 years of data (2016–2024), aggregated to **weekly** frequency  
- **Log returns** used as target  
- **Sliding window** of 20 weeks as model input  
- Train / validation / test split: **70% / 15% / 15%**, with proper time ordering and no leakage  

> **Note:** Raw data is not tracked here for size/licensing reasons. You can either:
> - download from Yahoo Finance and FRED, or  
> - adapt the data loading cells in the notebooks to your own data paths.

---

## Repository Structure

```text
Stock-Trading/
├── data/           # Data directory (raw/processed – not fully tracked)
├── models/         # Model and experiment notebooks/scripts
├── .gitignore
└── README.md
