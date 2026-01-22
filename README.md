# 📈 Deep Learning Stock Price Predictor

A comparative study of deep learning architectures (**RNN, LSTM, GRU, and Transformers**) for time-series forecasting of stock prices. This project implements a complete end-to-end pipeline, from data ingestion using `yfinance` to model training and evaluation using PyTorch.

## 📌 Project Overview
Predicting stock market trends is notoriously difficult due to volatility and non-stationary data. This project investigates whether modern attention-based mechanisms (Transformers) outperform traditional recurrent networks (RNNs, LSTMs, GRUs) in short-term price forecasting.

We test these models on three distinct stock profiles:
* **NVDA** (High Volatility / Tech)
* **KO** (Low Volatility / Consumer Stable)
* **JPM** (Moderate Volatility / Banking)

## 🚀 Key Features
* **Multi-Model Architecture:** Modular implementation of Vanilla RNN, LSTM, GRU, and a custom Time-Series Transformer.
* **Advanced Feature Engineering:** Incorporates technical indicators like **RSI** (Momentum), **MACD** (Trend), and **SMA** (Smoothing) alongside raw price data.
* **Sliding Window Preprocessing:** Converts time-series data into supervised learning sequences (Lookback: 60 days).
* **Robust Training Loop:** Includes Early Stopping, Adaptive Learning Rate (Adam), and Validation Monitoring.
* **Visual Analysis:** Generates plots comparing Predicted vs. Actual prices to visually assess model performance.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Data Manipulation:** Pandas, NumPy
* **Data Source:** `yfinance` API
* **Technical Analysis:** `ta` library
* **Visualization:** Matplotlib, Seaborn

## 🧠 Model Architectures

### 1. Recurrent Models (RNN / LSTM / GRU)
These models process data sequentially.
* **Input:** Sequence of 60 days × 8 Features.
* **Hidden Layer:** Captures temporal dependencies (Short-term memory).
* **Output:** The hidden state of the last time step ($t=60$) is passed to a Linear Decoder to predict $t=61$.

### 2. Time-Series Transformer
Adapts the "Attention is All You Need" architecture for regression.
* **Positional Encoding:** Fixed Sinusoidal waves injected to preserve temporal order.
* **Encoder:** Multi-Head Self-Attention allows the model to relate any day in the window to any other day, regardless of distance.
* **Decoder Head:** A linear projection of the final context vector to the target price.

## 📂 Project Structure
```bash
├── FinalDLProject2026.ipynb   # Main Jupyter Notebook (Training & Eval)
├── README.md                  # Project Documentation
└── requirements.txt           # Dependencies
