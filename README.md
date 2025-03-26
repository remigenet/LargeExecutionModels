# LEMS: Large Execution Models for VWAP/TWAP Optimization

This repository provides the full implementation of **Large Execution Models (LEMS)**, a deep learning framework for optimal VWAP/TWAP execution across multiple assets and market conditions. LEMS leverages modern Keras3 tooling and is compatible with **TensorFlow**, **JAX**, and **PyTorch** backends. It integrates time-aware architectures, multi-objective allocation, and efficient fused layers to offer scalable and expressive modeling for high-performance trading execution.

---

## 📦 Installation

Install via pip:

```bash
pip install .
```

Or using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

**Requirements:**
- Python 3.9–3.12
- `keras>=3.0.0`
- `tkan`

Optional (for training and testing):
- `jax`, `torch`, or `tensorflow`
- `numpy`, `pandas`, `matplotlib` (for data and visualization)

---

## 🚀 Features

- **Backend-agnostic** with full support for `JAX`, `TensorFlow`, and `PyTorch`.
- **Flexible target modeling**: simultaneous optimization of VWAP/TWAP for both volume and notional allocation.
- **FusedMLP layer** for highly parallelized MLP computation across model variants (volume/notional, VWAP/TWAP, buy/sell).
- **Custom constraints and initializers** for learnable allocation curves with sum-to-one and positivity constraints.
- **Soft clipping mechanisms** to enforce trading limits while maintaining differentiability.
- **Multiple execution-aware loss/metrics** to track and optimize performance across multiple trading strategies.
- **Full training pipeline with evaluation-ready data preparation tools**.

---

## 🧠 Model Components

The main model, `LargeExecutionModel`, consists of:

- `EmbeddingLayer`: learns per-feature representations.
- `VariableSelectionNetwork`: soft attention over input variables.
- `TKAN` RNN layers: efficient temporal modeling with attention kernels.
- `MultiHeadAttention`: causal self-attention for cross-step dependency modeling.
- `FusedMLP`: efficiently computes multiple allocation predictions in parallel (buy/sell × VWAP/TWAP × volume/notional).
- `Custom constraints`: such as `PositiveSumToOneConstraint` and `EqualInitializer` for normalized allocation curves.

---

## 🧪 Example Usage

```python
from lems.models import LargeExecutionModel
from lems.loss import lem_loss
from lems.metrics import (
    buy_vwap_avg_imp, buy_twap_avg_imp,
    sell_vwap_avg_imp, sell_twap_avg_imp,
    buy_vwap_avg_risk, buy_twap_avg_risk,
    sell_vwap_avg_risk, sell_twap_avg_risk,
    unsigned_vwap_avg_risk, unsigned_twap_avg_risk
)

model = LargeExecutionModel(
    lookback=120,
    n_ahead=12,
    hidden_size=200,
    num_embedding=20,
    num_heads=8,
    fused_mlp_hidden_dim=100
)

model.compile(
    optimizer="adam",
    loss=lem_loss,
    metrics=[
        buy_vwap_avg_imp, buy_twap_avg_imp,
        sell_vwap_avg_imp, sell_twap_avg_imp,
        buy_vwap_avg_risk, buy_twap_avg_risk,
        sell_vwap_avg_risk, sell_twap_avg_risk,
        unsigned_vwap_avg_risk, unsigned_twap_avg_risk
    ]
)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64)
```

---

## 📊 Data Preparation

A full pipeline is included via `data_formatter.py`. Example:

```python
from lems.data_formater import full_generate, add_config_to_X, add_config_to_y

X_train, X_test, y_train, y_test, dates = full_generate(
    volumes_df, notionals_df,
    target_asset='AAPL',
    lookback=120,
    n_ahead=12,
    split_date='2022-01-01'
)

X_train = add_config_to_X(X_train, minimum_traded_per_period=0.0, maximum_traded_per_period=1.0)
y_train = add_config_to_y(y_train)
```

---

## ✅ Testing & Backend Compatibility

Tests are defined using `pytest` and validate:

- Consistency before and after saving the model
- Compatibility across **JAX**, **TensorFlow**, and **Torch** backends
- GPU compatibility on **NVIDIA** (currently not compatible with AMD GPUs due to fused ops)

---

## 📚 Example Results

Two notebooks in `example_and_results/` show results on:
- **Dow Jones equities** (e.g. AAPL, MSFT)
- **Cryptocurrencies** (e.g. BTC, ETH)

They demonstrate significant performance improvements over baseline TWAP/VWAP curves using LEMS.

---

## 📖 Key Files

- `models.py`: Full model definition
- `metrics.py`: Custom metrics (VWAP/TWAP performance, risk-adjusted returns)
- `loss.py`: LEM loss function for efficient and risk-sensitive execution
- `data_formatter.py`: Preprocessing pipeline from raw time series to training arrays
- `example_and_results/`: Notebooks with full training code and benchmarks

---

## 📈 Metrics

Each allocation path (Buy/Sell × VWAP/TWAP) is evaluated using custom Keras metrics:
- `*_avg_imp`: average improvement in execution price vs market benchmark
- `*_avg_risk`: risk-adjusted reward relative to baseline
- `unsigned_*_avg_risk`: neutral performance regardless of trade direction

These are designed for both interpretability and reward shaping.

---

## 📝 License



Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
---



