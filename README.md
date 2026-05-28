# ATM Cash Demand Forecasting

Multi-horizon ATM cash demand prediction with LSTM and GRU sequence-to-sequence
networks. Companion code for the SAMI 2026 paper.

> P. Szabó, B. Gáspár. **"Predicting ATM Cash Demand Using Machine Learning."**
> *IEEE 24th World Symposium on Applied Machine Intelligence and Informatics
> (SAMI 2026)*, Stará Lesná, Slovakia, pp. 411–416.

## Overview

- **Data:** 727K real ATM transactions across 10 ATMs (2011–2012), aggregated to
  4-hour buckets per ATM.
- **Features (13):** hour / weekday / weekend / holiday / month-end flags,
  PACF-selected lags (43, 47, 126), cross-ATM peer mean, and rolling
  daily/weekly averages.
- **Horizons (5):** 4 h, 1 day, 2 days, 1 week, 2 weeks.
- **Models:** encoder-decoder LSTM (128) and GRU (128) trained per horizon with
  early stopping, LR scheduling, and best-checkpoint saving.
- **Metrics:** MAE, R², and NMAE (normalized by the overall mean cash demand).

A temporal 70 / 15 / 15 train / val / test split is used to avoid leakage from
future buckets into the training set.

## Setup

```bash
pip install -r requirements.txt
# place ATM_OK3.csv into data/
python train.py
```

Outputs:

- `models/multihorizon_{LSTM|GRU}_{4h|1day|2days|1week|2weeks}.keras`
- `results.csv` — per-architecture × per-horizon MAE / R² / NMAE table

The `colab.ipynb` notebook runs the same pipeline end-to-end in Colab.

## Requirements

- Python 3.9+
- TensorFlow 2.13+
- GPU recommended for the longer horizons (1w, 2w)
