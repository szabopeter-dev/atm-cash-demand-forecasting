# ATM Cash Demand Forecasting

Multi-horizon ATM cash demand prediction using LSTM/GRU encoder-decoder networks.

## Setup

```bash
git clone <repo-url>
cd atm-forecasting
pip install -r requirements.txt
```

## Data

Place `ATM_OK3.csv` in the `data/` folder:

```
data/
└── ATM_OK3.csv
```

## Run

```bash
python train.py
```

Trained models are saved to `models/`, results to `results.csv`.

## What it does

- Loads 727K ATM transactions (10 ATMs, 2011–2012)
- Aggregates to 4-hour intervals
- Engineers 13 features (temporal, PACF lags, cross-ATM averages)
- Trains LSTM and GRU seq2seq encoder-decoder models
- Evaluates on 5 forecast horizons: 4h, 1d, 2d, 1w, 2w
- Metrics: MAE, R², NMAE

## Requirements

- Python 3.9+
- TensorFlow 2.13+
- GPU recommended for the longer horizons (1w, 2w)
