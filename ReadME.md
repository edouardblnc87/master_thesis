# Deep-Learning Hedge-Timing Demo (OCLHV Project)

This repo is a compact, notebook-driven demo of the core idea in the accompanying master’s thesis: **turn price/volume time series into images**, train a **CNN** to predict “large” intraday moves, and then use that prediction to **time part of a delta hedge** (hedge earlier at the open when a big up-move is likely).

The project is intentionally “research-style” (Jupyter notebooks + small utility modules) rather than a fully packaged library.

---

## What’s in this repository

- `1-dataset_generations.ipynb`  
  Builds the dataset: downloads OHLCV history and converts rolling windows into candlestick-like images, plus labels.

- `2-basic_cnn.ipynb`  
  Defines and trains the CNN on those images; saves weights to a `.keras` file.

- `3-results.ipynb`  
  Loads the model and runs the hedging cost comparison: standard close-only delta hedging vs “model-informed” two-step hedging (open + close).

- `utils_datasets.py`  
  Dataset tooling: download prices, generate images, create per-ticker datasets on disk, and build `tf.data.Dataset` pipelines. :contentReference[oaicite:0]{index=0}

- `utils_results.py`  
  Backtest / option math utilities: Black–Scholes(-like) pricing + Greeks, realized vol proxy, and a transaction-cost style hedging ledger. :contentReference[oaicite:1]{index=1}

- `env.txt`  
  A “requirements-style” list of packages used in the environment. :contentReference[oaicite:2]{index=2}

- `model.keras`  
  A saved trained model (produced by the training notebook).

- `master_thesis_edouard_blanc copy.pdf`  
  Full thesis writeup + an annex that reproduces the key code listings and the hedging algorithm. :contentReference[oaicite:3]{index=3}

---

## How the pipeline works (high level)

The workflow is linear and notebook-driven:

1. **Download OHLCV** for a list of tickers (daily data; start date around 2000). :contentReference[oaicite:4]{index=4}  
2. **Slice rolling windows** of length `n_days` (typically 20 trading days).  
3. **Encode each window as an RGB image** where each “day” becomes a candlestick-like glyph. The three color channels are used to inject extra information (return, intraday range, relative volume), and an optional moving-average overlay provides trend context. :contentReference[oaicite:5]{index=5}  
4. **Label** each window with a binary target `y` (class 1 if the next day’s open→close move exceeds a threshold, else class 0). The thesis uses **0.7%** as the main separation threshold. :contentReference[oaicite:6]{index=6}  
5. **Train a CNN classifier** on the images. The architecture is a straightforward stack of convolution blocks + global average pooling + a dense head. :contentReference[oaicite:7]{index=7}  
6. **Use predictions to time hedging**: if the model predicts a “large move” day, do a partial anticipatory hedge at the open using a *shadow delta* (delta adjusted by gamma and an assumed move), then rebalance again at the close; otherwise hedge once at the close. :contentReference[oaicite:8]{index=8}

---

## Dataset creation details (what the code is doing)

### 1) Price download + features
`utils_datasets.download_historical_prices(ticker)` pulls daily OHLCV via `yfinance`, resets the index, and computes a 20-day moving average (`MA20`). :contentReference[oaicite:9]{index=9}

### 2) OHLC window → image (the key preprocessing step)
The main image encoder used in the thesis is implemented as `generate_ohlc_image_bis(...)`. For a fixed window of `n_days`, it:

- Normalizes prices by the first day’s close (so the network focuses on *shape*, not absolute price level).
- Normalizes volume within the window (so volume is relative).
- Maps price levels into pixel y-coordinates.
- For each day, draws the candle’s high–low range and open/close ticks.
- Encodes **three numerical signals into RGB**:
  - `R`: daily open→close return (clipped/rescaled),
  - `G`: intraday range proxy (high–low relative to open),
  - `B`: normalized volume. :contentReference[oaicite:10]{index=10}

This matches the “OHLC rows → RGB candlestick image” listing reproduced in the thesis annex. :contentReference[oaicite:11]{index=11}

### 3) Labeling
When building a ticker dataset (`create_ticker_dataset_bis`), each window’s label is computed as:

> `y = 1` if `Open_last_day * (1 + threshold) < Close_next_day`, else `0`

and both images and `labels.csv` are written to disk under:


