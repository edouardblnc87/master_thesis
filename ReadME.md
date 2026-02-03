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

dataset_bis/<TICKER>/
images/ # 0.png, 1.png, ...
labels/labels.csv


See the dataset builder logic in `utils_datasets.py`. :contentReference[oaicite:12]{index=12}

### 4) Train/val/test split
`split_global_dataset(dataset_dir, ...)` loads all tickers’ labels, attaches file paths to each row, then uses stratified `train_test_split` twice to get train/val/test. :contentReference[oaicite:13]{index=13}

### 5) TensorFlow input pipeline
`dataframe_to_dataset_color(df, image_size=(114, 120))` creates a `tf.data.Dataset` that lazily:
- reads PNGs from disk,
- decodes RGB,
- resizes,
- scales to `[0, 1]`. :contentReference[oaicite:14]{index=14}

---

## Model architecture (CNN)

The CNN defined in the training notebook is also reproduced in the thesis annex (“Model declaration”). It uses:

- Input `(114, 120, 3)`
- Conv blocks with BatchNorm and LeakyReLU in deeper layers
- MaxPooling between blocks
- GlobalAveragePooling2D
- Dense(256) + Dropout(0.5)
- Output: 2-way softmax

and is trained with Adam (lr ≈ 5e-4) using a binary cross-entropy objective. :contentReference[oaicite:15]{index=15}

The repo also includes Grad-CAM helpers (`make_gradcam_heatmap`, `superimpose_heatmap_on_image`, etc.) so you can visualize which candle regions drive the classification decision. :contentReference[oaicite:16]{index=16}

---

## Hedging / evaluation logic

### Greeks + pricing proxy
`utils_results.py` provides a Black–Scholes(-style) European call setup:

- `black_scholes_call_price(...)`
- `call_delta(...)`, `call_gamma(...)`, `call_vega(...)`
- a realized volatility estimate from historical log returns (`get_realized_volatility`) :contentReference[oaicite:17]{index=17}

The thesis frames these Greeks as a **proxy** (not live market-implied Greeks) and discusses the limitations explicitly. :contentReference[oaicite:18]{index=18}

### Standard delta hedging ledger
`get_hedgigng_dataframe(...)` (and the helper `normal_hedging(...)`) maintain a day-by-day ledger:
- how many shares you need to hold (`notional * delta`)
- how many you buy/sell to rebalance
- average buy/sell prices
- cumulative hedging “cost” tracked as net cash from trades

This is used as the baseline “hedge once at the close” strategy. :contentReference[oaicite:19]{index=19}

### Model-informed “two-step” hedging
The strategy described in the thesis (Algorithm 1) is:

- Each day, use the last 20 days’ image to predict whether a large intraday move is likely.
- If yes (class 1), compute a **shadow delta**:
  - conceptually: `Δ_shadow = Δ + Γ * (expected_move)`
- Execute an early hedge at the **open** with `Δ_shadow` (so you buy more shares before the up-move).
- At the **close**, rebalance again to the true delta.
- If no (class 0), hedge only at the close. :contentReference[oaicite:20]{index=20}

This is the “hedge timing” mechanism the repo is designed to illustrate. :contentReference[oaicite:21]{index=21}

---

## Quickstart (reproduce the notebooks)

### 1) Install dependencies
This project was developed in a Conda-like environment; `env.txt` is used as a package list. :contentReference[oaicite:22]{index=22}  
Common approaches:
- Create/activate a clean environment (conda or venv)
- Install packages listed in `env.txt` (you may need to adapt it into a `requirements.txt` depending on your setup)

### 2) Run notebooks in order
Open Jupyter and run:

1. `1-dataset_generations.ipynb`  
   Generates the on-disk dataset folders (`dataset_bis/...`) from the chosen tickers.

2. `2-basic_cnn.ipynb`  
   Set `path_dataset = "<your dataset_bis path>"`, then train. Best weights are saved as `.keras`.

3. `3-results.ipynb`  
   Loads the trained model and evaluates the hedging experiment.

---

## Key knobs you can change

- **Window length**: `n_days` / `window` (commonly 20)
- **Threshold** for class 1: `tresh` (thesis uses ~0.007 = 0.7%) :contentReference[oaicite:23]{index=23}
- **Image size**: the TF pipeline resizes to `(114, 120)` by default. :contentReference[oaicite:24]{index=24}
- **Universe of tickers**: the dataset notebook uses a basket of liquid US industrial names (by design). :contentReference[oaicite:25]{index=25}
- **Hedging period / notional**: set in `3-results.ipynb` (the thesis example highlights UPS over a stressed 2008–2009 window). :contentReference[oaicite:26]{index=26}

---

## Notes / caveats (important if you extend this)

- Greeks are computed from a simplified proxy rather than live option chains; the thesis explicitly flags this as a limitation. :contentReference[oaicite:27]{index=27}
- Execution assumptions for “open” trading are optimistic; realistic slippage and microstructure effects are not fully modeled. :contentReference[oaicite:28]{index=28}
- The repo is notebook-first. If you want to productionize it, the clean next step is to refactor:
  - a config file for paths/params,
  - a CLI to run dataset build / training / evaluation,
  - and deterministic experiment tracking.

---

## Reference

For full methodological context, parameter choices, and the written motivation behind the image encoding + two-step hedging protocol, see the thesis PDF. :contentReference[oaicite:29]{index=29}
