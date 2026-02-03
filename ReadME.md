# Deep-Learning Hedge-Timing Demo (OCLHV Project)

This repository contains a compact, research-style implementation of the core idea developed in the accompanying master’s thesis: **using CNNs on OHLCV-based images to improve the timing of delta hedging**.

The project converts rolling windows of financial time series into candlestick-like RGB images, trains a convolutional neural network to predict large intraday moves, and then uses those predictions to decide *when* to hedge (at the open + close vs only at the close).

---

## Repository structure

- `1-dataset_generations.ipynb`  
  Builds the dataset by downloading historical OHLCV data and converting rolling windows into images with binary labels.

- `2-basic_cnn.ipynb`  
  Defines, trains, and saves a convolutional neural network classifier on the generated image dataset.

- `3-results.ipynb`  
  Loads the trained model and evaluates its impact on delta-hedging costs compared to a standard close-only strategy.

- `utils_datasets.py`  
  Helper functions for data download, feature engineering, image generation, dataset splitting, and TensorFlow input pipelines.

- `utils_results.py`  
  Option pricing, Greeks computation, realized volatility estimation, and hedging backtest utilities.

- `model.keras`  
  Saved trained CNN model.

- `env.txt`  
  List of Python packages used to reproduce the environment.

- `master_thesis_edouard_blanc copy.pdf`  
  Full thesis document describing the methodology, financial motivation, and experimental results.

---

## How the code works

### 1. Data and image generation

Historical daily OHLCV data are downloaded using `yfinance`. For each ticker, rolling windows of fixed length (typically 20 trading days) are extracted.

Each window is converted into a **candlestick-style RGB image**:
- Prices are normalized relative to the first close in the window to remove scale effects.
- Each day is drawn as a vertical high–low bar with open/close ticks.
- The RGB channels encode additional information:
  - **Red**: daily open-to-close return,
  - **Green**: intraday range (high–low),
  - **Blue**: normalized trading volume.
- A moving average (e.g. 20-day MA) can be overlaid to provide trend context.

Each image is labeled according to the *next day’s* open-to-close return. If this return exceeds a fixed threshold (e.g. 0.7%), the label is 1 (“large move”), otherwise 0.

Images and labels are saved to disk in per-ticker folders.

---

### 2. Dataset pipeline

All ticker datasets are merged and split into train, validation, and test sets using stratified sampling to preserve class balance.

A TensorFlow `tf.data.Dataset` pipeline:
- loads images lazily from disk,
- decodes PNGs into RGB tensors,
- resizes them to a fixed shape (default: 114×120),
- rescales pixel values to [0, 1].

This keeps memory usage low and enables efficient training.

---

### 3. CNN model

The CNN architecture is intentionally simple and interpretable:
- multiple convolution + batch normalization blocks,
- max pooling for spatial downsampling,
- global average pooling,
- a dense layer with dropout,
- a 2-class softmax output.

The model is trained using Adam optimization and cross-entropy loss.  
Grad-CAM utilities are included to visualize which parts of the candlestick images drive the model’s predictions.

---

### 4. Hedging logic and evaluation

The financial evaluation compares two strategies:

**Baseline**  
Delta hedge once per day at the close using standard Black–Scholes delta.

**Model-informed strategy**  
- If the model predicts a large intraday move:
  - hedge partially at the open using a *shadow delta* (delta adjusted by gamma and an assumed move),
  - rebalance again at the close.
- If no large move is predicted:
  - hedge only at the close.

Option prices and Greeks are computed using a simplified Black–Scholes-style proxy with realized volatility.  
A transaction ledger tracks daily share trades and cumulative hedging costs.

---

## Running the project

1. Create a Python environment and install dependencies listed in `env.txt`.
2. Run the notebooks in order:
   - `1-dataset_generations.ipynb`
   - `2-basic_cnn.ipynb`
   - `3-results.ipynb`
3. Adjust key parameters (window length, labeling threshold, tickers, notional, dates) directly in the notebooks.

---

## Notes and limitations

- Greeks are approximations (not market-implied).
- Execution assumes idealized liquidity at open and close.
- The project is research-oriented and notebook-driven; refactoring into scripts or a package would be the natural next step.

---

## Reference

See the thesis PDF for full motivation, mathematical details, and discussion of limitations.
