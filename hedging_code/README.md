# Deep-Learning Hedge-Timing Demo (OCLHV Project)

This repository accompanies my master’s thesis **“Deep Learning for Options Pricing and Hedging.”**  
It shows, in a few Jupyter notebooks, how to turn raw OHLCV data into images, train a convolutional neural network, and evaluate whether the resulting signal can lower delta-hedging cost.

---

## Notebook guide

| Order | File | Purpose |
|-------|------|---------|
| **1** | `1-dataset_generations.ipynb` | Downloads daily OHLCV with *yfinance* and converts each 20-day window into a 224 × 224 RGB candlestick image. The script also writes `labels.csv` with the target class (large move yes/no). |
| **2** | `2-basic_cnn.ipynb` | Builds and trains the CNN on the generated images. The fitted weights are saved to `model.keras`. |
| **3** | `3-results.ipynb` | Loads the trained model, computes accuracy/precision/recall, and plots the cost comparison between the image-based hedge-timing strategy and the standard close-only hedge. |

Run the notebooks in that order; each step depends on the artefacts produced by the previous one.

---

## Quick setup

```bash
pip install -r env.txt      # install required Python packages
jupyter lab                 # open the notebooks
