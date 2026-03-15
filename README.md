# HMM-Based Market Regime Detection for SPY

> A Hidden Markov Model that identifies latent Bull, Bear, and Sideways market regimes from SPY (S&P 500 ETF) price data — using a constrained Viterbi decoder that encodes real-world market structure to build a regime-driven trading strategy that nearly triples the Sharpe ratio of buy-and-hold.

---

## Overview

Financial markets operate in distinct regimes — extended periods of trending, mean-reverting, or risk-off behaviour — yet these states are latent and cannot be observed directly. A model that reliably identifies the current regime can dramatically improve portfolio allocation by reducing exposure during Bear phases and maximising it during Bull phases.

Standard HMMs applied to financial data often produce regime paths that are noisy and financially unrealistic: a model that allows direct Bull-to-Bear transitions in a single day generates signals that no practitioner would act on. This project addresses that directly by encoding financial domain knowledge into the model architecture itself — disallowing structurally invalid transitions, enforcing minimum regime persistence, and modifying the Viterbi decoder to produce stable, interpretable regime sequences.

The result is a regime-detection system evaluated on 10 years of SPY daily data that delivers a Sharpe ratio of 1.98 against a buy-and-hold baseline of 0.73, while reducing maximum drawdown from 33.72% to 10.56%.

---

## Dataset

**Source:** SPY Historical Price Data — S&P 500 ETF

**File:** `SPY_10y_processed.csv`

| Property | Detail |
|---|---|
| Asset | SPY (S&P 500 ETF) |
| Coverage | 10 years of daily price data |
| Frequency | Daily OHLCV |

**Features engineered during preprocessing (`preprocessing.ipynb`):**

| Feature | Construction | Binning |
|---|---|---|
| `Return` | Daily log return | Tertile binning → Low / Medium / High |
| `Volatility` | 5-day annualised rolling standard deviation | Median split → Low / High |
| `RSI` | 14-day Relative Strength Index | Threshold-based → Oversold (<30) / Neutral / Overbought (>70) |
| `Observation` | Tuple of (Return_Bin, Vol_Bin, RSI_Signal) | Used directly as HMM observation sequence |

---

## Methodology

### Hidden States

Three latent market regimes are defined:

| Regime | Market Character |
|---|---|
| Bull | Rising prices, low volatility, momentum building |
| Sideways | Low directional movement, uncertain conditions |
| Bear | Falling prices, high volatility, risk-off behaviour |

### Emission Probabilities

Each hidden state generates observable feature combinations with manually specified likelihoods, grounded in market intuition:

| Regime | Return Signal | Volatility Signal | RSI Signal |
|---|---|---|---|
| Bull | High | Low | Overbought |
| Sideways | Medium | Low/High | Neutral |
| Bear | Low | High | Oversold |

### Transition Model with Real-World Constraints

Standard HMMs allow any state to transition to any other, producing unrealistic regime paths. This model enforces two structural constraints:

**No direct Bull → Bear or Bear → Bull transitions.** Markets do not switch from full bull to full bear in a single day — they pass through Sideways as an intermediate regime. These transitions are set to probability 0.

**Minimum 5-day regime persistence.** A regime must persist for at least 5 trading days before a state change is allowed. This prevents noisy single-day flips driven by short-term volatility.

**Transition matrix:**

| From \ To | Bull | Sideways | Bear |
|---|---|---|---|
| Bull | 0.70 | 0.30 | 0.00 |
| Sideways | 0.30 | 0.40 | 0.30 |
| Bear | 0.00 | 0.40 | 0.60 |

High self-transition probabilities model trend continuation — a Bull market is more likely to remain Bull than to change regime on any given day.

### Constrained Viterbi Decoding (`viterbi.py`)

The standard Viterbi algorithm was modified to enforce both constraints during decoding:

- At each time step, Bull → Bear and Bear → Bull transitions are skipped entirely (probability set to zero in the recursion)
- A `min_persist` parameter (default: 5) forces the decoder to verify that the previous state has persisted for at least the minimum duration before allowing a regime change

The result is a regime path that is simultaneously the maximum-likelihood sequence given the observations and a financially valid sequence of market states.

### Backtesting Strategy

Detected regimes drive a principled allocation rule applied over the full 10-year backtest period:

| Regime | Portfolio Action |
|---|---|
| Bull | Fully invested in SPY |
| Sideways | Reduced exposure / neutral |
| Bear | Defensive — cash or minimal exposure |

---

## Results

The HMM regime strategy was benchmarked against a passive buy-and-hold strategy on the same SPY data over the full 10-year period.

**Strategy Comparison:**

| Strategy | Sharpe Ratio | Max Drawdown |
|---|---|---|
| Buy and Hold | 0.73 | 33.72% |
| HMM Regime Strategy | 1.98 | 10.56% |

The HMM strategy improved the Sharpe ratio by approximately 2.7× while reducing maximum drawdown by more than two-thirds. The strategy demonstrated stable regime detection across multiple full market cycles including the 2018 correction, the 2020 COVID crash, and the 2022 rate-hike drawdown.

**Stress Testing:** A 10-day extension of the worst return periods from March 2020 and synthetic volatility spikes were simulated to test model robustness. The HMM maintained regime stability, transitioned out of Bear predictably, and produced no excessive regime switching under these conditions.

---

## Limitations & Future Work

**Current Limitations:**

- Transition and emission probabilities are manually specified rather than learned from data — the absence of Baum-Welch training means the model cannot adapt its parameters to new market regimes or structural breaks
- The model assumes stationary behaviour across the full 10-year period; the same transition matrix governs a low-rate 2015 market and a post-COVID 2021 rally
- Observations are based on three binned features with fixed thresholds; finer-grained signals and continuous-valued emissions are not captured
- The model is trained and tested on SPY only, limiting generalisability to other indices or asset classes

**Future Directions:**

- Implement Baum-Welch (Expectation-Maximisation) to learn transition and emission probabilities directly from data, or use semi-supervised learning with rolling windows for adaptive updating
- Extend observations to include macroeconomic features such as interest rates, inflation, VIX levels, and yield curve slope to improve regime characterisation
- Evaluate the strategy on QQQ, IWM, and DIA to assess cross-index generalisability
- Replace fixed binning with continuous emission distributions (Gaussian HMM) or neural emission networks for richer observation modelling
- Explore Hidden Semi-Markov Models (HSMMs) which explicitly model regime duration distributions rather than enforcing a fixed minimum persistence heuristic

---

## How to Run This Project

### Prerequisites

```bash
Python 3.8+
```

### 1. Clone the Repository

```bash
git clone https://github.com/d4h2nu8h/hmm-market-regime-detection.git
cd hmm-market-regime-detection
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### 3. Run Preprocessing

```bash
jupyter notebook preprocessing.ipynb
```

This loads `SPY_10y_raw.csv`, engineers the Return, Volatility, and RSI features, bins them into discrete observation symbols, and saves `SPY_preprocessed.csv`.

### 4. Run the HMM and Backtest

```bash
jupyter notebook hmm_template.ipynb
```

This loads the preprocessed data, defines the transition and emission probability matrices, runs constrained Viterbi decoding, applies the regime-based allocation strategy, and produces the performance and visualisation outputs.

### 5. Test the Viterbi Decoder Standalone

```bash
python viterbi.py
```

Runs the built-in sanity check with a small example observation sequence and prints the predicted regime path.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| HMM & Decoding | Custom implementation (NumPy) |
| Data Processing | Pandas, NumPy |
| Technical Indicators | Manual RSI and rolling volatility computation |
| Visualisation | Matplotlib, Seaborn |
| Notebook Environment | Jupyter Notebook / Google Colab |
| Dataset | SPY historical daily OHLCV |

---

## Authors

Dhanush Sambasivam

[![GitHub](https://img.shields.io/badge/GitHub-d4h2nu8h-181717?style=flat&logo=github)](https://github.com/d4h2nu8h)

---

## License

This project is intended for academic and research purposes.
