# PyMC Model Interactivity

Interactive exploration of Bayesian models with PyMC using [marimo](https://marimo.io/) reactive notebooks.

## Overview

This repository demonstrates how to build interactive dashboards for exploring PyMC models. The example notebook implements a **Marketing Mix Model (MMM)** with a marketing funnel structure:

- **Direct Channel**: Standard media effect (spend → saturation → adstock → sales)
- **Upper Funnel**: Brand awareness channel that improves lower funnel efficiency
- **Lower Funnel**: Performance channel where impressions depend on CPM (affected by upper funnel)

### Features

- Interactive sliders to explore saturation and response curves
- Real-time visualization of posterior predictions
- Counterfactual analysis: "What would sales look like if we changed spend?"
- Efficient compiled PyTensor functions for fast interactive updates

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Running the Notebook

### Option 1: Using uv (Recommended)

The notebook is a standalone script with embedded dependencies. Simply run:

```bash
uv run marimo edit marketing_funnel_mmm.py
```

This automatically creates an isolated environment and installs all dependencies.

### Option 2: Using pip

First, install the dependencies:

```bash
pip install marimo pymc pymc-marketing arviz numpy pandas matplotlib pytensor jax numpyro
```

Then run the notebook:

```bash
marimo edit marketing_funnel_mmm.py
```

### Option 3: Run as an App

To view the notebook as a read-only app (no code editing):

```bash
uv run marimo run marketing_funnel_mmm.py
```

## Notebook Structure

1. **Data Generation**: Synthetic marketing data with realistic funnel dynamics
2. **Model Definition**: PyMC model with saturation (Hill function), adstock (geometric), and funnel interactions
3. **Model Fitting**: MCMC sampling with convergence diagnostics
4. **Interactive Exploration**:
   - Saturation curves explorer
   - Response curves explorer  
   - Sales time series counterfactual analysis

## Key Concepts

### Saturation Curves
Shows the instantaneous effect of spend on sales, accounting for diminishing returns (Hill function).

### Response Curves
Shows the total effect (summed over all time periods) when scaling historical spend patterns, including adstock carry-over effects.

### Funnel Interaction
Upper funnel spending reduces CPM (cost per mille) for lower funnel, meaning brand awareness makes performance marketing more efficient.

## Files

- `marketing_funnel_mmm.py` - Main marimo notebook
- `generate_data.py` - Synthetic data generation module

## License

MIT
