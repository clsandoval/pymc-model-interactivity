# PyMC Model Interactivity

Interactive exploration of Bayesian models with PyMC using [marimo](https://marimo.io/) reactive notebooks.

## Overview

This repository demonstrates how to build interactive dashboards for exploring PyMC models.

## What is special
After fitting a model, I show how one can define a set of imputs and responses and freeze the rest of the model graph. This makes exloring counterfactuals VERY fast.

## Example

The example notebook implements a **Marketing Mix Model (MMM)** with a marketing funnel structure:

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
- [uv](https://docs.astral.sh/uv/) (recommended)

## Running the Notebook

The notebook is a standalone script with embedded dependencies. Simply run:

```bash
uvx run marimo run marketing_funnel_mmm.py --sandbox
```

This automatically creates an isolated environment and installs all dependencies.

If you want to edit the notebook, you can run it in edit mode with:

```bash
uvx run marimo run marketing_funnel_mmm.py --sandbox
```

## Notebook Structure

1. **Data Generation**: Synthetic marketing data with realistic funnel dynamics
2. **Model Definition**: PyMC model with saturation (Hill function), adstock (geometric), and funnel interactions
3. **Model Fitting**: MCMC sampling with convergence diagnostics
4. **Interactive Exploration –  THE EXCITING PART**:
   - Saturation curves explorer
   - Response curves explorer  
   - Sales time series counterfactual analysis

## Files

- `marketing_funnel_mmm.py` - Main marimo notebook
- `generate_data.py` - Synthetic data generation module

## License

MIT