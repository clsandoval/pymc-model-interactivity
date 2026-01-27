# PyMC Marketing Mix Model Explorer

Interactive exploration of Bayesian Marketing Mix Models (MMM) with PyMC using [marimo](https://marimo.io/) reactive notebooks.

## Overview

This repository demonstrates how to build interactive dashboards for exploring PyMC models, with a focus on Marketing Mix Modeling. The key innovation is the **frozen predictor** approach that enables fast counterfactual exploration after model fitting.

## What Makes This Special

After fitting a model, you can define a set of inputs and responses and freeze the rest of the model graph. This makes exploring counterfactuals **VERY fast** - no need to re-run MCMC or recompile the entire model. The `frozen_predictor.py` module implements this by:

- Identifying only the necessary nodes in the computation graph
- Freezing posterior samples for parameters you're not exploring
- Compiling efficient PyTensor functions for interactive updates
- Supporting both single-scenario and multi-scenario inputs

## Notebooks

### 1. Marketing Funnel MMM (`marketing_funnel_mmm.py`)

The main example notebook implementing a **Marketing Mix Model** with a marketing funnel structure:

- **Direct Channel**: Standard media effect (spend → saturation → adstock → sales)
- **Upper Funnel**: Brand awareness channel that improves lower funnel efficiency
- **Lower Funnel**: Performance channel where impressions depend on CPM (affected by upper funnel)

**Features:**
- Interactive sliders to explore saturation and response curves
- Real-time visualization of posterior predictions
- Counterfactual analysis: "What would sales look like if we changed spend?"
- Efficient compiled PyTensor functions for fast interactive updates

**Structure:**
1. Data Generation: Synthetic marketing data with realistic funnel dynamics
2. Model Definition: PyMC model with saturation (Hill function), adstock (geometric), and funnel interactions
3. Model Fitting: MCMC sampling with convergence diagnostics
4. Interactive Exploration:
   - Saturation curves explorer
   - Response curves explorer  
   - Sales time series counterfactual analysis

### 2. Mediation MMM (`mediation_mmm.py`)

A mediation model example showing how to handle observed intermediate variables in the causal chain (e.g., spend → visits → purchases).

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended)

## Running the Notebooks

All notebooks are standalone scripts with embedded dependencies. Simply run:

```bash
# Marketing Funnel MMM
uvx run marimo run marketing_funnel_mmm.py --sandbox

# Mediation MMM
uvx run marimo run mediation_mmm.py --sandbox
```

This automatically creates an isolated environment and installs all dependencies.

## Files

- `marketing_funnel_mmm.py` - Main MMM notebook with funnel structure
- `mediation_mmm.py` - Mediation model example
- `frozen_predictor.py` - Core module for fast counterfactual exploration
- `generate_data.py` - Synthetic data generation module
- `tests/test_frozen_predictor.py` - Tests for the frozen predictor functionality

## License

MIT