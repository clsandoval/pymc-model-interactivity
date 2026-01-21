"""
Synthetic data generation for marketing funnel MMM demonstration.

This module generates realistic marketing data with:
- Three media channels (direct, upper funnel, lower funnel)
- Funnel dynamics where upper funnel affects lower funnel CPM
- Seasonal baseline patterns
- Inflation as a control variable
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Ground truth parameters for data generation
TRUE_PARAMS = {
    # Baseline
    "intercept_base": 1000.0,
    "noise_sigma": 50.0,
    
    # Control variable
    "beta_inflation": -30.0,  # Negative effect of inflation on sales
    
    # Direct channel parameters
    "beta_direct": 200.0,
    "slope_direct": 2.0,
    "kappa_direct": 0.5,  # Half-saturation point (normalized)
    "alpha_direct": 0.4,  # Adstock decay
    
    # Upper funnel channel parameters
    "beta_upper": 100.0,
    "slope_upper": 1.5,
    "kappa_upper": 0.4,
    "alpha_upper": 0.5,  # Higher carryover for brand awareness
    
    # Lower funnel channel parameters
    "beta_lower": 250.0,
    "slope_lower": 2.5,
    "kappa_lower": 0.3,  # Saturates impressions, not spend
    "alpha_lower": 0.3,  # Faster decay for performance channels
    
    # Funnel interaction
    "cpm_base": 10.0,  # Base CPM in dollars
    "gamma": 0.8,  # How much upper funnel reduces CPM
    
    # Adstock settings
    "l_max": 8,  # Maximum lag for adstock
}


def hill_saturation_np(x: np.ndarray, slope: float, kappa: float) -> np.ndarray:
    """
    Hill saturation function (numpy implementation).
    
    Parameters
    ----------
    x : array
        Input values (normalized spend or impressions)
    slope : float
        Controls steepness of the curve
    kappa : float
        Half-saturation point (x value where output = 0.5)
    
    Returns
    -------
    array
        Saturated values in [0, 1]
    """
    return x**slope / (kappa**slope + x**slope)


def geometric_adstock_np(
    x: np.ndarray, 
    alpha: float, 
    l_max: int = 8,
    normalize: bool = True
) -> np.ndarray:
    """
    Geometric adstock transformation (numpy implementation).
    
    Applies exponentially decaying carryover effect.
    
    Parameters
    ----------
    x : array
        Input time series
    alpha : float
        Decay rate (0 = no carryover, 1 = full carryover)
    l_max : int
        Maximum lag to consider
    normalize : bool
        If True, normalize weights to sum to 1
    
    Returns
    -------
    array
        Adstocked time series
    """
    # Create decay weights
    weights = np.array([alpha**i for i in range(l_max)])
    if normalize:
        weights = weights / weights.sum()
    
    # Pad input and convolve
    padded = np.concatenate([np.zeros(l_max - 1), x])
    result = np.convolve(padded, weights, mode='valid')
    
    return result


def generate_marketing_funnel_data(
    n_periods: int = 104,
    seed: int = 42,
    params: dict = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate synthetic marketing funnel data.
    
    Parameters
    ----------
    n_periods : int
        Number of time periods (weeks)
    seed : int
        Random seed for reproducibility
    params : dict, optional
        Override default TRUE_PARAMS
    
    Returns
    -------
    df : DataFrame
        Generated data with columns:
        - date: datetime index
        - spend_direct, spend_upper, spend_lower: media spends
        - impressions_lower: computed impressions for lower funnel
        - cpm_lower: effective CPM for lower funnel
        - inflation: control variable
        - sales: target variable
        - baseline_true: true baseline (for validation)
    params : dict
        Parameters used for generation (for model validation)
    """
    np.random.seed(seed)
    
    # Use provided params or defaults
    p = {**TRUE_PARAMS, **(params or {})}
    
    # Generate date index
    dates = pd.date_range(start="2024-01-01", periods=n_periods, freq="W-MON")
    t = np.arange(n_periods)
    
    # Generate seasonal baseline pattern (normalized to mean ~1)
    # Combination of yearly seasonality + slight trend
    seasonality = (
        0.15 * np.sin(2 * np.pi * t / 52)  # Yearly cycle
        + 0.08 * np.sin(4 * np.pi * t / 52)  # Semi-annual
        + 0.02 * t / n_periods  # Slight upward trend
    )
    baseline_multiplier = 1.0 + seasonality
    baseline_true = p["intercept_base"] * baseline_multiplier
    
    # Generate media spends with realistic patterns
    # Direct channel: relatively stable with some variation
    spend_direct = (
        50 + 30 * np.random.rand(n_periods)
        + 10 * np.sin(2 * np.pi * t / 13)  # Quarterly pattern
    )
    spend_direct = np.maximum(spend_direct, 0)
    
    # Upper funnel: brand campaigns, more variable, sometimes paused
    spend_upper = (
        40 + 50 * np.random.rand(n_periods)
        + 20 * np.sin(2 * np.pi * t / 26)  # Semi-annual campaigns
    )
    # Add some periods with reduced spending
    spend_upper *= np.where(np.random.rand(n_periods) > 0.15, 1.0, 0.3)
    spend_upper = np.maximum(spend_upper, 0)
    
    # Lower funnel: performance channel, relatively consistent
    spend_lower = (
        60 + 25 * np.random.rand(n_periods)
        + 15 * np.sin(2 * np.pi * t / 52)  # Yearly pattern
    )
    spend_lower = np.maximum(spend_lower, 0)
    
    # Generate inflation (centered around 0, representing deviation from normal)
    inflation = 0.5 * np.cumsum(np.random.randn(n_periods) * 0.1)
    inflation = inflation - inflation.mean()  # Center it
    
    # Normalize spends for saturation (relative to typical spend levels)
    spend_direct_norm = spend_direct / 100.0
    spend_upper_norm = spend_upper / 100.0
    spend_lower_norm = spend_lower / 100.0
    
    # Process direct channel: saturation -> adstock
    direct_saturated = hill_saturation_np(
        spend_direct_norm, p["slope_direct"], p["kappa_direct"]
    )
    direct_effect = geometric_adstock_np(
        direct_saturated, p["alpha_direct"], p["l_max"]
    )
    
    # Process upper funnel: saturation -> adstock
    upper_saturated = hill_saturation_np(
        spend_upper_norm, p["slope_upper"], p["kappa_upper"]
    )
    upper_effect = geometric_adstock_np(
        upper_saturated, p["alpha_upper"], p["l_max"]
    )
    
    # Compute lower funnel CPM (affected by upper funnel)
    # Higher upper funnel activity -> lower CPM (better efficiency)
    cpm_lower = p["cpm_base"] * np.exp(-p["gamma"] * upper_effect)
    
    # Compute impressions for lower funnel
    impressions_lower = spend_lower / cpm_lower
    impressions_lower_norm = impressions_lower / (spend_lower.mean() / p["cpm_base"])
    
    # Process lower funnel: saturation on impressions -> adstock
    lower_saturated = hill_saturation_np(
        impressions_lower_norm, p["slope_lower"], p["kappa_lower"]
    )
    lower_effect = geometric_adstock_np(
        lower_saturated, p["alpha_lower"], p["l_max"]
    )
    
    # Compute total sales
    sales = (
        baseline_true
        + p["beta_inflation"] * inflation
        + p["beta_direct"] * direct_effect
        + p["beta_upper"] * upper_effect
        + p["beta_lower"] * lower_effect
        + np.random.randn(n_periods) * p["noise_sigma"]
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "spend_direct": spend_direct,
        "spend_upper": spend_upper,
        "spend_lower": spend_lower,
        "impressions_lower": impressions_lower,
        "cpm_lower": cpm_lower,
        "inflation": inflation,
        "sales": sales,
        "baseline_true": baseline_true,
    })
    
    # Also store intermediate effects for validation
    df["direct_effect_true"] = direct_effect
    df["upper_effect_true"] = upper_effect
    df["lower_effect_true"] = lower_effect
    
    return df, p


if __name__ == "__main__":
    # Test data generation
    df, params = generate_marketing_funnel_data()
    print("Generated data shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSummary statistics:")
    print(df.describe())
    print("\nTrue parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
