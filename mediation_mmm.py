# /// script
# dependencies = [
#     "marimo>=0.19.0",
#     "pymc>=5.10.0",
#     "arviz",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "pytensor",
#     "pyzmq>=27.1.0",
#     "jax==0.9.0",
#     "numpyro==0.19.0",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    # Import the posterior predictor
    from frozen_predictor import create_frozen_predictor


@app.cell(hide_code=True)
def _():
    # Model Structure Diagram
    _diagram = """
    flowchart LR
        subgraph Inputs
            spend[Ad Spend]
        end
        
        subgraph Intermediate[Observed Intermediate]
            visits[Website Visits]
        end
        
        subgraph Outcome[Observed Outcome]
            purchases[Purchases]
        end
        
        spend -->|saturation| visits
        visits -->|conversion| purchases
        baseline[Baseline Trend] --> purchases
    """

    mo.md(f"""
    # Mediation Model: Observed Intermediate Variable

    This notebook demonstrates a simple mediation model where:
    
    - **Ad Spend** drives **Website Visits** (observed intermediate)
    - **Website Visits** drive **Purchases** (observed outcome)
    - A **baseline trend** also affects purchases

    ## Model Structure

    {mo.mermaid(_diagram).text}

    **Key feature:** The model has **two observed random variables**:
    1. `visits ~ Normal(mu_visits, sigma_visits)` - observed intermediate
    2. `purchases ~ Normal(mu_purchases, sigma_purchases)` - observed outcome

    This structure allows us to:
    - Predict visits given spend
    - Predict purchases given visits (or given spend, propagating through visits)
    - Explore counterfactuals: "What if we had more visits at the same spend?"
    """)
    return


@app.cell(hide_code=True)
def _():
    # ============================================================
    # Self-contained data generation
    # ============================================================
    
    # True parameters (known for validation)
    TRUE_PARAMS = {
        # Spend -> Visits (saturation curve)
        "intercept_visits": 100.0,      # Base visits (intercept)
        "beta_spend": 50.0,         # Max additional visits from spend
        "slope_spend": 4.0,         # Hill slope
        "kappa_spend": 0.5,         # Half-saturation point
        "sigma_visits": 15.0,       # Visits noise
        
        # Visits -> Purchases
        "intercept_purchases": 10.0,    # Base purchases (intercept)
        "conversion_rate_visits": 0.3,         # Conversion rate (purchases per visit)
        "trend_purchases": 0.05,    # Weekly trend in purchases
        "sigma_purchases": 1.0,     # Purchases noise
    }
    
    def generate_mediation_data(n_periods=52, seed=42):
        """Generate synthetic data for the mediation model."""
        rng = np.random.default_rng(seed)
        
        # Time index
        t = np.arange(n_periods)
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="W")
        
        # Generate spend with some seasonality
        spend_base = 50 + 30 * np.sin(2 * np.pi * t / 52)  # Yearly cycle
        spend_noise = rng.normal(0, 20, n_periods)
        spend = np.maximum(spend_base + spend_noise, 0)  # Minimum spend of 60
        
        # Saturation function (Hill)
        def hill(x, slope, kappa):
            return x ** slope / (kappa ** slope + x ** slope)
        
        # Normalize spend for saturation
        spend_norm = spend / 100.0
        
        # True visits: saturated response to spend
        mu_visits = (
            TRUE_PARAMS["intercept_visits"] 
            + TRUE_PARAMS["beta_spend"] * hill(spend_norm, TRUE_PARAMS["slope_spend"], TRUE_PARAMS["kappa_spend"])
        )
        visits = mu_visits + rng.normal(0, TRUE_PARAMS["sigma_visits"], n_periods)
        visits = np.maximum(visits, 0)  # No negative visits
        
        # True purchases: linear in visits + trend
        mu_purchases = (
            TRUE_PARAMS["intercept_purchases"]
            + TRUE_PARAMS["conversion_rate_visits"] * visits
            + TRUE_PARAMS["trend_purchases"] * t
        )
        purchases = mu_purchases + rng.normal(0, TRUE_PARAMS["sigma_purchases"], n_periods)
        purchases = np.maximum(purchases, 0)  # No negative purchases
        
        df = pd.DataFrame({
            "date": dates,
            "t": t,
            "spend": spend,
            "visits": visits,
            "purchases": purchases,
            "mu_visits_true": mu_visits,
            "mu_purchases_true": mu_purchases,
        })
        
        return df, TRUE_PARAMS
    
    # Generate data
    df, true_params = generate_mediation_data(n_periods=52, seed=42)
    return df, true_params, TRUE_PARAMS


@app.cell(hide_code=True)
def _(df):
    # Data Visualization
    _fig, _axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot spend
    _axes[0].plot(df["date"], df["spend"], color="blue", alpha=0.8)
    _axes[0].set_ylabel("Spend")
    _axes[0].set_title("Ad Spend Over Time")
    _axes[0].grid(True, alpha=0.3)

    # Plot visits
    _axes[1].scatter(df["date"], df["visits"], color="orange", alpha=0.6, s=30, label="Observed")
    _axes[1].plot(df["date"], df["mu_visits_true"], color="gray", linestyle="--", 
                  alpha=0.7, label="True mean")
    _axes[1].set_ylabel("Website Visits")
    _axes[1].set_title("Website Visits (Observed Intermediate)")
    _axes[1].legend()
    _axes[1].grid(True, alpha=0.3)

    # Plot purchases
    _axes[2].scatter(df["date"], df["purchases"], color="green", alpha=0.6, s=30, label="Observed")
    _axes[2].plot(df["date"], df["mu_purchases_true"], color="gray", linestyle="--",
                  alpha=0.7, label="True mean")
    _axes[2].set_ylabel("Purchases")
    _axes[2].set_xlabel("Date")
    _axes[2].set_title("Purchases (Observed Outcome)")
    _axes[2].legend()
    _axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(df):
    # ============================================================
    # Model Definition
    # ============================================================
    # 
    # Key: Two observed RVs - visits and purchases
    # This creates an observed node between inputs (spend) and outcome (purchases)
    
    # Prepare data
    n_obs = len(df)
    t_idx = df["t"].values.astype(np.float64)
    spend_data = df["spend"].values / 100.0  # Normalize
    visits_data = df["visits"].values
    purchases_data = df["purchases"].values

    with pm.Model(coords={"date": df["date"].values}) as mediation_model:
        # === Data inputs ===
        spend = pm.Data("spend", spend_data, dims="date")
        time_idx = pm.Data("time_idx", t_idx, dims="date")
        
        # === Stage 1: Spend -> Visits ===
        # Saturation parameters
        slope_spend = pm.Normal("slope_spend", mu=2, sigma=0.5)
        kappa_spend = pm.Beta("kappa_spend", alpha=5, beta=5)
        
        # Visits coefficients
        intercept_visits = pm.Normal("intercept_visits", mu=100, sigma=30)
        beta_spend = pm.Normal("beta_spend", mu=50, sigma=20)
        
        # Hill saturation
        spend_saturated = spend ** slope_spend / (kappa_spend ** slope_spend + spend ** slope_spend)
        
        # Expected visits
        mu_visits = pm.Deterministic(
            "mu_visits",
            intercept_visits + beta_spend * spend_saturated,
            dims="date"
        )
        
        # Observed visits (intermediate node!)
        sigma_visits = pm.HalfNormal("sigma_visits", sigma=20)
        visits = pm.Normal("visits", mu=mu_visits, sigma=sigma_visits, observed=visits_data, dims="date")
        
        # === Stage 2: Visits -> Purchases ===
        # Purchases coefficients
        intercept_purchases = pm.Normal("intercept_purchases", mu=10, sigma=10)
        conversion_rate_visits = pm.Normal("conversion_rate_visits", mu=0.3, sigma=0.15)  # Wider prior covering true value
        trend_purchases = pm.Normal("trend_purchases", mu=0, sigma=0.1)
        
        # Expected purchases (depends on visits!)
        mu_purchases = pm.Deterministic(
            "mu_purchases",
            intercept_purchases + conversion_rate_visits * visits + trend_purchases * time_idx,
            dims="date"
        )
        
        # Observed purchases (outcome)
        sigma_purchases = pm.HalfNormal("sigma_purchases", sigma=10)
        purchases = pm.Normal("purchases", mu=mu_purchases, sigma=sigma_purchases, observed=purchases_data, dims="date")
        
        # === Total effect: spend -> purchases (for response curves) ===
        # This is the indirect effect mediated through visits
        total_visits_effect = pm.Deterministic("total_visits_effect", mu_visits.sum())
        total_purchases_effect = pm.Deterministic("total_purchases_effect", mu_purchases.sum())

    mediation_model
    return (mediation_model,)


@app.cell(hide_code=True)
def _(mediation_model):
    # Model fitting - runs automatically (simple model, fast sampling)
    mo.output.append(mo.md("## Model Fitting"))
    mo.output.append(mo.md("Running MCMC sampling..."))
    
    with mediation_model:
        idata = pm.sample(
            draws=500,
            tune=500,
            chains=4,
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
            nuts_sampler="numpyro"
        )
    
    mo.output.append(mo.md("**Model fitting complete!**"))
    return (idata,)


@app.cell(hide_code=True)
def _(idata):
    convergence_summary = None
    if idata:
        # Convergence summary (numerical check)
        convergence_summary = az.summary(idata, var_names=[
            "intercept_visits", "beta_spend", "slope_spend", "kappa_spend", "sigma_visits",
            "intercept_purchases", "conversion_rate_visits", "trend_purchases", "sigma_purchases"
        ])
    return (convergence_summary,)


@app.cell(hide_code=True)
def _():
    # Trace plot controls
    trace_plot_control = mo.ui.radio(
        options={"Hide": False, "Show": True},
        value="Hide",
        label="Show convergence diagnostics"
    )
    trace_plot_control
    return (trace_plot_control,)


@app.cell(hide_code=True)
def _(convergence_summary, trace_plot_control, TRUE_PARAMS):
    _out = None
    if convergence_summary is not None:
        _max_rhat = convergence_summary["r_hat"].max()
        _min_ess = convergence_summary["ess_bulk"].min()

        if _max_rhat < 1.01:
            _status = mo.md(f"**Status: All parameters converged** (max R-hat = {_max_rhat:.4f}, min ESS = {_min_ess:.0f})")
        else:
            _problematic = convergence_summary[convergence_summary["r_hat"] >= 1.01].index.tolist()
            _status = mo.md(f"**Warning: Some parameters may not have converged**\n\nmax R-hat = {_max_rhat:.4f}. Check: {_problematic}")

        # Add comparison to true values
        _true_values = {
            "intercept_visits": TRUE_PARAMS["intercept_visits"],
            "beta_spend": TRUE_PARAMS["beta_spend"],
            "slope_spend": TRUE_PARAMS["slope_spend"],
            "kappa_spend": TRUE_PARAMS["kappa_spend"],
            "sigma_visits": TRUE_PARAMS["sigma_visits"],
            "intercept_purchases": TRUE_PARAMS["intercept_purchases"],
            "conversion_rate_visits": TRUE_PARAMS["conversion_rate_visits"],
            "trend_purchases": TRUE_PARAMS["trend_purchases"],
            "sigma_purchases": TRUE_PARAMS["sigma_purchases"],
        }
        
        _summary_with_true = convergence_summary.copy()
        _summary_with_true["true_value"] = [_true_values.get(idx, np.nan) for idx in _summary_with_true.index]
        _summary_with_true = _summary_with_true.reset_index().rename(columns={"index": "parameter"})

        if trace_plot_control.value:
            _out = mo.vstack([
                mo.md("## Convergence Diagnostics"),
                _status,
                mo.md("### Parameter Summary (with true values)"),
                mo.ui.table(_summary_with_true),
            ])
        else:
            _out = mo.md(f"{_status}\n\n*Select 'Show convergence diagnostics' above to display diagnostics*")
    _out
    return


@app.cell(hide_code=True)
def _(idata, trace_plot_control):
    # Conditional trace plots
    if not idata:
        _out = mo.md("*Click 'Fit Model' to start sampling*")
    else:
        _out = mo.md("*Select 'Show convergence diagnostics' above to display trace plots*")    
        if trace_plot_control.value:
            _fig = plt.figure(figsize=(14, 16))
            az.plot_trace(
                idata, 
                var_names=[
                    "intercept_visits", "beta_spend", "slope_spend", "kappa_spend", "sigma_visits",
                    "intercept_purchases", "conversion_rate_visits", "trend_purchases", "sigma_purchases"
                ],
                figsize=(14, 16),
            )
            _out = mo.vstack([
                mo.md("### Trace Plots"),
                plt.gcf(),
            ])
    _out  
    return


@app.cell
def _(idata, mediation_model):
    # ============================================================
    # Create posterior predictors
    # ============================================================
    # 
    # We create predictors for:
    # 1. Visits given spend (Stage 1)
    # 2. Purchases given visits (Stage 2) - uses the observed visits as input
    #
    # Note: post_fn receives PyTensor tensors, so use pt operations (not np)
    
    _n_samples = 500
    _idx_low = int(round((_n_samples - 1) * 0.05))
    _idx_high = int(round((_n_samples - 1) * 0.95))
    
    # PyTensor-compatible post-processing functions
    _mean_fn = lambda x: x.mean(axis=0)
    _low_fn = lambda x: pt.sort(x, axis=0)[_idx_low]
    _high_fn = lambda x: pt.sort(x, axis=0)[_idx_high]
    
    if idata is not None:
        # Predictor for Stage 1: Spend -> Visits
        # This predicts mu_visits given new spend values
        visits_predictor = create_frozen_predictor(
            model=mediation_model,
            inference_data=idata,
            response_exprs={
                "mu_visits": ("mu_visits", _mean_fn),
                "mu_visits_low": ("mu_visits", _low_fn),
                "mu_visits_high": ("mu_visits", _high_fn),
            },
            input_vars=["spend"],
            num_samples=_n_samples,
            rng=42,
        )
        
        # Predictor for Stage 2: Visits -> Purchases (with time)
        # Note: 'visits' is an observed RV that depends on 'spend' input.
        # The predictor replaces the observed RV with its expected value (mu_visits)
        # for response curve predictions at arbitrary input sizes.
        purchases_predictor = create_frozen_predictor(
            model=mediation_model,
            inference_data=idata,
            response_exprs={
                "mu_purchases": ("mu_purchases", _mean_fn),
                "mu_purchases_low": ("mu_purchases", _low_fn),
                "mu_purchases_high": ("mu_purchases", _high_fn),
            },
            input_vars=["spend", "time_idx"],  # visits comes from mu_visits in the graph
            num_samples=_n_samples,
            rng=42,
        )
    else:
        visits_predictor = None
        purchases_predictor = None
    return visits_predictor, purchases_predictor


@app.cell(hide_code=True)
def _(df):
    # Interactive slider UI for Saturation Explorer
    _spend_range = df["spend"].values
    _min_spend = 0
    _max_spend = 150

    spend_slider = mo.ui.slider(
        start=_min_spend,
        stop=_max_spend,
        value=70,
        step=5,
        label="Spend level:",
        full_width=True,
    )

    show_uncertainty = mo.ui.checkbox(label="Show uncertainty bands", value=True)

    mo.vstack([
        mo.md("## Interactive Saturation Explorer"),
        mo.md("""
        Adjust the spend level to see how it affects:
        1. **Website Visits** (Stage 1: spend → visits)
        2. **Purchases** (Stage 2: visits → purchases)
        
        The model propagates uncertainty through both stages.
        """),
        mo.hstack([show_uncertainty]),
        spend_slider,
    ])
    return spend_slider, show_uncertainty


@app.cell(hide_code=True)
def _(
    df,
    spend_slider,
    show_uncertainty,
    visits_predictor,
    purchases_predictor,
):
    # Interactive saturation curve visualization
    # Shows both Stage 1 (spend -> visits) and Stage 2 (visits -> purchases)

    mo.stop(
        visits_predictor is None,
        mo.md("*Click 'Fit Model' to start sampling*")
    )

    _slider_spend = spend_slider.value / 100.0  # Normalize
    
    # Create spend range for curves
    # Note: Use same size as original data to avoid shape mismatch with observed RV replacement
    _n_points = len(df)  # Same as original data size
    _spend_range = np.linspace(0, 150, _n_points) / 100.0
    
    # Compute visits curve (Stage 1)
    _visits_result = visits_predictor(spend=_spend_range)
    _visits_mean = _visits_result["mu_visits"]
    _visits_low = _visits_result["mu_visits_low"]
    _visits_high = _visits_result["mu_visits_high"]
    
    # Get visits at slider position
    _visits_at_slider = visits_predictor(spend=np.array([_slider_spend]))
    _current_visits = _visits_at_slider["mu_visits"][0]
    
    # Compute purchases curve (Stage 2) - at mean time index
    _mean_time = df["t"].mean()
    _time_array = np.full_like(_spend_range, _mean_time)
    _purchases_result = purchases_predictor(spend=_spend_range, time_idx=_time_array)
    _purchases_mean = _purchases_result["mu_purchases"]
    _purchases_low = _purchases_result["mu_purchases_low"]
    _purchases_high = _purchases_result["mu_purchases_high"]
    
    # Get purchases at slider position
    _purchases_at_slider = purchases_predictor(
        spend=np.array([_slider_spend]), 
        time_idx=np.array([_mean_time])
    )
    _current_purchases = _purchases_at_slider["mu_purchases"][0]

    # === Create visualization ===
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Stage 1: Spend -> Visits ---
    _ax1.plot(_spend_range * 100, _visits_mean, color='#ff7f0e', linewidth=2.5, label='Expected Visits')
    if show_uncertainty.value:
        _ax1.fill_between(
            _spend_range * 100, _visits_low, _visits_high,
            alpha=0.3, color='#ff7f0e', label='90% CI'
        )
    
    # Marker at slider position
    _ax1.axvline(_slider_spend * 100, color='#ff7f0e', linestyle='--', alpha=0.5)
    _ax1.scatter([_slider_spend * 100], [_current_visits], color='#ff7f0e', s=120, 
                 zorder=5, edgecolors='white', linewidths=2)
    _ax1.annotate(
        f'Spend: {_slider_spend * 100:.0f}\nVisits: {_current_visits:.1f}',
        xy=(_slider_spend * 100, _current_visits),
        xytext=(15, 15), textcoords='offset points',
        fontsize=10, color='#ff7f0e',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ff7f0e', alpha=0.8)
    )
    
    # Add observed data points
    _ax1.scatter(df["spend"], df["visits"], color='gray', alpha=0.4, s=20, label='Observed')
    
    _ax1.set_xlabel('Ad Spend')
    _ax1.set_ylabel('Website Visits')
    _ax1.set_title('Stage 1: Spend → Visits (Saturation)')
    _ax1.legend(loc='lower right')
    _ax1.grid(True, alpha=0.3)
    
    # --- Stage 2: Spend -> Purchases (through visits) ---
    _ax2.plot(_spend_range * 100, _purchases_mean, color='#2ca02c', linewidth=2.5, label='Expected Purchases')
    if show_uncertainty.value:
        _ax2.fill_between(
            _spend_range * 100, _purchases_low, _purchases_high,
            alpha=0.3, color='#2ca02c', label='90% CI'
        )
    
    # Marker at slider position
    _ax2.axvline(_slider_spend * 100, color='#2ca02c', linestyle='--', alpha=0.5)
    _ax2.scatter([_slider_spend * 100], [_current_purchases], color='#2ca02c', s=120,
                 zorder=5, edgecolors='white', linewidths=2)
    _ax2.annotate(
        f'Spend: {_slider_spend * 100:.0f}\nPurchases: {_current_purchases:.1f}',
        xy=(_slider_spend * 100, _current_purchases),
        xytext=(15, 15), textcoords='offset points',
        fontsize=10, color='#2ca02c',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2ca02c', alpha=0.8)
    )
    
    # Add observed data points
    _ax2.scatter(df["spend"], df["purchases"], color='gray', alpha=0.4, s=20, label='Observed')
    
    _ax2.set_xlabel('Ad Spend')
    _ax2.set_ylabel('Purchases')
    _ax2.set_title('Stage 2: Spend → Purchases (via Visits)')
    _ax2.legend(loc='lower right')
    _ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    mo.vstack([
        mo.md("### Saturation Curves: Two-Stage Mediation"),
        plt.gcf(),
        mo.md(f"""
        **At spend = {_slider_spend * 100:.0f}:**
        - Expected visits: {_current_visits:.1f}
        - Expected purchases: {_current_purchases:.1f}
        - Implied conversion rate: {_current_purchases / _current_visits:.3f} purchases/visit
        """),
    ])
    return


@app.cell(hide_code=True)
def _(df, idata):
    # Posterior predictive check: compare observed vs predicted
    if idata is None:
        _out = mo.md("*Click 'Fit Model' to start sampling*")
    else:
        _visits_post = idata.posterior["mu_visits"].mean(dim=["chain", "draw"]).values
        _purchases_post = idata.posterior["mu_purchases"].mean(dim=["chain", "draw"]).values
        
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Visits: observed vs predicted
        _ax1.scatter(df["visits"], _visits_post, alpha=0.6, color='#ff7f0e')
        _max_val = max(df["visits"].max(), _visits_post.max())
        _ax1.plot([0, _max_val], [0, _max_val], 'k--', alpha=0.5, label='Perfect fit')
        _ax1.set_xlabel("Observed Visits")
        _ax1.set_ylabel("Predicted Visits")
        _ax1.set_title("Visits: Observed vs Predicted")
        _ax1.legend()
        _ax1.grid(True, alpha=0.3)
        
        # Purchases: observed vs predicted
        _ax2.scatter(df["purchases"], _purchases_post, alpha=0.6, color='#2ca02c')
        _max_val = max(df["purchases"].max(), _purchases_post.max())
        _ax2.plot([0, _max_val], [0, _max_val], 'k--', alpha=0.5, label='Perfect fit')
        _ax2.set_xlabel("Observed Purchases")
        _ax2.set_ylabel("Predicted Purchases")
        _ax2.set_title("Purchases: Observed vs Predicted")
        _ax2.legend()
        _ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        _out = mo.vstack([
            mo.md("## Model Fit: Observed vs Predicted"),
            plt.gcf(),
        ])
    _out
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
