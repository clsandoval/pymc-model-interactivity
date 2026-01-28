# /// script
# dependencies = [
#     "marimo>=0.19.0",
#     "pymc>=5.10.0",
#     "pymc-marketing>=0.17.1",
#     "arviz",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "pytensor",
#     "pyzmq>=27.1.0",
#     "jax==0.9.0",
#     "numpyro==0.19.0",
#     "wigglystuff>=0.2.17",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import arviz as az
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    from scipy.interpolate import interp1d
    from wigglystuff import ChartPuck
    
    # Import components from pymc-marketing
    from pymc_marketing.mmm.hsgp import SoftPlusHSGP
    from pymc_marketing.mmm.transformers import geometric_adstock, hill_function

    # Import local data generation
    from generate_data import generate_marketing_funnel_data, TRUE_PARAMS

    # Import posterior predictor
    from frozen_predictor import create_frozen_predictor


@app.cell(hide_code=True)
def _():
    # Model Structure Diagram
    _diagram = """
    flowchart TD
        subgraph Baseline[Time-Varying Baseline]
            baseline["baseline(t)"]
        end

        subgraph Controls[Control Variables]
            ctrl_effect["Inflation"]
        end

        subgraph Direct[Direct Channel]
            effect_direct["Spend"]
        end

        subgraph Upper[Upper Funnel Channel]
            spend_upper["Spend"]
        end

        subgraph Lower[Lower Funnel Channel]
            spend_upper --> CPM
            spend_lower["Spend"] --> effect_lower
            CPM --> effect_lower
            effect_lower["impressions"]
        end

        baseline --> Sales
        ctrl_effect --> Sales
        effect_direct --> Sales
        spend_upper --> Sales
        effect_lower --> Sales
    """

    mo.md(f"""
    # Interactive Counterfactual Budget Analysis

    This notebook demonstrates interactive counterfactual analysis for a Marketing Mix Model.

    **Features:**
    - **Plot 1**: Interactive budget layout editor using ChartPuck - drag pucks vertically to adjust spend
    - **Plot 2**: Counterfactual sales prediction based on confirmed budget allocation

    ## Model Structure

    {mo.mermaid(_diagram).text}

    **Key equations:**
    - `Sales(t) = intercept_base * baseline(t) + beta_inflation * inflation(t) + channel_effects + noise`
    - `CPM_lower(t) = CPM_base * exp(-gamma * upper_funnel_effect(t))`
    - `impressions_lower(t) = spend_lower(t) / CPM_lower(t)`
    """)
    return


@app.cell(hide_code=True)
def _():
    # Load synthetic data
    df, true_params = generate_marketing_funnel_data(n_periods=104, seed=42)
    return (df,)


@app.cell(hide_code=True)
def _(df):
    # Data Visualization
    _fig, _axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot media spends
    _axes[0].plot(df["date"], df["spend_direct"], label="Direct", alpha=0.8)
    _axes[0].plot(df["date"], df["spend_upper"], label="Upper Funnel", alpha=0.8)
    _axes[0].plot(df["date"], df["spend_lower"], label="Lower Funnel", alpha=0.8)
    _axes[0].set_ylabel("Spend")
    _axes[0].legend()
    _axes[0].set_title("Media Spends Over Time")

    # Plot inflation
    _axes[1].plot(df["date"], df["inflation"], color="orange", alpha=0.8)
    _axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    _axes[1].set_ylabel("Inflation (deviation)")
    _axes[1].set_title("Inflation Control Variable")

    # Plot sales
    _axes[2].plot(df["date"], df["sales"], color="green", alpha=0.8)
    _axes[2].plot(df["date"], df["baseline_true"], color="gray", linestyle="--", 
                  alpha=0.5, label="True baseline")
    _axes[2].set_ylabel("Sales")
    _axes[2].set_xlabel("Date")
    _axes[2].set_title("Sales Over Time")
    _axes[2].legend()

    plt.tight_layout()

    plt.gca()
    return


@app.cell(hide_code=True)
def _(df):
    # Model Definition
    # Using pymc-marketing's built-in transformers:
    # - hill_function(x, slope, kappa) for saturation
    # - geometric_adstock(x, alpha, l_max, normalize) for carry-over effects

    # Prepare data
    n_obs = len(df)
    t_idx = np.arange(n_obs)

    # Normalize spends for saturation
    spend_direct_data = df["spend_direct"].values / 100.0
    spend_upper_data = df["spend_upper"].values / 100.0
    spend_lower_data = df["spend_lower"].values / 100.0
    inflation_data = df["inflation"].values
    sales_data = df["sales"].values

    l_max = 8  # Maximum adstock lag

    # Build the model
    # Configure HSGP from data (automatically sets good defaults for priors)
    hsgp = SoftPlusHSGP.parameterize_from_data(X=t_idx, dims="date")

    with pm.Model(coords={"date": df["date"].values}) as funnel_model:
        # === Data inputs ===
        spend_direct = pm.Data("spend_direct", spend_direct_data, dims="date")
        spend_upper = pm.Data("spend_upper", spend_upper_data, dims="date")
        spend_lower = pm.Data("spend_lower", spend_lower_data, dims="date")
        inflation = pm.Data("inflation", inflation_data, dims="date")

        # === Time-varying intercept using SoftPlusHSGP ===
        intercept_base = pm.HalfNormal("intercept_base", sigma=500)

        # HSGP for time-varying component (normalized to mean ~1)
        # parameterize_from_data sets up appropriate priors based on data scale
        time_data = pm.Data("time_idx", t_idx, dims="date")
        baseline_gp = hsgp.register_data(time_data).create_variable("baseline_gp")

        baseline = pm.Deterministic("baseline", intercept_base * baseline_gp, dims="date")

        # === Control variable ===
        beta_inflation = pm.Normal("beta_inflation", mu=0, sigma=50)
        control_effect = beta_inflation * inflation

        # === Direct channel ===
        slope_direct = pm.Normal("slope_direct", mu=2, sigma=0.5)
        kappa_direct = pm.Beta("kappa_direct", alpha=8, beta=8)
        alpha_direct = pm.Beta("alpha_direct", alpha=8, beta=8)
        beta_direct = pm.Normal("beta_direct", mu=200, sigma=50)

        # Store saturation as Deterministic for later extraction
        direct_saturated = pm.Deterministic(
            "direct_saturated",
            hill_function(spend_direct, slope_direct, kappa_direct),
            dims="date"
        )
        direct_adstocked = geometric_adstock(direct_saturated, alpha=alpha_direct, l_max=l_max, normalize=True)
        direct_effect = pm.Deterministic("direct_effect", beta_direct * direct_adstocked, dims="date")

        # === Upper funnel channel ===
        slope_upper = pm.Normal("slope_upper", mu=2, sigma=0.5)
        kappa_upper = pm.Beta("kappa_upper", alpha=8, beta=8)
        alpha_upper = pm.Beta("alpha_upper", alpha=8, beta=8)
        beta_upper = pm.Normal("beta_upper", mu=200, sigma=50)

        # Store saturation as Deterministic
        upper_saturated = pm.Deterministic(
            "upper_saturated",
            hill_function(spend_upper, slope_upper, kappa_upper),
            dims="date"
        )
        upper_adstocked = geometric_adstock(upper_saturated, alpha=alpha_upper, l_max=l_max, normalize=True)
        upper_effect = pm.Deterministic("upper_effect", beta_upper * upper_adstocked, dims="date")

        # === Funnel interaction: upper funnel affects lower funnel CPM ===
        cpm_base = pm.Normal("cpm_base", mu=10, sigma=3)
        gamma = pm.HalfNormal("gamma", sigma=1)

        # CPM decreases with upper funnel activity (uses saturation, not adstocked for instant response)
        # For saturation curves, we want CPM based on saturation level
        cpm_lower = pm.Deterministic(
            "cpm_lower", 
            cpm_base * pt.exp(-gamma * upper_saturated),
            dims="date"
        )

        # === Lower funnel channel (impressions-based) ===
        # impressions = spend / CPM
        impressions_lower = spend_lower * 1000.0 / cpm_lower  # Un-normalize spend for division
        impressions_norm = impressions_lower / 100.0  # Normalize impressions

        slope_lower = pm.Normal("slope_lower", mu=2, sigma=0.5)
        kappa_lower = pm.Beta("kappa_lower", alpha=8, beta=8)
        alpha_lower = pm.Beta("alpha_lower", alpha=8, beta=8)
        beta_lower = pm.Normal("beta_lower", mu=200, sigma=50)

        # Store saturation as Deterministic
        lower_saturated = pm.Deterministic(
            "lower_saturated",
            hill_function(impressions_norm, slope_lower, kappa_lower),
            dims="date"
        )
        lower_adstocked = geometric_adstock(lower_saturated, alpha=alpha_lower, l_max=l_max, normalize=True)
        lower_effect = pm.Deterministic("lower_effect", beta_lower * lower_adstocked, dims="date")

        # === Total effects (sum over dates) for response curves ===
        total_direct_effect = pm.Deterministic("total_direct_effect", direct_effect.sum())
        total_upper_effect = pm.Deterministic("total_upper_effect", upper_effect.sum())
        total_lower_effect = pm.Deterministic("total_lower_effect", lower_effect.sum())

        # === Total prediction ===
        mu = pm.Deterministic(
            "mu",
            baseline + control_effect + direct_effect + upper_effect + lower_effect
        )

        # === Likelihood ===
        sigma = pm.HalfNormal("sigma", sigma=100)
        sales_obs = pm.Normal("sales", mu=mu, sigma=sigma, observed=sales_data)
    funnel_model
    return (funnel_model,)


@app.cell(hide_code=True)
def _():
    # Model fitting controls
    fit_button = mo.ui.run_button(label="Fit Model (this may take a few minutes)")

    mo.vstack([
        mo.md("## Model Fitting"),
        mo.md("Click the button below to start MCMC sampling. This typically takes 2-5 minutes."),
        fit_button,
    ])
    return (fit_button,)


@app.cell(hide_code=True)
def _(fit_button, funnel_model):
    if not fit_button.value:
        mo.md("*Click 'Fit Model' to start sampling*")
        idata = None
    else:
        with funnel_model:
            idata = pm.sample(
                draws=500,
                tune=500,
                chains=4,
                random_seed=42,
                return_inferencedata=True,
                progressbar=True,
                nuts_sampler="numpyro"
            )
        mo.md("**Model fitting complete!**")
    return (idata,)


@app.cell(hide_code=True)
def _(idata):
    convergence_summary = None
    if idata:
        # Convergence summary (numerical check)
        convergence_summary = az.summary(idata, var_names=[
            "intercept_base", "beta_inflation",
            "beta_direct", "slope_direct", "kappa_direct", "alpha_direct",
            "beta_upper", "slope_upper", "kappa_upper", "alpha_upper",
            "beta_lower", "slope_lower", "kappa_lower", "alpha_lower",
            "cpm_base", "gamma", "sigma"
        ])
    return (convergence_summary,)


@app.cell(hide_code=True)
def _():
    # Trace plot controls
    trace_plot_control = mo.ui.radio(
        options={"Hide":False, "Show": True},
        value="Hide",
        label="Show convergence diagnostics"
    )
    trace_plot_control
    return (trace_plot_control,)


@app.cell(hide_code=True)
def _(convergence_summary, trace_plot_control):
    _out = None
    if convergence_summary is not None:
        _max_rhat = convergence_summary["r_hat"].max()
        _min_ess = convergence_summary["ess_bulk"].min()

        if _max_rhat < 1.01:
            _status = mo.md(f"**Status: All parameters converged** (max R-hat = {_max_rhat:.4f}, min ESS = {_min_ess:.0f})")
        else:
            _problematic = convergence_summary[convergence_summary["r_hat"] >= 1.01].index.tolist()
            _status = mo.md(f"**Warning: Some parameters may not have converged**\n\nmax R-hat = {_max_rhat:.4f}. Check: {_problematic}")

        if trace_plot_control.value:
            _out = mo.vstack([
                mo.md("## Convergence Diagnostics"),
                _status,
                mo.md("### Parameter Summary"),
                mo.ui.table(convergence_summary.reset_index().rename(columns={"index": "parameter"})),
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
            _fig = plt.figure(figsize=(14, 20))
            az.plot_trace(
                idata, 
                var_names=[
                    "intercept_base", "beta_inflation",
                    "beta_direct", "alpha_direct",
                    "beta_upper", "alpha_upper",
                    "beta_lower", "alpha_lower",
                    "cpm_base", "gamma", "sigma"
                ],
                figsize=(14, 20),
            )
            _out = mo.vstack([
                mo.md("### Trace Plots"),
                plt.gcf(),
            ])
    _out  
    return


@app.cell
def _(funnel_model, idata):
    # Create compiled pytensor predictor for sales time series
    if idata:
        sales_timeseries_predictor = create_sales_timeseries_predictor(
            funnel_model, idata, n_samples=500
        )
    else:
        sales_timeseries_predictor = None
    return (sales_timeseries_predictor,)


# ============================================================================
# INTERACTIVE COUNTERFACTUAL ANALYSIS
# ============================================================================

@app.cell
def _(df):
    # Initialize budget state with historical values
    # This stores the confirmed budget allocation for all channels
    
    # Get historical spend values (normalized)
    historical_spend = {
        'direct': df["spend_direct"].values / 100.0,
        'upper': df["spend_upper"].values / 100.0,
        'lower': df["spend_lower"].values / 100.0,
    }
    
    # State to track confirmed budget (starts with historical values)
    get_budget_state, set_budget_state = mo.state({
        'direct': historical_spend['direct'].copy(),
        'upper': historical_spend['upper'].copy(),
        'lower': historical_spend['lower'].copy(),
    })
    
    return historical_spend, get_budget_state, set_budget_state


@app.cell
def _():
    # Channel selector for interactive editing
    budget_channel_selector = mo.ui.dropdown(
        options={"Direct Channel": "direct", "Upper Funnel": "upper", "Lower Funnel": "lower"},
        value="Direct Channel",
        label="Select channel to edit:"
    )
    
    return (budget_channel_selector,)


@app.cell
def _(budget_channel_selector, df, get_budget_state, historical_spend, set_budget_state):
    # Interactive Budget Layout Editor using ChartPuck
    # - Shows all channels, but only the selected one is editable
    # - Non-selected channels are gray and transparent
    # - Pucks constrained to vertical movement only (fixed x positions)
    
    mo.stop(
        sales_timeseries_predictor is None,
        mo.md("*Click 'Fit Model' above to enable interactive analysis*")
    )
    
    _selected_channel = budget_channel_selector.value
    _current_budget = get_budget_state()
    _n_periods = len(df)
    
    # Subsample weeks for puck placement (every 4th week for usability)
    _puck_step = 4
    _puck_indices = list(range(0, _n_periods, _puck_step))
    _n_pucks = len(_puck_indices)
    
    # Channel colors
    _channel_colors = {
        'direct': '#1f77b4',
        'upper': '#ff7f0e', 
        'lower': '#2ca02c',
    }
    
    # Get the current puck y-values for the selected channel
    _puck_y_values = [_current_budget[_selected_channel][i] * 100.0 for i in _puck_indices]
    _puck_x_values = list(_puck_indices)  # Fixed x positions (week indices)
    
    # Max spend for y-axis scaling
    _max_spend = max(
        historical_spend['direct'].max(),
        historical_spend['upper'].max(),
        historical_spend['lower'].max(),
    ) * 100.0 * 1.5  # 50% headroom
    
    def draw_budget_editor(ax, widget):
        """Draw the budget editor chart with all channels visible."""
        # Get puck positions (y values may have changed, x stays fixed)
        puck_y = list(widget.y)
        
        # Channel properties
        channels = {
            'direct': {'color': '#1f77b4', 'label': 'Direct Channel'},
            'upper': {'color': '#ff7f0e', 'label': 'Upper Funnel'},
            'lower': {'color': '#2ca02c', 'label': 'Lower Funnel'},
        }
        
        selected = _selected_channel
        
        # Plot each channel
        for ch, props in channels.items():
            is_selected = (ch == selected)
            
            if is_selected:
                # Interpolate from puck positions to full time series
                puck_x = _puck_indices
                interp_fn = interp1d(puck_x, puck_y, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
                full_y = interp_fn(np.arange(_n_periods))
                
                # Plot with full color
                ax.plot(range(_n_periods), full_y, 
                       color=props['color'], linewidth=2.5, alpha=1.0,
                       label=f"{props['label']} (editing)")
            else:
                # Plot non-selected channels as gray and transparent
                full_y = _current_budget[ch] * 100.0
                ax.plot(range(_n_periods), full_y,
                       color='gray', linewidth=1.5, alpha=0.3,
                       label=props['label'])
        
        # Styling
        ax.set_xlim(-2, _n_periods + 2)
        ax.set_ylim(0, _max_spend)
        ax.set_xlabel('Week')
        ax.set_ylabel('Spend (normalized × 100)')
        ax.set_title(f'Budget Layout Editor - Editing: {channels[selected]["label"]}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Create ChartPuck with vertical-only constraint
    # The pucks are placed at fixed x positions (week indices)
    budget_puck = ChartPuck.from_callback(
        draw_fn=draw_budget_editor,
        x_bounds=(0, _n_periods),
        y_bounds=(0, _max_spend),
        figsize=(14, 5),
        x=_puck_x_values,
        y=_puck_y_values,
        puck_color=_channel_colors[_selected_channel],
        puck_radius=8,
        drag_x_bounds=(None, None),  # Allow x drag (we'll ignore it in the draw function)
    )
    
    budget_widget = mo.ui.anywidget(budget_puck)
    
    return (budget_widget, _puck_indices, _n_periods, _max_spend)


@app.cell
def _(budget_channel_selector, budget_widget, get_budget_state, historical_spend, set_budget_state, _puck_indices, _n_periods):
    # Confirm and Reset buttons
    
    def on_confirm_click(_):
        """Lock in the current puck positions as the confirmed budget."""
        _selected = budget_channel_selector.value
        _current = get_budget_state()
        
        # Get current puck y-values
        puck_y = list(budget_widget.y)
        
        # Interpolate to full time series
        interp_fn = interp1d(_puck_indices, puck_y, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        full_spend = interp_fn(np.arange(_n_periods)) / 100.0  # Un-normalize
        full_spend = np.clip(full_spend, 0, None)  # Ensure non-negative
        
        # Update state
        new_budget = {**_current}
        new_budget[_selected] = full_spend
        set_budget_state(new_budget)
    
    def on_reset_click(_):
        """Reset ALL channels to historical values."""
        set_budget_state({
            'direct': historical_spend['direct'].copy(),
            'upper': historical_spend['upper'].copy(),
            'lower': historical_spend['lower'].copy(),
        })
    
    confirm_button = mo.ui.button(
        label="✓ Confirm Budget",
        on_click=on_confirm_click,
        kind="success"
    )
    
    reset_button = mo.ui.button(
        label="↺ Reset All Channels",
        on_click=on_reset_click,
        kind="warn"
    )
    
    return confirm_button, reset_button


@app.cell
def _(budget_channel_selector, budget_widget, confirm_button, reset_button):
    # Display Plot 1: Interactive Budget Editor
    mo.vstack([
        mo.md("## Plot 1: Interactive Budget Layout Editor"),
        mo.md("""
        **Instructions:**
        - Select a channel from the dropdown to edit its spend pattern
        - Drag pucks **vertically** to adjust spend at that time point
        - Non-selected channels are shown in gray (read-only)
        - Click **Confirm Budget** to lock in your changes
        - Click **Reset All Channels** to restore historical values
        """),
        mo.hstack([budget_channel_selector, confirm_button, reset_button], gap=2),
        budget_widget,
    ])
    return


@app.cell
def _(df, get_budget_state, historical_spend, sales_timeseries_predictor):
    # Plot 2: Counterfactual Sales Prediction
    # This plot reacts to the confirmed budget state
    
    mo.stop(
        sales_timeseries_predictor is None,
        mo.md("*Click 'Fit Model' above to enable interactive analysis*")
    )
    
    _confirmed_budget = get_budget_state()
    
    # Get predicted sales for the confirmed budget
    _counterfactual_result = sales_timeseries_predictor(
        _confirmed_budget['direct'],
        _confirmed_budget['upper'],
        _confirmed_budget['lower']
    )
    _pred_mean = _counterfactual_result['mean']
    _pred_low = _counterfactual_result['low']
    _pred_high = _counterfactual_result['high']
    
    # Get baseline prediction (historical spend)
    _baseline_result = sales_timeseries_predictor(
        historical_spend['direct'],
        historical_spend['upper'],
        historical_spend['lower']
    )
    _baseline_mean = _baseline_result['mean']
    
    # Calculate summary statistics
    _total_baseline = _baseline_mean.sum()
    _total_counterfactual = _pred_mean.sum()
    _total_diff = _total_counterfactual - _total_baseline
    _pct_diff = (_total_diff / _total_baseline) * 100 if _total_baseline != 0 else 0
    
    # Determine if budget has changed from historical
    _budget_changed = any(
        not np.allclose(_confirmed_budget[ch], historical_spend[ch], rtol=1e-3)
        for ch in ['direct', 'upper', 'lower']
    )
    
    # Create visualization
    _fig, _ax = plt.subplots(figsize=(14, 7))
    _dates = df["date"].values
    
    # Plot actual historical sales in gray
    _ax.scatter(_dates, df["sales"].values, color='gray', s=15, alpha=0.4, 
                label='Historical Sales (Observed)', zorder=2)
    
    # Determine color based on lift/drop
    if _total_diff >= 0:
        _line_color = '#2ca02c'  # Green for positive
        _fill_color = '#90EE90'
    else:
        _line_color = '#d62728'  # Red for negative
        _fill_color = '#FFB6C1'
    
    if _budget_changed:
        # Plot counterfactual prediction
        _ax.plot(_dates, _pred_mean, color=_line_color, linewidth=2.5,
                 label=f'Counterfactual Prediction ({_pct_diff:+.1f}%)')
        _ax.fill_between(_dates, _pred_low, _pred_high, color=_line_color, alpha=0.2,
                         label='90% Credible Interval')
        
        # Add shaded area showing lift/drop vs baseline
        _ax.fill_between(_dates, _baseline_mean, _pred_mean, 
                        color=_fill_color, alpha=0.4, label='Lift/Drop vs Baseline')
        
        # Plot baseline (dashed)
        _ax.plot(_dates, _baseline_mean, color='gray', linestyle='--', linewidth=1.5,
                 alpha=0.7, label='Baseline (Historical Budget)')
    else:
        # No changes yet - just show baseline
        _ax.plot(_dates, _baseline_mean, color='#1f77b4', linewidth=2,
                 label='Baseline Prediction')
        _ax.fill_between(_dates, _baseline_result['low'], _baseline_result['high'], 
                        color='#1f77b4', alpha=0.2, label='90% Credible Interval')
    
    _ax.set_xlabel('Date')
    _ax.set_ylabel('Sales')
    _ax.set_title('Counterfactual Sales Prediction')
    _ax.legend(loc='upper left')
    _ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Summary annotation
    if _budget_changed:
        _summary_text = f"Total Sales Change: {_total_diff:+,.0f} ({_pct_diff:+.1f}%)"
        _ax.annotate(
            _summary_text,
            xy=(0.98, 0.02), xycoords='axes fraction',
            fontsize=12, color=_line_color, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=_line_color, alpha=0.9)
        )
    
    mo.vstack([
        mo.md("## Plot 2: Counterfactual Sales Prediction"),
        mo.md("""
        This plot shows predicted sales based on your **confirmed** budget allocation.
        - **Gray dots**: Historical observed sales
        - **Colored line**: Counterfactual prediction with your budget
        - **Shaded area**: Difference from baseline (green = lift, red = drop)
        
        **Note:** This plot updates only when you click "Confirm Budget" above.
        """),
        plt.gcf(),
        mo.md(f"""
        **Summary:**
        - Baseline total sales: {_total_baseline:,.0f}
        - Counterfactual total sales: {_total_counterfactual:,.0f}
        - **Difference: {_total_diff:+,.0f} ({_pct_diff:+.1f}%)**
        """) if _budget_changed else mo.md("*Adjust and confirm a budget in Plot 1 to see counterfactual predictions*"),
    ])
    return


@app.function
def create_sales_timeseries_predictor(model, inference_data, n_samples=200):
    """
    Create a compiled pytensor function for predicted sales time series.

    Returns the full mu (predicted sales) time series with uncertainty,
    allowing counterfactual analysis of different spend scenarios.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model (must have mu Deterministic node)
    inference_data : InferenceData
        ArviZ InferenceData with posterior samples
    n_samples : int
        Number of posterior samples to use

    Returns
    -------
    predict_fn : callable
        Function(spend_direct, spend_upper, spend_lower) -> dict with 'mean', 'low', 'high'
        Each is a 1D array of length n_dates (time series of predicted sales)
    """
    _input_vars = ["spend_direct", "spend_upper", "spend_lower"]

    # Post-processing functions for mean and HDI
    _mean_fn = lambda x: x.mean(axis=0)
    _idx_low = int(round((n_samples - 1) * 0.05))
    _idx_high = int(round((n_samples - 1) * 0.95))
    _low_fn = lambda x, idx=_idx_low: pt.sort(x, axis=0)[idx]
    _high_fn = lambda x, idx=_idx_high: pt.sort(x, axis=0)[idx]

    _response_exprs = {
        'mean': ("mu", _mean_fn),
        'low': ("mu", _low_fn),
        'high': ("mu", _high_fn),
    }

    # Use the general predictor
    _general_predict_fn = create_frozen_predictor(
        model=model,
        inference_data=inference_data,
        response_exprs=_response_exprs,
        input_vars=_input_vars,
        num_samples=n_samples,
    )

    def predict_fn(spend_direct, spend_upper, spend_lower):
        """
        Compute predicted sales time series for given spend scenarios.

        Parameters
        ----------
        spend_direct, spend_upper, spend_lower : array-like
            Spend time series (normalized) for each channel. Should be 1D arrays
            of the same length representing spend over time.

        Returns
        -------
        predictions : dict
            Keys: 'mean', 'low', 'high'
            Values: 1D arrays of predicted sales (length = n_dates)
        """
        return _general_predict_fn(
            spend_direct=np.asarray(spend_direct),
            spend_upper=np.asarray(spend_upper),
            spend_lower=np.asarray(spend_lower),
        )

    return predict_fn


if __name__ == "__main__":
    app.run()
