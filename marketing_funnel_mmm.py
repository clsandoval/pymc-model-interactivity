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
    import pytensor
    import pytensor.tensor as pt
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
    # Interactive model exploration with PyMC

    This notebook demonstrates a Marketing Mix Model with a marketing funnel structure:

    - **Direct Channel**: Standard media effect (spend -> saturation -> adstock -> sales)
    - **Upper Funnel**: Brand awareness channel that also improves lower funnel efficiency
    - **Lower Funnel**: Performance channel where impressions depend on CPM (affected by upper funnel)

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
    with mo.redirect_stdout():
    # Model fitting cell
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
    # Create compiled pytensor predictors for saturation curves
    # Two versions: mean-only (fast) and with HDI (for uncertainty bands)
    # Post-processing (mean/percentile) is computed inside pytensor for efficiency
    if idata:
        saturation_predictor_mean, _ = create_saturation_predictor(
            funnel_model, idata, n_samples=200, include_hdi=False
        )
        saturation_predictor_hdi, _ = create_saturation_predictor(
            funnel_model, idata, n_samples=200, include_hdi=True
        )
    else:
        saturation_predictor_mean = None
        saturation_predictor_hdi = None
    return saturation_predictor_hdi, saturation_predictor_mean


@app.cell(hide_code=True)
def _(df):
    # Interactive slider UI
    _spend_range = df["spend_direct"].values
    _min_spend = float(_spend_range.min()) * 0.5
    _max_spend = float(_spend_range.max()) * 1.5
    _default_spend = float(_spend_range.mean())

    channel_selector = mo.ui.dropdown(
        options={"Direct Channel":"direct", "Upper Funnel":"upper", "Lower Funnel":"lower"},
        value="Direct Channel",
        label="Select channel to analyze:"
    )

    spend_slider = mo.ui.slider(
        start=0,
        stop=150,
        value=70,
        step=5,
        label="Spend level (normalized):",
        full_width=True,
    )

    show_uncertainty = mo.ui.checkbox(label="Show uncertainty", value=True)

    mo.vstack([
        mo.md("## Interactive Saturation Explorer"),
        mo.md("Adjust the spend level to see how saturation changes. The model uses frozen posterior samples to compute saturation quickly."),
        mo.hstack([channel_selector, show_uncertainty]),
        spend_slider,
    ])
    return channel_selector, show_uncertainty, spend_slider


@app.cell(hide_code=True)
def _(
    channel_selector,
    df,
    saturation_predictor_hdi,
    saturation_predictor_mean,
    show_uncertainty,
    spend_slider,
):
    # Interactive saturation curve visualization - showing ALL channels
    # Uses optimized predictors that compute mean/HDI inside pytensor

    mo.stop(
        saturation_predictor_mean is None,
        mo.md("*Click 'Fit Model' to start sampling*")
    )

    # Choose predictor based on uncertainty toggle
    _predictor = saturation_predictor_hdi if show_uncertainty.value else saturation_predictor_mean
    _selected_channel = channel_selector.value

    # Slider controls ONLY the selected channel's spend level
    _slider_spend = spend_slider.value / 100.0

    # Baseline spend levels for each channel (mean historical values)
    _baseline_spends = {
        'direct': df["spend_direct"].mean() / 100.0,
        'upper': df["spend_upper"].mean() / 100.0,
        'lower': df["spend_lower"].mean() / 100.0,
    }

    # Determine current spend levels for each channel
    _current_spends = {}
    for _ch in ['direct', 'upper', 'lower']:
        if _ch == _selected_channel:
            _current_spends[_ch] = _slider_spend
        else:
            _current_spends[_ch] = _baseline_spends[_ch]

    # Channel display properties
    _channels = {
        'direct': {'color': '#1f77b4', 'label': 'Direct Channel'},
        'upper': {'color': '#ff7f0e', 'label': 'Upper Funnel'},
        'lower': {'color': '#2ca02c', 'label': 'Lower Funnel'},
    }

    # === Compute saturation curves ===
    # For each channel, vary its spend while keeping others at current levels
    _spend_range = np.linspace(0, 150, 50) / 100.0  # Normalized spend values

    # Compute response curves for each channel using vectorized calls
    # Direct: vary direct spend, keep upper/lower at current
    _result_direct = _predictor(
        _spend_range, 
        _current_spends['upper'], 
        _current_spends['lower']
    )

    # Upper: vary upper spend, keep direct/lower at current
    _result_upper = _predictor(
        _current_spends['direct'], 
        _spend_range, 
        _current_spends['lower']
    )

    # Lower: vary lower spend at current upper level
    _result_lower = _predictor(
        _current_spends['direct'], 
        _current_spends['upper'], 
        _spend_range
    )

    # Extract mean responses (already computed in pytensor)
    _responses = {
        'direct': _result_direct['direct'],
        'upper': _result_upper['upper'],
        'lower': _result_lower['lower'],
    }

    # Get effects at current spend levels (for markers)
    _current_result = _predictor(
        _current_spends['direct'],
        _current_spends['upper'],
        _current_spends['lower']
    )
    _marker_effects = {
        'direct': np.asarray(_current_result['direct']).item(),
        'upper': np.asarray(_current_result['upper']).item(),
        'lower': np.asarray(_current_result['lower']).item(),
    }

    # Extract uncertainty bands if using HDI predictor (already computed in pytensor)
    _uncertainty_bands = {}
    if show_uncertainty.value:
        _uncertainty_bands = {
            'direct': {'low': _result_direct['direct_low'], 'high': _result_direct['direct_high']},
            'upper': {'low': _result_upper['upper_low'], 'high': _result_upper['upper_high']},
            'lower': {'low': _result_lower['lower_low'], 'high': _result_lower['lower_high']},
        }

    # === Create visualization ===
    _fig, _ax = plt.subplots(figsize=(12, 7))

    # Plot saturation curves for all channels
    for _ch, _props in _channels.items():
        _is_selected = (_ch == _selected_channel)
        _linewidth = 3 if _is_selected else 1.5
        _alpha_line = 1.0 if _is_selected else 0.6
        _alpha_fill = 0.3 if _is_selected else 0.1

        _ax.plot(
            _spend_range * 100, _responses[_ch],
            color=_props['color'],
            linewidth=_linewidth,
            alpha=_alpha_line,
            label=_props['label']
        )

        if show_uncertainty.value and _ch in _uncertainty_bands:
            _ax.fill_between(
                _spend_range * 100,
                _uncertainty_bands[_ch]['low'],
                _uncertainty_bands[_ch]['high'],
                alpha=_alpha_fill,
                color=_props['color']
            )

    # Vertical line only for the selected channel's slider position
    _sel_color = _channels[_selected_channel]['color']
    _ax.axvline(_slider_spend * 100, color=_sel_color, linestyle='--', alpha=0.5)

    # Add markers for all channels at their respective spend levels
    for _ch, _props in _channels.items():
        _is_selected = (_ch == _selected_channel)
        _marker_size = 120 if _is_selected else 60
        _zorder = 6 if _is_selected else 5
        _marker_style = 'o' if _is_selected else 's'
        _ax.scatter(
            [_current_spends[_ch] * 100], [_marker_effects[_ch]],
            color=_props['color'], s=_marker_size, zorder=_zorder,
            edgecolors='white', linewidths=1.5, marker=_marker_style
        )

    # Annotate selected channel with slider value
    _ax.annotate(
        f'{_channels[_selected_channel]["label"]}\nSpend: {_slider_spend * 100:.0f}\nEffect: {_marker_effects[_selected_channel]:.1f}',
        xy=(_slider_spend * 100, _marker_effects[_selected_channel]),
        xytext=(15, 15), textcoords='offset points',
        fontsize=10, color=_sel_color,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=_sel_color, alpha=0.8)
    )

    _ax.set_xlabel('Spend (normalized)')
    _ax.set_ylabel('Channel Effect on Sales')
    _ax.set_title('Saturation Curves with Funnel Interaction (Upper affects Lower CPM)')
    _ax.legend(loc='upper left')
    _ax.grid(True, alpha=0.3)

    mo.md(f"### Saturation Curves - All Channels")
    plt.gca()
    return


@app.cell
def _(funnel_model, idata):
    # Create compiled pytensor predictors for response curves (total effects)
    # Two versions: mean-only (fast) and with HDI (for uncertainty bands)
    # Post-processing (mean/percentile) is computed inside pytensor for efficiency
    if idata:
        response_predictor_mean, _ = create_response_predictor(
            funnel_model, idata, n_samples=200, include_hdi=False
        )
        response_predictor_hdi, _ = create_response_predictor(
            funnel_model, idata, n_samples=200, include_hdi=True
        )
    else:
        response_predictor_mean = None
        response_predictor_hdi = None
    return response_predictor_hdi, response_predictor_mean


@app.cell(hide_code=True)
def _():
    # Interactive slider UI for Response Curves
    response_channel_selector = mo.ui.dropdown(
        options={"Direct Channel": "direct", "Upper Funnel": "upper", "Lower Funnel": "lower"},
        value="Direct Channel",
        label="Select channel to analyze:"
    )

    response_spend_slider = mo.ui.slider(
        start=0,
        stop=300,
        value=100,
        step=10,
        label="Spend multiplier (% of historical):",
        full_width=True,
    )

    response_show_uncertainty = mo.ui.checkbox(label="Show uncertainty", value=True)

    mo.vstack([
        mo.md("## Response Curves Explorer"),
        mo.md("""
        Response curves show the **total effect** (sum over all time periods) when scaling the historical spend pattern.

        - **X-axis**: Total spend = historical total × multiplier
        - **Y-axis**: Total effect including adstock carry-over
        - **Slider**: Sets the spend multiplier for the selected channel (100% = historical baseline)

        This answers: "If we scale our historical spend pattern by X%, what is the total impact on sales?"
        """),
        mo.hstack([response_channel_selector, response_show_uncertainty]),
        response_spend_slider,
    ])
    return (
        response_channel_selector,
        response_show_uncertainty,
        response_spend_slider,
    )


@app.cell(hide_code=True)
def _(
    df,
    response_channel_selector,
    response_predictor_hdi,
    response_predictor_mean,
    response_show_uncertainty,
    response_spend_slider,
):
    # Interactive response curve visualization - showing total effects (sum over dates)
    # Uses actual historical spend time series scaled by multiplication factors

    mo.stop(
        response_predictor_mean is None,
        mo.md("*Click 'Fit Model' to start sampling*")
    )

    # Choose predictor based on uncertainty toggle
    _predictor = response_predictor_hdi if response_show_uncertainty.value else response_predictor_mean
    _selected_channel = response_channel_selector.value

    # Slider controls the multiplication factor for the selected channel (100 = 1.0x)
    _slider_factor = response_spend_slider.value / 100.0

    # Historical spend time series (normalized) for each channel
    _baseline_series = {
        'direct': df["spend_direct"].values / 100.0,
        'upper': df["spend_upper"].values / 100.0,
        'lower': df["spend_lower"].values / 100.0,
    }

    # Total historical spend for each channel (for x-axis scaling)
    _baseline_totals = {ch: series.sum() for ch, series in _baseline_series.items()}

    # Channel display properties
    _channels = {
        'direct': {'color': '#1f77b4', 'label': 'Direct Channel'},
        'upper': {'color': '#ff7f0e', 'label': 'Upper Funnel'},
        'lower': {'color': '#2ca02c', 'label': 'Lower Funnel'},
    }

    # Current multiplication factors for each channel
    _current_factors = {}
    for _ch in _channels:
        if _ch == _selected_channel:
            _current_factors[_ch] = _slider_factor
        else:
            _current_factors[_ch] = 1.0  # baseline for non-selected

    # === Compute response curves ===
    # X-axis: total spend = baseline_total × factor
    # Y-axis: total effect when historical series is scaled by factor
    _factor_range = np.linspace(0, 3, 25)  # 0x to 3x baseline

    # Store results for all channels
    _total_spends = {'direct': [], 'upper': [], 'lower': []}
    _responses = {'direct': [], 'upper': [], 'lower': []}
    _responses_low = {'direct': [], 'upper': [], 'lower': []}
    _responses_high = {'direct': [], 'upper': [], 'lower': []}

    # Compute response curves for each channel by varying its multiplication factor
    for _factor in _factor_range:
        # === Direct channel curve ===
        # Vary direct factor, keep upper/lower at their current factors
        _spend_direct = _baseline_series['direct'] * _factor
        _spend_upper = _baseline_series['upper'] * _current_factors['upper']
        _spend_lower = _baseline_series['lower'] * _current_factors['lower']

        _result = _predictor(_spend_direct, _spend_upper, _spend_lower)
        _total_spends['direct'].append(_baseline_totals['direct'] * _factor * 100)  # back to original scale
        _responses['direct'].append(np.asarray(_result['direct']).item())
        if response_show_uncertainty.value:
            _responses_low['direct'].append(np.asarray(_result['direct_low']).item())
            _responses_high['direct'].append(np.asarray(_result['direct_high']).item())

        # === Upper funnel curve ===
        # Vary upper factor - this also affects lower funnel through CPM!
        _spend_direct = _baseline_series['direct'] * _current_factors['direct']
        _spend_upper = _baseline_series['upper'] * _factor
        _spend_lower = _baseline_series['lower'] * _current_factors['lower']

        _result = _predictor(_spend_direct, _spend_upper, _spend_lower)
        _total_spends['upper'].append(_baseline_totals['upper'] * _factor * 100)
        _responses['upper'].append(np.asarray(_result['upper']).item())
        if response_show_uncertainty.value:
            _responses_low['upper'].append(np.asarray(_result['upper_low']).item())
            _responses_high['upper'].append(np.asarray(_result['upper_high']).item())

        # === Lower funnel curve ===
        # Vary lower factor, keep upper at current (which affects CPM)
        _spend_direct = _baseline_series['direct'] * _current_factors['direct']
        _spend_upper = _baseline_series['upper'] * _current_factors['upper']
        _spend_lower = _baseline_series['lower'] * _factor

        _result = _predictor(_spend_direct, _spend_upper, _spend_lower)
        _total_spends['lower'].append(_baseline_totals['lower'] * _factor * 100)
        _responses['lower'].append(np.asarray(_result['lower']).item())
        if response_show_uncertainty.value:
            _responses_low['lower'].append(np.asarray(_result['lower_low']).item())
            _responses_high['lower'].append(np.asarray(_result['lower_high']).item())

    # Convert to arrays
    _total_spends = {_ch: np.array(_total_spends[_ch]) for _ch in _channels}
    _responses = {_ch: np.array(_responses[_ch]) for _ch in _channels}

    # Get total effects at current factors (for markers)
    _current_spend_direct = _baseline_series['direct'] * _current_factors['direct']
    _current_spend_upper = _baseline_series['upper'] * _current_factors['upper']
    _current_spend_lower = _baseline_series['lower'] * _current_factors['lower']
    _current_result = _predictor(_current_spend_direct, _current_spend_upper, _current_spend_lower)

    _marker_effects = {
        'direct': np.asarray(_current_result['direct']).item(),
        'upper': np.asarray(_current_result['upper']).item(),
        'lower': np.asarray(_current_result['lower']).item(),
    }
    _marker_total_spends = {
        ch: _baseline_totals[ch] * _current_factors[ch] * 100 for ch in _channels
    }

    # Extract uncertainty bands (already computed in pytensor)
    _uncertainty_bands = {}
    if response_show_uncertainty.value:
        for _ch in _channels:
            _uncertainty_bands[_ch] = {
                'low': np.array(_responses_low[_ch]),
                'high': np.array(_responses_high[_ch]),
            }

    # === Create visualization ===
    _fig, _ax = plt.subplots(figsize=(12, 7))

    # Plot response curves for all channels
    # Each channel has its own x-axis scale (total spend for that channel)
    for _ch, _props in _channels.items():
        _is_selected = (_ch == _selected_channel)
        _linewidth = 3 if _is_selected else 1.5
        _alpha_line = 1.0 if _is_selected else 0.6
        _alpha_fill = 0.3 if _is_selected else 0.1

        _ax.plot(
            _total_spends[_ch], _responses[_ch],
            color=_props['color'],
            linewidth=_linewidth,
            alpha=_alpha_line,
            label=_props['label']
        )

        if response_show_uncertainty.value and _ch in _uncertainty_bands:
            _ax.fill_between(
                _total_spends[_ch],
                _uncertainty_bands[_ch]['low'],
                _uncertainty_bands[_ch]['high'],
                alpha=_alpha_fill,
                color=_props['color']
            )

    # Add markers for all channels at their current spend/effect
    _sel_color = _channels[_selected_channel]['color']
    for _ch, _props in _channels.items():
        _is_selected = (_ch == _selected_channel)
        _marker_size = 120 if _is_selected else 60
        _zorder = 6 if _is_selected else 5
        _marker_style = 'o' if _is_selected else 's'
        _ax.scatter(
            [_marker_total_spends[_ch]], [_marker_effects[_ch]],
            color=_props['color'], s=_marker_size, zorder=_zorder,
            edgecolors='white', linewidths=1.5, marker=_marker_style
        )

    # Annotate selected channel with slider value
    _ax.annotate(
        f'{_channels[_selected_channel]["label"]}\n{_slider_factor:.0%} of baseline\nTotal Spend: {_marker_total_spends[_selected_channel]:,.0f}\nTotal Effect: {_marker_effects[_selected_channel]:,.0f}',
        xy=(_marker_total_spends[_selected_channel], _marker_effects[_selected_channel]),
        xytext=(15, 15), textcoords='offset points',
        fontsize=10, color=_sel_color,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=_sel_color, alpha=0.8)
    )

    _ax.set_xlabel('Total Channel Spend (sum over all periods)')
    _ax.set_ylabel('Total Channel Effect on Sales (sum over all periods)')
    _ax.set_title('Response Curves: Scaling Historical Spend Pattern')
    _ax.legend(loc='upper left')
    _ax.grid(True, alpha=0.3)

    # Format axes with thousands separator
    _ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    mo.md(f"### Response Curves - All Channels")
    plt.gca()
    return


@app.cell
def _(funnel_model, idata):
    # Create compiled pytensor predictor for sales time series
    if idata:
        sales_timeseries_predictor, _ = create_sales_timeseries_predictor(
            funnel_model, idata, n_samples=200
        )
    else:
        sales_timeseries_predictor = None
    return (sales_timeseries_predictor,)


@app.cell
def _():
    # Interactive slider UI for Sales Time Series
    timeseries_channel_selector = mo.ui.dropdown(
        options={"Direct Channel": "direct", "Upper Funnel": "upper", "Lower Funnel": "lower"},
        value="Direct Channel",
        label="Select channel to scale:"
    )

    timeseries_spend_slider = mo.ui.slider(
        start=0,
        stop=300,
        value=100,
        step=10,
        label="Spend multiplier (% of historical):",
        full_width=True,
    )

    mo.vstack([
        mo.md("## Sales Time Series Explorer"),
        mo.md("""
        Explore how scaling a channel's spend affects predicted sales over time.

        - **X-axis**: Date (time periods)
        - **Y-axis**: Predicted sales with 90% credible interval
        - **Slider**: Multiply the selected channel's historical spend (100% = baseline)

        This shows the counterfactual: "What would sales look like if we had spent X% on this channel?"
        """),
        mo.hstack([timeseries_channel_selector]),
        timeseries_spend_slider,
    ])
    return timeseries_channel_selector, timeseries_spend_slider


@app.cell
def _(
    df,
    sales_timeseries_predictor,
    timeseries_channel_selector,
    timeseries_spend_slider,
):
    # Interactive sales time series visualization
    # Shows predicted sales when scaling one channel's spend

    mo.stop(
        sales_timeseries_predictor is None,
        mo.md("*Click 'Fit Model' to start sampling*")
    )

    _selected_channel = timeseries_channel_selector.value
    _slider_factor = timeseries_spend_slider.value / 100.0

    # Historical spend time series (normalized)
    _baseline_series = {
        'direct': df["spend_direct"].values / 100.0,
        'upper': df["spend_upper"].values / 100.0,
        'lower': df["spend_lower"].values / 100.0,
    }

    # Channel display properties
    _channels = {
        'direct': {'color': '#1f77b4', 'label': 'Direct Channel'},
        'upper': {'color': '#ff7f0e', 'label': 'Upper Funnel'},
        'lower': {'color': '#2ca02c', 'label': 'Lower Funnel'},
    }

    # Build spend scenarios: selected channel scaled, others at baseline
    _spend_direct = _baseline_series['direct'].copy()
    _spend_upper = _baseline_series['upper'].copy()
    _spend_lower = _baseline_series['lower'].copy()

    if _selected_channel == 'direct':
        _spend_direct = _spend_direct * _slider_factor
    elif _selected_channel == 'upper':
        _spend_upper = _spend_upper * _slider_factor
    else:  # lower
        _spend_lower = _spend_lower * _slider_factor

    # Compute predicted sales time series with uncertainty
    _result = sales_timeseries_predictor(_spend_direct, _spend_upper, _spend_lower)
    _pred_mean = _result['mean']
    _pred_low = _result['low']
    _pred_high = _result['high']

    # Also compute baseline (100% of all channels) for comparison
    _baseline_result = sales_timeseries_predictor(
        _baseline_series['direct'],
        _baseline_series['upper'],
        _baseline_series['lower']
    )
    _baseline_mean = _baseline_result['mean']

    # Get dates for x-axis
    _dates = df["date"].values

    # === Create visualization ===
    _fig, _ax = plt.subplots(figsize=(14, 7))

    _sel_color = _channels[_selected_channel]['color']

    # Plot actual observed sales
    _ax.scatter(_dates, df["sales"].values, color='black', s=15, alpha=0.5, 
                label='Observed Sales', zorder=3)

    # Plot baseline prediction (dashed)
    _ax.plot(_dates, _baseline_mean, color='gray', linestyle='--', linewidth=1.5,
             alpha=0.7, label='Baseline Prediction (100%)')

    # Plot counterfactual prediction with uncertainty
    _ax.plot(_dates, _pred_mean, color=_sel_color, linewidth=2.5,
             label=f'Predicted ({_slider_factor:.0%} {_channels[_selected_channel]["label"]})')
    _ax.fill_between(_dates, _pred_low, _pred_high, color=_sel_color, alpha=0.25,
                     label='90% Credible Interval')

    # Calculate and display summary statistics
    _total_baseline = _baseline_mean.sum()
    _total_counterfactual = _pred_mean.sum()
    _total_diff = _total_counterfactual - _total_baseline
    _pct_diff = (_total_diff / _total_baseline) * 100

    _ax.set_xlabel('Date')
    _ax.set_ylabel('Sales')
    _ax.set_title(f'Predicted Sales Time Series: {_channels[_selected_channel]["label"]} at {_slider_factor:.0%}')
    _ax.legend(loc='upper left')
    _ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Add summary annotation
    _summary_text = f"Total Sales Change: {_total_diff:+,.0f} ({_pct_diff:+.1f}%)"
    _ax.annotate(
        _summary_text,
        xy=(0.98, 0.02), xycoords='axes fraction',
        fontsize=11, color=_sel_color,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=_sel_color, alpha=0.9)
    )

    mo.vstack([
        mo.md(f"### Sales Impact: {_channels[_selected_channel]['label']} at {_slider_factor:.0%}"),
        plt.gcf(),
        mo.md(f"""
        **Summary:**
        - Baseline total sales: {_total_baseline:,.0f}
        - Counterfactual total sales: {_total_counterfactual:,.0f}
        - **Difference: {_total_diff:+,.0f} ({_pct_diff:+.1f}%)**
        """),
    ])
    return


@app.function
def create_saturation_predictor(model, inference_data, n_samples=200, include_hdi=False):
    """
    Create a compiled pytensor function for saturation curve predictions.

    Convenience wrapper around create_frozen_predictor for computing
    beta * saturation effects for each channel.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model (must have *_saturated Deterministic nodes)
    inference_data : InferenceData
        ArviZ InferenceData with posterior samples
    n_samples : int
        Number of posterior samples to use
    include_hdi : bool
        If True, include 5th/95th percentile outputs ({name}_low, {name}_high)

    Returns
    -------
    predict_fn : callable
        Function(spend_direct, spend_upper, spend_lower) -> dict of effects
        Inputs can be scalars or arrays (broadcast to same length).
    """
    _input_vars = ["spend_direct", "spend_upper", "spend_lower"]

    # Base expressions: beta * saturation for each channel
    # Note: lower_saturated depends on cpm_lower which depends on upper_saturated,
    # so the spend_upper → lower effect dependency is preserved in the graph
    _direct_expr = model["beta_direct"] * model["direct_saturated"]
    _upper_expr = model["beta_upper"] * model["upper_saturated"]
    _lower_expr = model["beta_lower"] * model["lower_saturated"]

    # Post-processing: mean over posterior samples (axis=0)
    _mean_fn = lambda x: x.mean(axis=0)

    _response_exprs = {
        'direct': (_direct_expr, _mean_fn),
        'upper': (_upper_expr, _mean_fn),
        'lower': (_lower_expr, _mean_fn),
    }

    # Add HDI bounds if requested
    # Note: pytensor doesn't have percentile, so we sort and index manually
    if include_hdi:
        # Compute indices for 5th and 95th percentiles (nearest rank method)
        _idx_low = int(round((n_samples - 1) * 0.05))
        _idx_high = int(round((n_samples - 1) * 0.95))
        _low_fn = lambda x, idx=_idx_low: pt.sort(x, axis=0)[idx]
        _high_fn = lambda x, idx=_idx_high: pt.sort(x, axis=0)[idx]
        _response_exprs['direct_low'] = (_direct_expr, _low_fn)
        _response_exprs['direct_high'] = (_direct_expr, _high_fn)
        _response_exprs['upper_low'] = (_upper_expr, _low_fn)
        _response_exprs['upper_high'] = (_upper_expr, _high_fn)
        _response_exprs['lower_low'] = (_lower_expr, _low_fn)
        _response_exprs['lower_high'] = (_lower_expr, _high_fn)

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
        Compute saturation effects for all channels at given spend levels.

        Parameters
        ----------
        spend_direct, spend_upper, spend_lower : float or array-like
            Spend levels (normalized). Can be scalars or arrays.
            Arrays are broadcast to the same length.

        Returns
        -------
        effects : dict
            Keys: 'direct', 'upper', 'lower' (and *_low/*_high if include_hdi=True)
        """
        return _general_predict_fn(
            spend_direct=spend_direct,
            spend_upper=spend_upper,
            spend_lower=spend_lower,
        )

    return predict_fn


@app.function
def create_response_predictor(model, inference_data, n_samples=200, include_hdi=False):
    """
    Create a compiled pytensor function for response curve predictions.

    Response curves show the total effect (sum over dates) at different 
    constant spend levels, including both saturation and adstock effects.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model (must have total_*_effect Deterministic nodes)
    inference_data : InferenceData
        ArviZ InferenceData with posterior samples
    n_samples : int
        Number of posterior samples to use
    include_hdi : bool
        If True, include 5th/95th percentile outputs ({name}_low, {name}_high)

    Returns
    -------
    predict_fn : callable
        Function(spend_direct, spend_upper, spend_lower) -> dict of total effects
    """
    _input_vars = ["spend_direct", "spend_upper", "spend_lower"]

    # Define response expressions with post-processing applied inside pytensor
    # Mean over posterior samples (axis=0)
    _mean_fn = lambda x: x.mean(axis=0)

    _response_exprs = {
        'direct': ("total_direct_effect", _mean_fn),
        'upper': ("total_upper_effect", _mean_fn),
        'lower': ("total_lower_effect", _mean_fn),
    }

    # Add HDI bounds if requested
    # Note: pytensor doesn't have percentile, so we sort and index manually
    if include_hdi:
        # Compute indices for 5th and 95th percentiles (nearest rank method)
        _idx_low = int(round((n_samples - 1) * 0.05))
        _idx_high = int(round((n_samples - 1) * 0.95))
        _low_fn = lambda x, idx=_idx_low: pt.sort(x, axis=0)[idx]
        _high_fn = lambda x, idx=_idx_high: pt.sort(x, axis=0)[idx]
        _response_exprs['direct_low'] = ("total_direct_effect", _low_fn)
        _response_exprs['direct_high'] = ("total_direct_effect", _high_fn)
        _response_exprs['upper_low'] = ("total_upper_effect", _low_fn)
        _response_exprs['upper_high'] = ("total_upper_effect", _high_fn)
        _response_exprs['lower_low'] = ("total_lower_effect", _low_fn)
        _response_exprs['lower_high'] = ("total_lower_effect", _high_fn)

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
        Compute total response effects for all channels at given spend time series.

        Parameters
        ----------
        spend_direct, spend_upper, spend_lower : array-like
            Spend time series (normalized) for each channel. Should be 1D arrays
            of the same length representing spend over time.

        Returns
        -------
        effects : dict
            Keys: 'direct', 'upper', 'lower' (and *_low/*_high if include_hdi=True)
            Values: scalar total effects (summed over all time periods)
        """
        return _general_predict_fn(
            spend_direct=np.asarray(spend_direct),
            spend_upper=np.asarray(spend_upper),
            spend_lower=np.asarray(spend_lower),
        )

    return predict_fn


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

    return predict_fn, _compiled_fn


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
