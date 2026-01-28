# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pymc>=5.17.0,<5.18",
#     "pymc-marketing>=0.16.0,<0.17",
#     "arviz",
#     "pytensor>=2.31.0,<2.32",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "wigglystuff",
#     "jax",
#     "numpyro",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    return datetime, mo, np, pd, plt, timedelta


@app.cell
def _(mo):
    mo.md("""
    # Interactive Counterfactual Analysis

    This notebook provides an interactive budget allocation tool for Marketing Mix Modeling.

    **Workflow:**
    1. Click **Fit Model** to train the MMM on synthetic data
    2. **Plot 1**: Select a channel and drag pucks to adjust weekly spend
    3. Click **Confirm Budget** to lock in your budget allocation
    4. **Plot 2**: View the counterfactual sales prediction (updates on Confirm)
    """)
    return


@app.cell
def _(np, pd, timedelta):
    # ==============================================================================
    # TRUE PARAMETERS - Used to generate synthetic data
    # ==============================================================================
    TRUE_PARAMS = {
        # Channel parameters: (adstock_alpha, saturation_lam, saturation_beta)
        "Direct": {
            "adstock_alpha": 0.3,  # Fast decay - direct response
            "saturation_lam": 0.8,
            "saturation_beta": 0.4,
            "base_spend": 5000,
            "spend_std": 1500,
        },
        "Upper_Funnel": {
            "adstock_alpha": 0.7,  # Slow decay - brand building
            "saturation_lam": 0.5,
            "saturation_beta": 0.25,
            "base_spend": 8000,
            "spend_std": 2000,
        },
        "Lower_Funnel": {
            "adstock_alpha": 0.4,  # Medium decay
            "saturation_lam": 0.6,
            "saturation_beta": 0.35,
            "base_spend": 6000,
            "spend_std": 1800,
        },
        # Global parameters
        "intercept": 10000,
        "noise_std": 500,
        "trend_coef": 50,  # Weekly trend
    }

    def geometric_adstock(x, alpha, l_max=8):
        """Apply geometric adstock transformation."""
        weights = np.array([alpha**i for i in range(l_max)])
        weights = weights / weights.sum()

        result = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            for j, w in enumerate(weights):
                if i - j >= 0:
                    result[i] += w * x[i - j]
        return result

    def logistic_saturation(x, lam, beta):
        """Apply logistic saturation transformation."""
        return beta * (1 - np.exp(-lam * x))

    def generate_marketing_funnel_data(n_weeks=104, seed=42):
        """
        Generate synthetic marketing data with known parameters.

        Returns:
            pd.DataFrame: Marketing data with columns for date, channels, and sales
        """
        np.random.seed(seed)

        # Generate dates
        start_date = pd.Timestamp("2022-01-01")
        dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]

        # Generate spend for each channel with seasonality
        t = np.arange(n_weeks)
        seasonality = 1 + 0.2 * np.sin(2 * np.pi * t / 52)  # Annual seasonality

        channels = ["Direct", "Upper_Funnel", "Lower_Funnel"]
        spend_data = {}
        contributions = {}

        for channel in channels:
            params = TRUE_PARAMS[channel]

            # Generate spend with seasonality and noise
            base_spend = params["base_spend"] * seasonality
            noise = np.random.normal(0, params["spend_std"], n_weeks)
            spend = np.maximum(base_spend + noise, 0)  # Ensure non-negative
            spend_data[channel] = spend

            # Apply adstock
            adstocked = geometric_adstock(spend / 1000, params["adstock_alpha"])  # Normalize spend

            # Apply saturation
            saturated = logistic_saturation(adstocked, params["saturation_lam"], params["saturation_beta"])

            # Scale contribution
            contributions[channel] = saturated * 10000

        # Calculate total sales
        intercept = TRUE_PARAMS["intercept"]
        trend = TRUE_PARAMS["trend_coef"] * t
        noise = np.random.normal(0, TRUE_PARAMS["noise_std"], n_weeks)

        sales = (
            intercept
            + trend
            + contributions["Direct"]
            + contributions["Upper_Funnel"]
            + contributions["Lower_Funnel"]
            + noise
        )

        # Create DataFrame
        df = pd.DataFrame({
            "date_week": dates,
            "Direct": spend_data["Direct"],
            "Upper_Funnel": spend_data["Upper_Funnel"],
            "Lower_Funnel": spend_data["Lower_Funnel"],
            "sales": sales,
        })

        return df, contributions

    # Generate the data
    marketing_data, true_contributions = generate_marketing_funnel_data(n_weeks=104)

    print(f"Generated {len(marketing_data)} weeks of synthetic marketing data")
    print(f"Channels: {['Direct', 'Upper_Funnel', 'Lower_Funnel']}")
    print(f"Date range: {marketing_data['date_week'].min()} to {marketing_data['date_week'].max()}")
    return (
        TRUE_PARAMS,
        generate_marketing_funnel_data,
        geometric_adstock,
        logistic_saturation,
        marketing_data,
        true_contributions,
    )


@app.cell
def _(marketing_data, mo, plt, true_contributions):
    # Visualize the generated data
    fig_data, axes_data = plt.subplots(2, 1, figsize=(12, 8))

    # Plot spend over time
    ax1 = axes_data[0]
    for channel in ["Direct", "Upper_Funnel", "Lower_Funnel"]:
        ax1.plot(marketing_data["date_week"], marketing_data[channel], label=channel, alpha=0.8)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Spend ($)")
    ax1.set_title("Marketing Spend by Channel Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot sales and contributions
    ax2 = axes_data[1]
    ax2.plot(marketing_data["date_week"], marketing_data["sales"], label="Total Sales", color="black", linewidth=2)
    for channel, contrib in true_contributions.items():
        ax2.fill_between(marketing_data["date_week"], 0, contrib, alpha=0.3, label=f"{channel} Contribution")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales / Contribution")
    ax2.set_title("Sales and Channel Contributions (True)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.md("## Generated Synthetic Data")
    return ax1, ax2, axes_data, fig_data


@app.cell
def _(axes_data):
    # Display the data visualization
    axes_data[0].figure
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Model Fitting

    Click the button below to fit the MMM model using PyMC-Marketing with the numpyro sampler.

    **Note:** This uses reduced sampling parameters (300 draws, 300 tune, 4 chains) for faster execution.
    """)
    return


@app.cell
def _(mo):
    # Fit model button
    fit_button = mo.ui.run_button(label="🔧 Fit Model", kind="success")
    fit_button
    return (fit_button,)


@app.cell
def _(fit_button, marketing_data, mo, pd):
    # Import PyMC-Marketing components
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    # Initialize model state
    mmm_model = None
    model_fitted = False

    if fit_button.value:
        mo.output.append(mo.md("⏳ **Fitting model... this may take a few minutes.**"))

        # Prepare data
        date_column = "date_week"
        channel_columns = ["Direct", "Upper_Funnel", "Lower_Funnel"]
        target_column = "sales"

        # Ensure date column is datetime
        df_model = marketing_data.copy()
        df_model[date_column] = pd.to_datetime(df_model[date_column])

        # Create X and y
        X = df_model[[date_column] + channel_columns]
        y = df_model[target_column].values

        # Initialize the MMM model
        mmm_model = MMM(
            date_column=date_column,
            channel_columns=channel_columns,
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            validate_data=True,
        )

        # Fit the model with numpyro sampler (faster)
        mmm_model.fit(
            X=X,
            y=y,
            target_accept=0.9,
            draws=300,
            tune=300,
            chains=4,
            nuts_sampler="numpyro",
            random_seed=42,
        )

        # Sample posterior predictive
        mmm_model.sample_posterior_predictive(
            X_pred=mmm_model.X,
            extend_idata=True,
            combined=True,
        )

        model_fitted = True
        mo.output.append(mo.md("✅ **Model fitted successfully!**"))

    mmm_model, model_fitted
    return (
        GeometricAdstock,
        LogisticSaturation,
        MMM,
        mmm_model,
        model_fitted,
    )


@app.cell
def _(mmm_model, mo, model_fitted):
    # Show model summary if fitted
    if model_fitted and mmm_model is not None:
        import arviz as az

        _summary = az.summary(
            mmm_model.idata,
            var_names=["intercept", "adstock_alpha", "saturation_lam", "saturation_beta"],
        )

        _result = mo.md(f"""
        ### Model Summary

        The model has been fitted with the following posterior estimates:

        {_summary.to_markdown()}
        """)
    else:
        _result = mo.md("*Model not yet fitted. Click 'Fit Model' above to train the MMM.*")

    _result
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Plot 1: Interactive Budget Layout Editor

    **Instructions:**
    - Select a channel from the dropdown to edit its spend
    - Drag the pucks **vertically** to adjust weekly spend values (x-position is fixed)
    - Non-selected channels are shown in gray (non-interactive)
    - Click **Confirm Budget** to save your budget layout
    - Click **Reset Channels** to restore all channels to historical values
    """)
    return


@app.cell
def _(mo):
    # Channel selector dropdown
    channels_list = ["Direct", "Upper_Funnel", "Lower_Funnel"]
    channel_selector = mo.ui.dropdown(
        options=channels_list,
        value=channels_list[0],
        label="Select Channel to Edit"
    )
    channel_selector
    return channel_selector, channels_list


@app.cell
def _(marketing_data, np):
    # Configuration for the interactive planner
    n_weeks_plan = 52  # First year for planning
    subsample_step = 4  # Show every 4th week (~26 pucks)
    subsample_indices = list(range(0, n_weeks_plan, subsample_step))
    n_pucks = len(subsample_indices)

    # Store ORIGINAL x positions - these are fixed week indices
    original_x = np.array(subsample_indices, dtype=float)

    # Get historical spend data for all channels
    channels = ["Direct", "Upper_Funnel", "Lower_Funnel"]
    historical_spend = {}
    for _channel in channels:
        historical_spend[_channel] = marketing_data[_channel].values[:n_weeks_plan].copy()

    # Get TRUE historical sales
    historical_sales = marketing_data["sales"].values[:n_weeks_plan].copy()

    # Get dates for the planning period
    planning_dates = marketing_data["date_week"].values[:n_weeks_plan]

    # Calculate spend range for scaling
    all_spend_values = []
    for _ch in channels:
        all_spend_values.extend(historical_spend[_ch])
    max_spend = max(all_spend_values) * 1.5
    min_spend = 0

    print(f"Planning horizon: {n_weeks_plan} weeks")
    print(f"Interactive pucks: {n_pucks} (every {subsample_step} weeks)")
    print(f"Spend range: {min_spend:.0f} to {max_spend:.0f}")
    return (
        all_spend_values,
        channels,
        historical_sales,
        historical_spend,
        max_spend,
        min_spend,
        n_pucks,
        n_weeks_plan,
        original_x,
        planning_dates,
        subsample_indices,
        subsample_step,
    )


@app.cell
def _(mo):
    # State management buttons
    confirm_button = mo.ui.run_button(label="✓ Confirm Budget", kind="success")
    reset_button = mo.ui.run_button(label="↺ Reset Channels", kind="danger")

    mo.hstack([confirm_button, reset_button], justify="start", gap=1)
    return confirm_button, reset_button


@app.cell
def _(
    channel_selector,
    channels,
    historical_spend,
    max_spend,
    min_spend,
    mo,
    n_weeks_plan,
    np,
    original_x,
    plt,
    subsample_indices,
):
    from wigglystuff import ChartPuck

    # Get the currently selected channel
    selected_channel = channel_selector.value

    # Initialize spend at subsample indices for all channels
    def get_initial_channel_spend():
        """Create initial spend dictionary from historical values."""
        spend_dict = {}
        for _ch in channels:
            spend_dict[_ch] = np.array([historical_spend[_ch][i] for i in subsample_indices])
        return spend_dict

    _initial_channel_spend = get_initial_channel_spend()
    _selected_spend = _initial_channel_spend[selected_channel]

    # Color palette for channels
    _channel_colors = {
        "Direct": "#e63946",       # Red
        "Upper_Funnel": "#2a9d8f", # Teal
        "Lower_Funnel": "#e9c46a", # Yellow
    }

    def draw_multi_channel_budget(ax, widget):
        """
        Draw multi-channel budget editor.
        - Selected channel: interactive, colored, uses puck values
        - Other channels: gray, transparent, uses historical values
        - X positions are FIXED (vertical-only movement)
        """
        ax.clear()

        # Get current puck Y positions (spend values) - IGNORE widget.x
        snapped_x = original_x.copy()  # Always use original x positions
        current_y = np.array(list(widget.y))
        current_y = np.clip(current_y, min_spend, max_spend)

        # Draw all channels
        for ch in channels:
            ch_spend = _initial_channel_spend[ch]

            if ch == selected_channel:
                # SELECTED CHANNEL: Use puck Y values, full color
                color = _channel_colors[ch]
                alpha = 1.0
                linewidth = 2.5
                marker_size = 120
                spend_to_plot = current_y
                zorder = 10
                label = f'{ch} (editing)'
            else:
                # NON-SELECTED: Use historical, gray, 30% transparent
                color = 'gray'
                alpha = 0.3
                linewidth = 1.5
                marker_size = 40
                spend_to_plot = ch_spend
                zorder = 1
                label = ch

            # Draw line connecting points
            if len(snapped_x) > 1:
                x_smooth = np.linspace(snapped_x.min(), snapped_x.max(), 100)
                y_smooth = np.interp(x_smooth, snapped_x, spend_to_plot)
                ax.plot(x_smooth, y_smooth, color=color, alpha=alpha * 0.7,
                       linewidth=linewidth, label=label)

            # Draw markers
            ax.scatter(snapped_x, spend_to_plot, s=marker_size, c=color,
                      alpha=alpha, edgecolors='white' if ch == selected_channel else 'none',
                      linewidth=2, zorder=zorder)

        # Add week labels on x-axis
        ax.set_xticks(snapped_x[::2])  # Every other puck to avoid crowding
        ax.set_xticklabels([f'W{int(x)}' for x in snapped_x[::2]], fontsize=8)

        # Formatting
        ax.set_xlim(-1, n_weeks_plan)
        ax.set_ylim(min_spend - max_spend * 0.05, max_spend * 1.1)
        ax.set_xlabel('Week')
        ax.set_ylabel('Spend ($)')
        ax.set_title(f'Budget Editor - Editing: {selected_channel}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Add summary for selected channel
        total_spend = np.sum(current_y) * (n_weeks_plan / len(current_y))  # Extrapolate to full period
        avg_spend = np.mean(current_y)
        ax.text(0.02, 0.98, f'{selected_channel}: Total≈${total_spend:,.0f} | Avg=${avg_spend:,.0f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Create the ChartPuck widget
    budget_puck = ChartPuck.from_callback(
        draw_fn=draw_multi_channel_budget,
        x_bounds=(-1, n_weeks_plan),
        y_bounds=(min_spend, max_spend),
        drag_y_bounds=(min_spend, max_spend),
        figsize=(14, 6),
        x=list(original_x),
        y=list(_selected_spend),
        puck_color=_channel_colors[selected_channel],
        puck_radius=12,
    )

    # Close the figure to prevent duplicate display
    plt.close()

    # Wrap in marimo widget
    budget_widget = mo.ui.anywidget(budget_puck)
    return (
        ChartPuck,
        budget_puck,
        budget_widget,
        draw_multi_channel_budget,
        get_initial_channel_spend,
        selected_channel,
    )


@app.cell
def _(budget_widget):
    # Display the interactive widget
    budget_widget
    return


@app.cell
def _(budget_widget, mo, np, original_x, selected_channel):
    # Extract and display current budget allocation for selected channel
    _current_spend_values = np.array(list(budget_widget.value.y))

    # Create a summary table
    _budget_summary = []
    for _i, (week, spend) in enumerate(zip(original_x, _current_spend_values)):
        _budget_summary.append({
            'Week': f'Week {int(week)}',
            'Spend': f'${spend:,.0f}',
        })

    _total_spend = np.sum(_current_spend_values)

    mo.md(f"""
    ### Current {selected_channel} Budget (Editing)

    **Total Spend (sampled weeks)**: ${_total_spend:,.0f}

    | Week | Spend |
    |------|-------|
    """ + '\n'.join([f"| {row['Week']} | {row['Spend']} |" for row in _budget_summary[:8]]) +
    f"\n| ... | ... |\n| Total | ${_total_spend:,.0f} |")
    return


@app.cell
def _(
    budget_widget,
    confirm_button,
    get_initial_channel_spend,
    mo,
    np,
    reset_button,
    selected_channel,
):
    # Manage confirmed state using mo.state
    confirmed_budget_state, set_confirmed_budget = mo.state(get_initial_channel_spend())

    # Track current puck values
    _current_puck_values = np.array(list(budget_widget.value.y))

    if confirm_button.value:
        # User clicked Confirm - save current puck values for selected channel
        _new_budget = confirmed_budget_state.copy()
        _new_budget[selected_channel] = _current_puck_values.copy()
        set_confirmed_budget(_new_budget)
        mo.output.append(mo.md("✅ **Budget confirmed!** Counterfactual plot will update."))

    if reset_button.value:
        # User clicked Reset - restore all channels to historical
        set_confirmed_budget(get_initial_channel_spend())
        mo.output.append(mo.md("↺ **All channels reset to historical values.**"))

    # Return the confirmed budget for use in Plot 2
    confirmed_budget = confirmed_budget_state
    return confirmed_budget, confirmed_budget_state, set_confirmed_budget


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Plot 2: Counterfactual Sales Prediction

    This plot shows:
    - **Gray line**: TRUE historical sales (baseline)
    - **Green/Red line**: Counterfactual prediction based on your **confirmed** budget
    - **Shaded area**: Difference between counterfactual and historical

    ⚠️ **This plot only updates when you click "Confirm Budget" above.**
    """)
    return


@app.cell
def _(
    TRUE_PARAMS,
    channels,
    confirmed_budget,
    geometric_adstock,
    historical_sales,
    logistic_saturation,
    mmm_model,
    mo,
    model_fitted,
    n_weeks_plan,
    np,
    original_x,
    planning_dates,
    plt,
):
    # ==============================================================================
    # Create Frozen Predictor for Counterfactual Analysis
    # ==============================================================================

    def create_frozen_predictor(mmm, true_params):
        """
        Create a predictor function using fitted model or true parameters.

        If mmm is fitted, uses posterior mean estimates.
        Otherwise, falls back to true generating parameters.
        """
        if mmm is not None and hasattr(mmm, 'idata'):
            # Use fitted model posterior means
            import arviz as az
            summary = az.summary(mmm.idata, var_names=["intercept", "adstock_alpha", "saturation_lam", "saturation_beta"])

            # Extract posterior means
            intercept = summary.loc["intercept", "mean"]
            alphas = {}
            lams = {}
            betas = {}

            for i, ch in enumerate(["Direct", "Upper_Funnel", "Lower_Funnel"]):
                alphas[ch] = summary.loc[f"adstock_alpha[{ch}]", "mean"]
                lams[ch] = summary.loc[f"saturation_lam[{ch}]", "mean"]
                betas[ch] = summary.loc[f"saturation_beta[{ch}]", "mean"]

            def predict(spend_dict, n_weeks):
                """Predict sales given spend dictionary."""
                t = np.arange(n_weeks)

                total_contribution = np.zeros(n_weeks)
                for ch in ["Direct", "Upper_Funnel", "Lower_Funnel"]:
                    spend = spend_dict[ch]
                    adstocked = geometric_adstock(spend / 1000, alphas[ch])
                    saturated = logistic_saturation(adstocked, lams[ch], betas[ch])
                    total_contribution += saturated * 10000

                # Use fitted intercept but no trend (keeping it simple)
                return intercept + total_contribution + true_params["trend_coef"] * t

            return predict
        else:
            # Fallback to true parameters
            def predict(spend_dict, n_weeks):
                """Predict sales given spend dictionary using true params."""
                t = np.arange(n_weeks)

                total_contribution = np.zeros(n_weeks)
                for ch in ["Direct", "Upper_Funnel", "Lower_Funnel"]:
                    params = true_params[ch]
                    spend = spend_dict[ch]
                    adstocked = geometric_adstock(spend / 1000, params["adstock_alpha"])
                    saturated = logistic_saturation(adstocked, params["saturation_lam"], params["saturation_beta"])
                    total_contribution += saturated * 10000

                return true_params["intercept"] + true_params["trend_coef"] * t + total_contribution

            return predict

    # Create predictor based on model state
    predictor = create_frozen_predictor(mmm_model if model_fitted else None, TRUE_PARAMS)

    # ==============================================================================
    # Compute Counterfactual Sales
    # ==============================================================================

    # Interpolate confirmed budget to all weeks
    counterfactual_spend = {}
    for _channel in channels:
        _confirmed_spend = confirmed_budget[_channel]
        counterfactual_spend[_channel] = np.interp(
            np.arange(n_weeks_plan),
            original_x,
            _confirmed_spend
        )

    # Predict counterfactual sales
    y_counterfactual = predictor(counterfactual_spend, n_weeks_plan)

    # Calculate metrics
    total_historical = np.sum(historical_sales)
    total_counterfactual = np.sum(y_counterfactual)
    sales_diff = total_counterfactual - total_historical
    lift_pct = (total_counterfactual / total_historical - 1) * 100 if total_historical > 0 else 0

    # ==============================================================================
    # Create Counterfactual Plot (Non-Interactive)
    # ==============================================================================

    fig_cf, ax_cf = plt.subplots(figsize=(14, 6))

    week_indices = np.arange(n_weeks_plan)

    # Determine color based on lift
    cf_color = 'green' if sales_diff >= 0 else 'red'

    # Plot shaded area (difference)
    ax_cf.fill_between(
        week_indices,
        historical_sales,
        y_counterfactual,
        alpha=0.3,
        color=cf_color,
        label='Lift/Drop'
    )

    # Plot TRUE historical sales in GRAY
    ax_cf.plot(
        week_indices,
        historical_sales,
        color='gray',
        linewidth=2.5,
        label='Historical (Actual)',
        marker='o',
        markersize=4,
        markevery=4
    )

    # Plot counterfactual in color
    ax_cf.plot(
        week_indices,
        y_counterfactual,
        color=cf_color,
        linewidth=2.5,
        linestyle='--',
        label='Counterfactual (Predicted)',
        marker='D',
        markersize=5,
        markevery=4
    )

    # Formatting
    ax_cf.set_xlabel('Week', fontsize=11)
    ax_cf.set_ylabel('Sales', fontsize=11)
    ax_cf.set_title('Counterfactual Sales Prediction (Updates on Confirm)', fontsize=14, fontweight='bold')
    ax_cf.legend(loc='upper left', fontsize=10)
    ax_cf.grid(True, alpha=0.3)

    # Add summary annotation
    direction_emoji = "📈" if sales_diff >= 0 else "📉"
    summary_text = f'{direction_emoji} Lift: {sales_diff:+,.0f} ({lift_pct:+.1f}%)'
    ax_cf.annotate(
        summary_text,
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        fontsize=12,
        fontweight='bold',
        ha='right',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=cf_color, alpha=0.9)
    )

    # Set x-axis labels
    ax_cf.set_xticks(week_indices[::4])
    ax_cf.set_xticklabels([f'W{int(w)}' for w in week_indices[::4]], fontsize=8)

    plt.tight_layout()

    # Display the plot
    mo.md("### Counterfactual Results")
    return (
        ax_cf,
        cf_color,
        counterfactual_spend,
        create_frozen_predictor,
        direction_emoji,
        fig_cf,
        lift_pct,
        predictor,
        sales_diff,
        summary_text,
        total_counterfactual,
        total_historical,
        week_indices,
        y_counterfactual,
    )


@app.cell
def _(fig_cf):
    # Display the counterfactual plot
    fig_cf
    return


@app.cell
def _(
    lift_pct,
    mo,
    np,
    sales_diff,
    total_counterfactual,
    total_historical,
    y_counterfactual,
):
    # Summary metrics for counterfactual analysis
    _direction = "📈 Increase" if sales_diff >= 0 else "📉 Decrease"

    mo.md(f"""
    ### Counterfactual Summary (Based on Confirmed Budget)

    | Metric | Value |
    |--------|-------|
    | Historical Sales (Actual) | ${total_historical:,.0f} |
    | Counterfactual Sales (Predicted) | ${total_counterfactual:,.0f} |
    | **Sales {_direction}** | **${sales_diff:+,.0f}** ({lift_pct:+.1f}%) |
    | Avg Weekly Counterfactual | ${np.mean(y_counterfactual):,.0f} |
    | Max Weekly Counterfactual | ${np.max(y_counterfactual):,.0f} |
    | Min Weekly Counterfactual | ${np.min(y_counterfactual):,.0f} |
    """)
    return


@app.cell
def _(channels, confirmed_budget, historical_spend, mo, np, original_x):
    # Show budget comparison: Historical vs Confirmed for all channels

    _comparison_rows = []
    for _ch in channels:
        _hist_total = np.sum([historical_spend[_ch][int(i)] for i in original_x])
        _conf_total = np.sum(confirmed_budget[_ch])
        _change = _conf_total - _hist_total
        _change_pct = (_conf_total / _hist_total - 1) * 100 if _hist_total > 0 else 0
        _comparison_rows.append(f"| {_ch} | ${_hist_total:,.0f} | ${_conf_total:,.0f} | ${_change:+,.0f} ({_change_pct:+.1f}%) |")

    mo.md(f"""
    ### Budget Comparison (All Channels - Sampled Weeks)

    | Channel | Historical | Confirmed | Change |
    |---------|------------|-----------|--------|
    """ + '\n'.join(_comparison_rows))
    return


if __name__ == "__main__":
    app.run()
