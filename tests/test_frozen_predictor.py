"""
Tests for frozen predictor implementations.

This module tests the create_frozen_predictor function from frozen_predictor.py
which handles both single-scenario and multi-scenario inputs, observed RV handling, and
proper shape management.
"""

import numpy as np
import pytest
import pymc as pm
import arviz as az
import pytensor.tensor as pt

from frozen_predictor import create_frozen_predictor


@pytest.fixture(scope="module")
def simple_model_and_samples():
    """Create a simple linear model and sample from it."""
    np.random.seed(42)
    n_obs = 20
    x_data = np.random.randn(n_obs)
    y_data = 2.0 * x_data + 1.0 + np.random.randn(n_obs) * 0.5

    with pm.Model(coords={"obs": range(n_obs)}) as model:
        x = pm.Data("x", x_data, dims="obs")
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = pm.Deterministic("mu", alpha + beta * x, dims="obs")
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data, dims="obs")

    with model:
        idata = pm.sample(
            draws=100,
            tune=100,
            chains=2,
            random_seed=42,
            progressbar=False,
        )

    return model, idata


@pytest.fixture(scope="module")
def mediation_model_and_samples():
    """Create a mediation model with observed intermediate RV."""
    np.random.seed(42)
    n_obs = 30

    # Generate data
    spend_data = np.random.uniform(0.2, 0.8, n_obs)
    visits_data = 100 + 50 * spend_data + np.random.normal(0, 5, n_obs)
    purchases_data = 10 + 0.1 * visits_data + np.random.normal(0, 3, n_obs)

    with pm.Model(coords={"time": range(n_obs)}) as model:
        spend = pm.Data("spend", spend_data, dims="time")

        # Stage 1: Spend -> Visits
        alpha_v = pm.Normal("alpha_visits", mu=100, sigma=30)
        beta_s = pm.Normal("beta_spend", mu=50, sigma=20)
        mu_visits = pm.Deterministic("mu_visits", alpha_v + beta_s * spend, dims="time")
        sigma_v = pm.HalfNormal("sigma_visits", sigma=20)
        visits = pm.Normal("visits", mu=mu_visits, sigma=sigma_v, observed=visits_data, dims="time")

        # Stage 2: Visits -> Purchases
        alpha_p = pm.Normal("alpha_purchases", mu=10, sigma=10)
        beta_v = pm.Normal("beta_visits", mu=0.1, sigma=0.05)
        mu_purchases = pm.Deterministic("mu_purchases", alpha_p + beta_v * visits, dims="time")
        sigma_p = pm.HalfNormal("sigma_purchases", sigma=10)
        purchases = pm.Normal("purchases", mu=mu_purchases, sigma=sigma_p, observed=purchases_data, dims="time")

    with model:
        idata = pm.sample(
            draws=100,
            tune=100,
            chains=2,
            random_seed=42,
            progressbar=False,
        )

    return model, idata


class TestBasicFunctionality:
    """Tests for basic posterior predictor functionality."""

    def test_1d_input_produces_correct_shape(self, simple_model_and_samples):
        """1D input produces (n_samples, n_points) output."""
        model, idata = simple_model_and_samples

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            num_samples=50,
            rng=42,
        )

        test_x = np.linspace(-2, 2, 20)
        result = predictor(x=test_x)

        assert result["mu"].shape == (50, 20)

    def test_2d_input_produces_correct_shape(self, simple_model_and_samples):
        """2D input produces (n_samples, n_scenarios, n_points) output."""
        model, idata = simple_model_and_samples

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            num_samples=50,
            rng=42,
        )

        test_x = np.random.randn(5, 10)  # 5 scenarios, 10 points
        result = predictor(x=test_x)

        assert result["mu"].shape == (50, 5, 10)

    def test_post_processing_applied(self, simple_model_and_samples):
        """Post-processing functions are applied correctly."""
        model, idata = simple_model_and_samples

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={
                "mu_mean": ("mu", lambda x: x.mean(axis=0)),
            },
            input_vars=["x"],
            num_samples=50,
            rng=42,
        )

        test_x = np.linspace(-2, 2, 20)
        result = predictor(x=test_x)

        # Post-processed output should have reduced dimension
        assert result["mu_mean"].shape == (20,)

    def test_dynamic_input_size(self, simple_model_and_samples):
        """Predictor works with different input sizes."""
        model, idata = simple_model_and_samples

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            num_samples=50,
            rng=42,
        )

        # Test with different sizes
        for size in [10, 20, 50, 100]:
            test_x = np.linspace(-2, 2, size)
            result = predictor(x=test_x)
            assert result["mu"].shape == (50, size)


class TestSeedParameter:
    """Tests for seed parameter reproducibility."""

    def test_same_seed_produces_identical_results(self, simple_model_and_samples):
        """Same seed produces identical results."""
        model, idata = simple_model_and_samples

        pred1 = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            num_samples=50,
            rng=42,
        )

        pred2 = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            num_samples=50,
            rng=42,
        )

        test_x = np.linspace(-2, 2, 20)
        result1 = pred1(x=test_x)
        result2 = pred2(x=test_x)

        np.testing.assert_allclose(result1["mu"], result2["mu"])

    def test_different_seeds_produce_different_results(self, simple_model_and_samples):
        """Different seeds produce different sample selections."""
        model, idata = simple_model_and_samples

        pred1 = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            num_samples=50,
            rng=42,
        )

        pred2 = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            num_samples=50,
            rng=123,
        )

        test_x = np.linspace(-2, 2, 20)
        result1 = pred1(x=test_x)
        result2 = pred2(x=test_x)

        # With different seeds, expect different sample selection
        assert not np.allclose(result1["mu"], result2["mu"])


class TestObservedRVHandling:
    """Tests for observed RV handling strategies."""

    def test_input_dependent_observed_rv_handled(self, mediation_model_and_samples):
        """Input-dependent observed RV is replaced with ICDF using frozen quantiles."""
        model, idata = mediation_model_and_samples

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu_purchases": "mu_purchases"},
            input_vars=["spend"],
            num_samples=50,
            rng=42,
        )

        # Should work with different input sizes
        for size in [30, 50, 10]:
            test_spend = np.linspace(0.1, 1.0, size)
            result = predictor(spend=test_spend)
            assert result["mu_purchases"].shape == (50, size)

    def test_stage1_predictor_no_observed_rv_issue(self, mediation_model_and_samples):
        """Stage 1 predictor (spend -> visits) has no observed RV dependency."""
        model, idata = mediation_model_and_samples

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu_visits": "mu_visits"},
            input_vars=["spend"],
            num_samples=50,
            rng=42,
        )

        # Should work with any size since mu_visits doesn't depend on observed visits
        for size in [30, 50, 10, 100]:
            test_spend = np.linspace(0.1, 1.0, size)
            result = predictor(spend=test_spend)
            assert result["mu_visits"].shape == (50, size)


class TestFixedShapeWarning:
    """Tests for warning when observed RVs have fixed shapes."""

    @pytest.fixture
    def model_without_dims(self):
        """Create a mediation model WITHOUT dims (fixed shapes)."""
        np.random.seed(42)
        n_obs = 20
        spend_data = np.random.uniform(0.2, 0.8, n_obs)
        visits_data = 100 + 50 * spend_data + np.random.normal(0, 5, n_obs)
        purchases_data = 10 + 0.1 * visits_data + np.random.normal(0, 3, n_obs)

        # Model WITHOUT dims - observed RV will have fixed shape
        with pm.Model() as model:
            spend = pm.Data("spend", spend_data)
            alpha_v = pm.Normal("alpha_visits", mu=100, sigma=30)
            beta_s = pm.Normal("beta_spend", mu=50, sigma=20)
            mu_visits = pm.Deterministic("mu_visits", alpha_v + beta_s * spend)
            sigma_v = pm.HalfNormal("sigma_visits", sigma=20)
            visits = pm.Normal("visits", mu=mu_visits, sigma=sigma_v, observed=visits_data)
            
            # Stage 2: depends on observed visits
            alpha_p = pm.Normal("alpha_purchases", mu=10, sigma=10)
            beta_v = pm.Normal("beta_visits", mu=0.1, sigma=0.05)
            mu_purchases = pm.Deterministic("mu_purchases", alpha_p + beta_v * visits)

        with model:
            idata = pm.sample(draws=50, tune=50, chains=2, random_seed=42, progressbar=False)

        return model, idata

    def test_warning_for_fixed_shape_observed_rv(self, model_without_dims, caplog):
        """Warning is issued when observed RV has fixed shape."""
        import logging
        model, idata = model_without_dims

        with caplog.at_level(logging.WARNING, logger="frozen_predictor"):
            # Request mu_purchases which depends on the observed visits RV
            create_frozen_predictor(
                model=model,
                inference_data=idata,
                response_exprs={"mu_purchases": "mu_purchases"},
                input_vars=["spend"],
                num_samples=50,
                rng=42,
            )

        # Check that warning about fixed shape was logged
        assert any("fixed shape" in record.message for record in caplog.records)
        assert any("dims" in record.message for record in caplog.records)

    def test_no_warning_for_dynamic_shape_observed_rv(self, mediation_model_and_samples, caplog):
        """No warning when observed RV has dynamic shape (defined with dims)."""
        import logging
        model, idata = mediation_model_and_samples

        with caplog.at_level(logging.WARNING, logger="frozen_predictor"):
            create_frozen_predictor(
                model=model,
                inference_data=idata,
                response_exprs={"mu_purchases": "mu_purchases"},
                input_vars=["spend"],
                num_samples=50,
                rng=42,
            )

        # Check that no warning about fixed shape was logged
        assert not any("fixed shape" in record.message for record in caplog.records)


class TestGroupParameter:
    """Tests for group parameter validation."""

    def test_posterior_group_works(self, simple_model_and_samples):
        """Posterior group works by default."""
        model, idata = simple_model_and_samples

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            group="posterior",
            num_samples=50,
            rng=42,
        )

        test_x = np.linspace(-2, 2, 20)
        result = predictor(x=test_x)
        assert "mu" in result

    def test_invalid_group_raises(self, simple_model_and_samples):
        """Invalid group raises ValueError."""
        model, idata = simple_model_and_samples

        with pytest.raises(ValueError, match="Can not extract"):
            create_frozen_predictor(
                model=model,
                inference_data=idata,
                response_exprs={"mu": "mu"},
                input_vars=["x"],
                group="nonexistent_group",
                num_samples=50,
                rng=42,
            )


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of predictions."""

    def test_linear_model_predictions(self, simple_model_and_samples):
        """Predictions should match expected linear relationship."""
        model, idata = simple_model_and_samples

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={"mu": "mu"},
            input_vars=["x"],
            num_samples=100,
            rng=42,
        )

        # Get posterior means
        alpha_mean = idata.posterior["alpha"].mean().values
        beta_mean = idata.posterior["beta"].mean().values

        # Predict at test points
        test_x = np.array([0.0, 1.0, 2.0])
        result = predictor(x=test_x)

        # Mean prediction should be close to posterior mean prediction
        pred_mean = result["mu"].mean(axis=0)
        expected = alpha_mean + beta_mean * test_x

        np.testing.assert_allclose(pred_mean, expected, rtol=0.1)


class TestPytensorPostProcessing:
    """Tests for PyTensor-based post-processing functions."""

    def test_pytensor_percentile_functions(self, simple_model_and_samples):
        """PyTensor post-processing for percentiles works correctly."""
        model, idata = simple_model_and_samples

        n_samples = 100
        idx_low = int(round((n_samples - 1) * 0.05))
        idx_high = int(round((n_samples - 1) * 0.95))

        predictor = create_frozen_predictor(
            model=model,
            inference_data=idata,
            response_exprs={
                "mu_mean": ("mu", lambda x: x.mean(axis=0)),
                "mu_low": ("mu", lambda x: pt.sort(x, axis=0)[idx_low]),
                "mu_high": ("mu", lambda x: pt.sort(x, axis=0)[idx_high]),
            },
            input_vars=["x"],
            num_samples=n_samples,
            rng=42,
        )

        test_x = np.linspace(-2, 2, 20)
        result = predictor(x=test_x)

        # All post-processed outputs should have same shape
        assert result["mu_mean"].shape == (20,)
        assert result["mu_low"].shape == (20,)
        assert result["mu_high"].shape == (20,)

        # Low should be <= mean <= high
        assert np.all(result["mu_low"] <= result["mu_mean"])
        assert np.all(result["mu_mean"] <= result["mu_high"])


class TestProtectedDims:
    """Tests for protected dimension handling in SpecifyShape removal."""

    @pytest.fixture
    def model_with_baseline(self):
        """Create a model with baseline that has date dim - simulates total_sales scenario."""
        np.random.seed(42)
        n_obs = 20
        
        spend_data = np.random.uniform(10, 100, n_obs)
        baseline_data = np.random.uniform(50, 150, n_obs)  # Time-varying baseline
        sales_data = spend_data * 0.5 + baseline_data + np.random.normal(0, 10, n_obs)
        
        with pm.Model(coords={"date": range(n_obs)}) as model:
            spend = pm.Data("spend", spend_data, dims="date")
            baseline = pm.Data("baseline", baseline_data, dims="date")
            
            beta = pm.Normal("beta", mu=0.5, sigma=0.2)
            sigma = pm.HalfNormal("sigma", sigma=20)
            
            # saturated_spend only depends on spend (input)
            saturated_spend = pm.Deterministic(
                "saturated_spend", 
                spend * beta,
                dims="date"
            )
            
            # total_sales depends on both spend AND baseline
            total_sales = pm.Deterministic(
                "total_sales",
                saturated_spend + baseline,
                dims="date"
            )
            
            y = pm.Normal("y", mu=total_sales, sigma=sigma, observed=sales_data, dims="date")
        
        with model:
            idata = pm.sample(
                draws=50,
                tune=50,
                chains=2,
                random_seed=42,
                progressbar=False,
            )
        
        return model, idata, n_obs

    def test_saturation_output_allows_dynamic_sizing(self, model_with_baseline):
        """Test that saturated_spend output (no baseline ancestor) allows dynamic input sizes."""
        model, idata, n_obs = model_with_baseline
        
        # Create predictor for saturated_spend only (baseline is NOT an ancestor)
        predictor = create_frozen_predictor(
            model, idata,
            response_exprs={"saturated": "saturated_spend"},
            input_vars=["spend"],
            num_samples=20,
            rng=42,
        )
        
        # Should work with different input sizes (100 != 20)
        test_spend = np.linspace(0, 200, 100)
        result = predictor(spend=test_spend)
        
        assert result["saturated"].shape == (20, 100)  # (n_samples, n_points)

    def test_total_sales_requires_fixed_dimension(self, model_with_baseline):
        """Test that total_sales output (baseline IS ancestor) requires matching input size."""
        model, idata, n_obs = model_with_baseline
        
        # Create predictor for total_sales (baseline IS an ancestor)
        predictor = create_frozen_predictor(
            model, idata,
            response_exprs={"total": "total_sales"},
            input_vars=["spend"],
            num_samples=20,
            rng=42,
        )
        
        # Should work with original size (20)
        test_spend = np.linspace(0, 200, n_obs)
        result = predictor(spend=test_spend)
        assert result["total"].shape == (20, n_obs)  # (n_samples, n_points)
        
        # Should fail with different size (100 != 20) due to baseline shape mismatch
        test_spend_wrong_size = np.linspace(0, 200, 100)
        with pytest.raises(Exception):  # Shape mismatch error
            predictor(spend=test_spend_wrong_size)

    def test_analyze_outputs_correctly_identifies_protected_dims(self, model_with_baseline):
        """Test that _find_protected_dims only protects dims in actual ancestors."""
        from frozen_predictor import _find_needed_nodes, _find_protected_dims
        
        model, idata, n_obs = model_with_baseline
        
        # For saturated_spend, baseline is NOT an ancestor
        saturated_output = [model["saturated_spend"]]
        _, _, all_ancestors, _ = _find_needed_nodes(model, saturated_output, {"spend"})
        protected_dim_names, _ = _find_protected_dims(model, saturated_output, {"spend"}, all_ancestors)
        
        # No dims should be protected (only spend is ancestor, and it's an input)
        assert len(protected_dim_names) == 0
        
        # For total_sales, baseline IS an ancestor
        total_output = [model["total_sales"]]
        _, _, all_ancestors, _ = _find_needed_nodes(model, total_output, {"spend"})
        protected_dim_names, _  = _find_protected_dims(model, total_output, {"spend"}, all_ancestors)
        
        # Date dim should be protected (baseline has date dim and is an ancestor)
        assert "date" in protected_dim_names


class TestRemoveSpecifyShape:
    """Tests for _remove_specify_shape function."""

    def test_removes_single_specify_shape(self):
        """Single SpecifyShape is removed."""
        from frozen_predictor import _remove_specify_shape
        from pytensor.tensor.shape import SpecifyShape
        from pytensor.graph.basic import ancestors
        
        # Create a tensor with SpecifyShape
        x = pt.vector("x")
        x_shaped = pt.specify_shape(x, (10,))
        y = x_shaped * 2
        
        # Verify SpecifyShape exists before removal
        def has_specify_shape(outputs):
            all_nodes = list(ancestors(outputs)) + list(outputs)
            return any(
                hasattr(n, 'owner') and n.owner and isinstance(n.owner.op, SpecifyShape) 
                for n in all_nodes
            )
        
        assert has_specify_shape([y]), "SpecifyShape should exist before removal"
        
        # Remove SpecifyShape
        result = _remove_specify_shape([y])
        
        assert not has_specify_shape(result), "SpecifyShape should be removed"

    def test_removes_multiple_specify_shapes(self):
        """Multiple independent SpecifyShape ops are removed."""
        from frozen_predictor import _remove_specify_shape
        from pytensor.tensor.shape import SpecifyShape
        from pytensor.graph.basic import ancestors
        
        # Create multiple tensors with SpecifyShape
        x = pt.vector("x")
        w = pt.vector("w")
        x_shaped = pt.specify_shape(x, (10,))
        w_shaped = pt.specify_shape(w, (10,))
        y = x_shaped + w_shaped
        
        def count_specify_shapes(outputs):
            all_nodes = list(ancestors(outputs)) + list(outputs)
            return sum(
                1 for n in all_nodes
                if hasattr(n, 'owner') and n.owner and isinstance(n.owner.op, SpecifyShape)
            )
        
        assert count_specify_shapes([y]) == 2, "Should have 2 SpecifyShape ops before removal"
        
        result = _remove_specify_shape([y])
        
        assert count_specify_shapes(result) == 0, "All SpecifyShape ops should be removed"

    def test_removes_nested_specify_shapes(self):
        """Nested SpecifyShape ops (SpecifyShape(SpecifyShape(x))) are all removed."""
        from frozen_predictor import _remove_specify_shape
        from pytensor.tensor.shape import SpecifyShape
        from pytensor.graph.basic import ancestors
        
        # Create nested SpecifyShape
        x = pt.vector("x")
        x_shaped1 = pt.specify_shape(x, (10,))
        x_shaped2 = pt.specify_shape(x_shaped1, (10,))  # Nested!
        y = x_shaped2 * 2
        
        def count_specify_shapes(outputs):
            all_nodes = list(ancestors(outputs)) + list(outputs)
            return sum(
                1 for n in all_nodes
                if hasattr(n, 'owner') and n.owner and isinstance(n.owner.op, SpecifyShape)
            )
        
        # Note: PyTensor may optimize away nested SpecifyShape with same shape
        # So we just verify all are removed, regardless of count
        initial_count = count_specify_shapes([y])
        assert initial_count >= 1, "Should have at least 1 SpecifyShape op"
        
        result = _remove_specify_shape([y])
        
        assert count_specify_shapes(result) == 0, "All SpecifyShape ops should be removed"

    def test_preserves_protected_dims(self):
        """SpecifyShape with protected dims is preserved."""
        from frozen_predictor import _remove_specify_shape
        from pytensor.tensor.shape import SpecifyShape
        from pytensor.graph.basic import ancestors
        
        # Create tensors - one protected, one not
        x = pt.vector("x")
        w = pt.vector("protected_var")
        x_shaped = pt.specify_shape(x, (10,))
        w_shaped = pt.specify_shape(w, (10,))
        y = x_shaped + w_shaped
        
        def count_specify_shapes(outputs):
            all_nodes = list(ancestors(outputs)) + list(outputs)
            return sum(
                1 for n in all_nodes
                if hasattr(n, 'owner') and n.owner and isinstance(n.owner.op, SpecifyShape)
            )
        
        # Remove with protected dim
        result = _remove_specify_shape(
            [y],
            protected_dim_names={"protected_dim"},
            named_var_dims={"protected_var": ("protected_dim",)}
        )
        
        # One should be preserved, one removed
        assert count_specify_shapes(result) == 1, "Protected SpecifyShape should remain"

    def test_functional_equivalence_after_removal(self):
        """Graph produces same results after SpecifyShape removal."""
        import pytensor
        from frozen_predictor import _remove_specify_shape
        
        # Create a computation with SpecifyShape
        x = pt.vector("x")
        x_shaped = pt.specify_shape(x, (5,))
        y = x_shaped ** 2 + x_shaped * 3
        
        # Compile original
        f_original = pytensor.function([x], [y])
        
        # Remove SpecifyShape and compile
        result = _remove_specify_shape([y])
        f_modified = pytensor.function([x], result)
        
        # Test with original size
        test_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(f_original(test_input), f_modified(test_input))
        
        # Modified version should also work with different sizes
        test_input_diff = np.array([1.0, 2.0, 3.0])
        result_diff = f_modified(test_input_diff)[0]  # Get first output
        expected = test_input_diff ** 2 + test_input_diff * 3
        np.testing.assert_allclose(result_diff, expected)
    
    def test_returns_original_if_no_specify_shapes(self):
        """Returns original outputs if no SpecifyShape ops exist."""
        from frozen_predictor import _remove_specify_shape
        
        # Create a graph without SpecifyShape
        x = pt.vector("x")
        y = x * 2
        
        result = _remove_specify_shape([y])
        
        # Should return the same outputs
        assert result == [y], "Should return original outputs when no SpecifyShape exists"


class TestMultipleDims:
    """Test handling of multiple dimensions (e.g., date and store)."""

    @pytest.fixture
    def model_with_date_and_store(self):
        """Create a model with date and store dims."""
        np.random.seed(42)
        n_dates = 20
        n_stores = 5
        
        spend_data = np.random.uniform(10, 100, n_dates)
        store_effects = np.random.normal(0, 10, n_stores)
        
        with pm.Model(coords={"date": range(n_dates), "store": range(n_stores)}) as model:
            spend = pm.Data("spend", spend_data, dims="date")
            store_intercept = pm.Data("store_intercept", store_effects, dims="store")
            
            beta = pm.Normal("beta", mu=0.5, sigma=0.2)
            
            # Effect only depends on spend (date dim)
            spend_effect = pm.Deterministic("spend_effect", spend * beta, dims="date")
        
        with model:
            prior = pm.sample_prior_predictive(draws=50, random_seed=42)
        
        return model, prior, n_dates, n_stores

    def test_date_dynamic_store_protected(self, model_with_date_and_store):
        """Test that date can be dynamic when store is protected but date is only in inputs."""
        from frozen_predictor import _find_needed_nodes, _find_protected_dims
        
        model, _, _, _ = model_with_date_and_store
        
        # For spend_effect, only spend (date dim) is an ancestor, not store_intercept
        output = [model["spend_effect"]]
        _, _, all_ancestors, _ = _find_needed_nodes(model, output, {"spend"})
        protected_dim_names = _find_protected_dims(model, output, {"spend"}, all_ancestors)
        
        # Store should NOT be protected (not an ancestor of spend_effect)
        # Date should NOT be protected (it's only in the input)
        assert "store" not in protected_dim_names
        assert "date" not in protected_dim_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
