"""
Unified posterior predictor implementation.

This module provides a single `create_frozen_predictor` function that handles
both single-scenario (1D) and multi-scenario (2D) inputs seamlessly.

Key features:
- Uses ancestors() to find only the RVs needed for outputs (efficient compilation)
- Supports both 1D inputs (n_points,) and 2D inputs (n_scenarios, n_points)
- Automatically adjusts output shape to match input dimensionality
- Reproducible sampling via seed parameter
- Handles observed intermediate RVs using ICDF replacement
"""

import logging
from collections.abc import Callable
from typing import Any

import arviz as az
import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import Variable, ancestors
from pytensor.graph.replace import clone_replace, vectorize_graph
from pytensor.tensor.shape import SpecifyShape
from pytensor.tensor.variable import TensorVariable

logger = logging.getLogger(__name__)


def _find_needed_nodes(
    model: pm.Model,
    outputs: list[TensorVariable],
    input_var_names: set[str],
) -> tuple[list[TensorVariable], list[TensorVariable], set[Variable], set[Variable]]:
    """
    Find all nodes needed to compute outputs.
    
    Traverses the computation graph backwards from outputs to identify:
    - Which free RVs need posterior samples
    - Which observed RVs need replacements
    - Which input variables are actually used
    
    Parameters
    ----------
    model : pm.Model
        The PyMC model.
    outputs : list[TensorVariable]
        Output expressions to analyze.
    input_var_names : set[str]
        Names of input variables.
    
    Returns
    -------
    needed_free_rvs : list[TensorVariable]
        Free RVs that are ancestors of outputs.
    needed_observed_rvs : list[TensorVariable]
        Observed RVs that are ancestors of outputs.
    all_ancestors : set[Variable]
        All ancestors of outputs (full traversal, no blocking).
    input_ancestors : set[Variable]
        Input variables that are actually used in computing outputs.
    """
    free_rvs_set = set(model.free_RVs)
    observed_rvs_set = set(model.observed_RVs)
    
    # Find all ancestors of outputs (full traversal, no blocking).
    all_ancestors = set(ancestors(outputs))
    
    # Needed RVs are simply the intersection with the full ancestry
    needed_free_rvs = all_ancestors & free_rvs_set
    needed_observed_rvs = all_ancestors & observed_rvs_set
    
    # Find input variables that are actually used
    input_ancestors = {
        model.named_vars[name]
        for name in input_var_names
        if model.named_vars[name] in all_ancestors
    }

    unused_input_ancestors = input_ancestors - all_ancestors
    if unused_input_ancestors:
        logger.warning(
            f"Input variables {sorted(unused_input_ancestors)} are not ancestors of outputs. "
            f"These inputs will have no effect on predictions."
        )
    
    return needed_free_rvs, needed_observed_rvs, all_ancestors, input_ancestors


def _find_protected_dims(
    model: pm.Model,
    outputs: list[TensorVariable],
    input_var_names: set[str],
    all_ancestors: set[Variable],
) -> set[str]:
    """
    Find input dimensions that must stay fixed as they are also present in
    non-input ancestors of the outputs.
    
    Parameters
    ----------
    model : pm.Model
    outputs : list[TensorVariable]
    input_var_names : set[str]
    all_ancestors : set[Variable]
    
    Returns
    -------
    set[str]
        Dimension names that must stay fixed.
    set[str]
        Dimension names that are used by input variables.
    """
    output_var_names = {out.name for out in outputs if out.name is not None}
    ancestor_var_names = {
        var.name for var in (all_ancestors - set(outputs))
        if var.name in model.named_vars
    }
    
    # Dims from input variables
    input_dim_names = {
        dim
        for name in input_var_names & model.named_vars_to_dims.keys()
        for dim in model.named_vars_to_dims[name]
    }
    
    # Dims from non-input ancestors (excluding outputs)
    non_input_ancestor_dim_names = {
        dim
        for var_name in (ancestor_var_names - output_var_names - input_var_names) & model.named_vars_to_dims.keys()
        for dim in model.named_vars_to_dims[var_name]
    }
    
    # Protected = dims used by inputs AND by other ancestors
    return input_dim_names & non_input_ancestor_dim_names, input_dim_names


def _replace_observed_rvs(
    model: pm.Model,
    needed_observed_rvs: list[TensorVariable],
    input_var_set: set[Variable],
    has_dynamic_dims: bool = True,
) -> tuple[dict[TensorVariable, Variable], dict[TensorVariable, TensorVariable]]:
    """
    Create replacement expressions for observed RVs.
    
    For input-independent observed RVs, uses the observed data as a constant.
    For input-dependent observed RVs, uses ICDF with uniform placeholders.
    
    Parameters
    ----------
    model : pm.Model
        The PyMC model.
    needed_observed_rvs : list[TensorVariable]
        Observed RVs that need replacements.
    input_var_set : set[Variable]
        Input variables that are ancestors of outputs.
    has_dynamic_dims : bool
        Whether the input dimensions are dynamic (no protected dims).
        If True and an observed RV has fixed shape, a warning is logged.
    
    Returns
    -------
    tuple[dict, dict]
        - replacements: Mapping from observed RVs to their replacement expressions.
        - uniform_placeholders: Mapping from observed RVs to their uniform placeholders
          (only for input-dependent RVs). These should be replaced with (num_samples,)
          uniform arrays during sample vectorization.
        
    Raises
    ------
    NotImplementedError
        If ICDF is not available for an input-dependent observed RV.
    """
    replacements = {}
    uniform_placeholders = {}
    
    for rv in needed_observed_rvs:
        depends_on_input = bool(set(ancestors(rv.owner.inputs[2:])) & input_var_set)
        
        if not depends_on_input:
            # Input-independent: use observed data as constant
            observed_data = model.rvs_to_values[rv]
            if hasattr(observed_data, 'get_value'):
                data_values = observed_data.get_value()
            else:
                data_values = observed_data.eval()
            replacements[rv] = pt.constant(data_values.astype(rv.dtype), name=rv.name)
        else:
            # Input-dependent: use ICDF with uniform placeholder
            logger.info(
                f"Observed RV '{rv.name}' depends on inputs. Replacing with ICDF "
                f"using uniform placeholders for predictive sampling."
            )
            
            # Warn about fixed shapes only if dynamic sizing is expected
            if has_dynamic_dims and rv.type.shape and any(s is not None for s in rv.type.shape):
                logger.warning(
                    f"Observed RV '{rv.name}' has fixed shape {rv.type.shape}. "
                    f"Use 'dims' on observed RVs to allow dynamic input sizes."
                )
            
            # Create a scalar placeholder for the uniform quantile
            uniform_placeholder = pt.scalar(f"{rv.name}_uniform", dtype=rv.dtype)
            uniform_placeholders[rv] = uniform_placeholder
            
            try:
                replacements[rv] = pm.icdf(rv, uniform_placeholder, warn_rvs=False)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"ICDF is not available for observed RV '{rv.name}' (distribution: {rv.owner.op}). "
                    f"This distribution does not support the inverse CDF method required for "
                    f"input-dependent observed RVs. Original error: {e}"
                ) from e
    
    return replacements, uniform_placeholders


def _remove_specify_shape(
    outputs: list[TensorVariable], 
    protected_dim_names: set[str] | None = None, 
    named_var_dims: dict[str, tuple[str, ...]] | None = None,
) -> list[TensorVariable]:
    """
    Remove SpecifyShape ops from the graph to allow dynamic input sizes.
    
    - Named variables with protected dims: Keep SpecifyShape (must stay fixed)
    - Named variables with only dynamic dims: Remove SpecifyShape
    - Unnamed tensors with SpecifyShape: Warn and remove
    
    Parameters
    ----------
    outputs : list
        PyTensor output expressions.
    protected_dim_names : set, optional
        Dimension names that must stay fixed (from graph analysis).
        If None, all SpecifyShape ops are removed.
    named_var_dims : dict, optional
        Mapping from variable names to their dim tuples.
        If None, all SpecifyShape ops are removed.
    
    Returns
    -------
    list
        Output expressions with appropriate SpecifyShape ops removed.
    """
    # Default to empty if not provided (removes all SpecifyShape)
    if protected_dim_names is None:
        protected_dim_names = set()
    if named_var_dims is None:
        named_var_dims = {}
    
    # Find all SpecifyShape ops and build replacement dict
    replacements = {}
    for node in list(ancestors(outputs)) + list(outputs):
        if hasattr(node, 'owner') and node.owner and isinstance(node.owner.op, SpecifyShape):
            tensor = node.owner.inputs[0]
            shape_inputs = node.owner.inputs[1:]
            
            # Check if this is a named variable with protected dims
            is_named = tensor.name is not None and tensor.name in named_var_dims
            
            if is_named:
                var_dims = named_var_dims.get(tensor.name, ())
                is_protected = bool(set(var_dims) & protected_dim_names)
                
                if is_protected:
                    continue  # Keep SpecifyShape for protected dims
            else:
                # Unnamed tensor with SpecifyShape - warn and remove
                sizes = [s.data if hasattr(s, 'data') else '?' for s in shape_inputs]
                logger.warning(
                    f"Found SpecifyShape on unnamed tensor with sizes {sizes}. "
                    f"Removing constraint - if this causes shape errors, consider "
                    f"naming intermediate variables with explicit dims."
                )
            
            # Add to replacements: SpecifyShape output -> its input
            replacements[node] = tensor
    
    if not replacements:
        return outputs
    
    # Apply replacements using clone_replace with rebuild_strict=False
    # This is necessary because the replacement targets might have different
    # types than their replacements (SpecifyShape enforces specific shapes).
    return clone_replace(outputs, replace=replacements, rebuild_strict=False)



def create_frozen_predictor(
    model: pm.Model,
    inference_data: az.InferenceData,
    response_exprs: dict[str, str | TensorVariable | tuple[str | TensorVariable, Callable]],
    input_vars: list[str],
    **extract_kwargs: Any,
) -> Callable[..., dict[str, npt.NDArray[np.floating[Any]]]]:
    """
    Create a compiled PyTensor function for posterior predictions.

    This function creates a predictor that evaluates model expressions at new
    input values, vectorized over posterior (or prior) samples. It handles both
    single-scenario and multi-scenario inputs automatically.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model containing the expressions to evaluate.
    inference_data : arviz.InferenceData
        ArviZ InferenceData object containing posterior or prior samples.
    response_exprs : dict[str, expr_spec]
        Mapping from output names to expressions. Each value can be:
        - str: Name of a model variable (e.g., "saturated_spend")
        - TensorVariable: A PyTensor expression from the model
        - tuple (expr, post_fn): Expression with a post-processing function.
          post_fn receives the raw output with shape (n_samples, [n_scenarios,] n_points)
          and should return the transformed result (e.g., lambda x: x.mean(axis=0))
    input_vars : list of str
        Names of pm.Data variables to use as inputs (e.g., ["spend_direct", "spend_upper"]).
        These will be replaced with the values provided at prediction time.
    **extract_kwargs
        Keyword arguments passed to `arviz.extract()`. Common options:
        - num_samples : int - Number of posterior samples to use (default: all)
        - group : str - Which group to use, e.g., "posterior" or "prior" (default: "posterior")
        - rng : int, bool, or None - Random seed for reproducible sampling.
          Use int for reproducibility, False to disable shuffling, None for random.
        See `arviz.extract` documentation for all available options.

    Returns
    -------
    predict_fn : callable
        Prediction function with signature: predict_fn(**inputs) -> dict
        
        Inputs can be:
        - 1D arrays of shape (n_points,): Single scenario
        - 2D arrays of shape (n_scenarios, n_points): Multiple scenarios
        - Scalars: Broadcast to match other inputs
        
        Output shapes (before post-processing):
        - 1D input: (n_samples, n_points)
        - 2D input: (n_samples, n_scenarios, n_points)

    Handling Observed Intermediate Variables
    ----------------------------------------
    When response expressions depend on observed random variables (not just free RVs),
    special handling is required. This commonly occurs in mediation models where an
    intermediate variable is observed.

    **Example:** In a model where ``visits ~ Normal(f(spend), sigma)`` and 
    ``purchases ~ Normal(g(visits), sigma2)``, the ``visits`` RV is observed but also
    depends on the input ``spend``. When predicting ``purchases`` at new spend values,
    we need to handle ``visits`` appropriately.

    The predictor classifies observed RVs into two categories:

    1. **Input-independent observed RVs**: Their distribution parameters don't depend
       on any ``input_vars``. These are replaced with their observed data values.

    2. **Input-dependent observed RVs**: Their distribution parameters depend on 
       ``input_vars``. These are replaced using the ICDF (inverse CDF) method:
       Uniform quantiles U ~ Uniform(0,1) are sampled once at predictor creation
       and frozen. The observed RV is replaced with ICDF(distribution, U),
       expressing it as a deterministic function of inputs and frozen quantiles.

    **Mathematical basis:** If U ~ Uniform(0,1), then F^{-1}(U) follows the target
    distribution (inverse transform sampling). By freezing U at creation time, we
    capture the scale of predictive variability while making predictions deterministic.

    To get different predictive realizations, create a new predictor with a different
    ``rng`` seed in ``extract_kwargs``.

    **Important: Dynamic Input Sizes**
    
    To use predictors with input sizes different from the original data, the model
    must be defined with ``dims`` on observed RVs, Deterministics, and Data variables.
    Without ``dims``, observed RVs inherit fixed shapes from the observed data, which
    prevents dynamic input sizes. A warning is logged if fixed-shape observed RVs are
    detected.
    
    Example of model definition supporting dynamic input sizes::
    
        with pm.Model(coords={"time": range(n_obs)}) as model:
            x = pm.Data("x", x_data, dims="time")                    # WITH dims
            mu = pm.Deterministic("mu", alpha + beta * x, dims="time")  # WITH dims
            y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data, dims="time")  # WITH dims

    Examples
    --------
    Basic usage with 1D input:
    
    >>> pred_fn = create_frozen_predictor(
    ...     model, idata,
    ...     response_exprs={"saturated": "saturated_spend"},
    ...     input_vars=["spend"],
    ... )
    >>> result = pred_fn(spend=np.linspace(0, 100, 50))
    >>> result["saturated"].shape  # (n_samples, 50)

    With post-processing to get mean prediction:
    
    >>> pred_fn = create_frozen_predictor(
    ...     model, idata,
    ...     response_exprs={"saturated": ("saturated_spend", lambda x: x.mean(axis=0))},
    ...     input_vars=["spend"],
    ...     num_samples=100,
    ... )
    >>> result = pred_fn(spend=np.linspace(0, 100, 50))
    >>> result["saturated"].shape  # (50,) - mean over samples

    Multi-scenario (2D) input for response curves:
    
    >>> factors = np.linspace(0.5, 2.0, 20)
    >>> baseline = np.linspace(0, 100, 50)
    >>> spend_scenarios = np.outer(factors, baseline)  # (20, 50)
    >>> result = pred_fn(spend=spend_scenarios)
    >>> result["saturated"].shape  # (n_samples, 20, 50)
    
    Using prior samples with reproducible selection:
    
    >>> pred_fn = create_frozen_predictor(
    ...     model, idata,
    ...     response_exprs={"mu": "mu"},
    ...     input_vars=["x"],
    ...     group="prior",
    ...     num_samples=50,
    ...     rng=42,
    ... )
    """

    # Validate input_vars exist in model
    missing_vars = set(input_vars) - set(model.named_vars.keys())
    if missing_vars:
        raise ValueError(f"input_vars not found in model: {missing_vars}")

    # Validate extract_kwargs - combined=False is not supported
    if extract_kwargs.get("combined") is False:
        raise ValueError(
            "combined=False is not supported. This function requires the `chain` and `draw` dimensions to be "
            "flattened into a single 'sample' dimension (combined=True, the default)."
        )

    # Extract posterior samples using arviz (handles chain flattening + sampling)
    posterior = az.extract(inference_data, **extract_kwargs).transpose("sample", ...)

    # Process response expressions into PyTensor outputs and optional post-processing functions
    output_names = list(response_exprs.keys())
    outputs = []
    post_fns = []
    for expr_spec in response_exprs.values():
        if isinstance(expr_spec, tuple):
            expr, post_fn = expr_spec
        else:
            expr, post_fn = expr_spec, None
        outputs.append(model[expr] if isinstance(expr, str) else expr)
        post_fns.append(post_fn)

    rng_seed = extract_kwargs.get("rng", None)
    needed_free_rvs, needed_observed_rvs, all_ancestors, input_var_set = _find_needed_nodes(
        model, outputs, input_vars
    )
    protected_dim_names, input_dim_names = _find_protected_dims(model, outputs, input_var_set, all_ancestors)
    
    # Check if any input dims are dynamic (not used by input variables)
    # If inputs have no named dims, we can't determine - assume dynamic (warn if fixed shape)
    has_dynamic_dims = not input_dim_names or bool(input_dim_names - protected_dim_names)

    # Create placeholders for free RVs (will be replaced with posterior samples later)
    rv_placeholders = {
        rv: pt.tensor(name=rv.name, shape=rv.type.shape, dtype=rv.dtype)
        for rv in needed_free_rvs
    }

    # Create symbolic inputs (1D for initial graph, will be vectorized to 2D)
    symbolic_inputs_1d = {
        name: pt.vector(f"{name}_1d", dtype="float64")
        for name in input_vars
    }
    symbolic_inputs_2d = {
        name: pt.matrix(f"{name}_in", dtype="float64")
        for name in input_vars
    }

    # Build data variable replacements
    data_replacements = {
        model.named_vars[name]: symbolic_inputs_1d[name]
        for name in input_vars
    }
    
    # Get observed RV replacements and uniform placeholders
    observed_rv_replacements, uniform_placeholders = _replace_observed_rvs(
        model, needed_observed_rvs, input_var_set, has_dynamic_dims
    )

    # Clone the ICDF replacement expressions to use placeholders instead of original RVs.
    # Include uniform placeholders mapped to themselves to preserve their identity.
    base_replacements = {
        **rv_placeholders,
        **data_replacements,
        **{p: p for p in uniform_placeholders.values()},
    }
    if observed_rv_replacements:
        cloned_icdf_exprs = clone_replace(
            list(observed_rv_replacements.values()), 
            replace=base_replacements
        )
        observed_rv_replacements = dict(zip(observed_rv_replacements.keys(), cloned_icdf_exprs))

    # Build complete replacement dict and clone outputs
    graph_replacements = {
        **base_replacements,
        **observed_rv_replacements,
    }
    cloned_outputs = clone_replace(outputs, replace=graph_replacements)

    # Vectorize: add scenario dimension (1D -> 2D inputs)
    scenario_replacements = {
        symbolic_inputs_1d[name]: symbolic_inputs_2d[name]
        for name in input_vars
    }
    batched_outputs = vectorize_graph(cloned_outputs, replace=scenario_replacements)

    # Vectorize: add sample dimension (replace placeholders with posterior samples)
    num_samples = posterior.sizes["sample"]
    sample_replacements = {
        placeholder: pt.constant(
            posterior[placeholder.name].values.astype(placeholder.dtype),
            name=placeholder.name,
        )
        for placeholder in rv_placeholders.values()
    }
    
    # Add uniform samples for ICDF replacements (one quantile per posterior sample)
    uniform_rng = np.random.default_rng(rng_seed if rng_seed not in (None, False) else 42)
    for rv, uniform_placeholder in uniform_placeholders.items():
        uniform_samples = uniform_rng.uniform(0, 1, size=num_samples).astype(uniform_placeholder.dtype)
        sample_replacements[uniform_placeholder] = pt.constant(
            uniform_samples,
            name=uniform_placeholder.name,
        )
    
    outputs_with_samples = vectorize_graph(batched_outputs, replace=sample_replacements)

    # remove SpecifyShape ops (respecting protected dims)
    final_outputs_raw = _remove_specify_shape(
        outputs_with_samples,
        protected_dim_names=protected_dim_names,
        named_var_dims=dict(model.named_vars_to_dims),
    )

    # Apply post-processing functions
    final_outputs = [
        post_fn(out) if post_fn is not None else out
        for out, post_fn in zip(final_outputs_raw, post_fns)
    ]

    # Compile the function
    ordered_inputs = [symbolic_inputs_2d[name] for name in input_vars]
    compiled_fn = pytensor.function(
        inputs=ordered_inputs,
        outputs=final_outputs,
        on_unused_input='ignore',
    )

    def predict_fn(**kwargs):
        """
        Compute predictions at given input values.

        Parameters
        ----------
        **kwargs : array-like
            Input values keyed by variable name. Can be:
            - Scalars: Broadcast to match other inputs
            - 1D arrays (n_points,): Single scenario
            - 2D arrays (n_scenarios, n_points): Multiple scenarios

        Returns
        -------
        dict
            Mapping from output names to result arrays.
            Shape depends on input dimensionality and post-processing.
        """
        first_input = kwargs[input_vars[0]]
        input_was_1d = np.asarray(first_input).ndim <= 1

        # Convert all inputs to 2D and run
        arrays = [np.atleast_2d(kwargs[name]).astype(np.float64) for name in input_vars]
        results = compiled_fn(*arrays)

        # Build output dict, squeezing scenario dimension if input was 1D
        output = {}
        for name, result in zip(output_names, results):
            if input_was_1d and result.ndim >= 2:
                # After post-processing (e.g., mean over samples), the scenario dim
                # could be at axis 0 or axis 1 depending on what was reduced.
                # Squeeze any axis that has size 1 (the scenario dim for 1D input).
                if result.shape[0] == 1:
                    result = result.squeeze(axis=0)
                elif result.ndim >= 2 and result.shape[1] == 1:
                    result = result.squeeze(axis=1)
            output[name] = result

        return output

    return predict_fn
