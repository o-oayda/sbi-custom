# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Base class for Approximate Bayesian Computation methods."""

import logging
from typing import Callable, Dict, Optional, Union

import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch import Tensor

from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.metrics import Distance


class ABCBASE:
    """Base class for Approximate Bayesian Computation methods."""

    def __init__(
        self,
        simulator: Callable,
        prior,
        distance: Union[str, Callable] = "l2",
        requires_iid_data: Optional[bool] = None,
        distance_kwargs: Optional[Dict] = None,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        distance_batch_size: int = -1,
        show_progress_bars: bool = True,
    ) -> None:
        r"""Base class for Approximate Bayesian Computation methods.

        Args:
            simulator: A function that takes parameters $\theta$ and maps them to
                simulations, or observations, `x`, $\mathrm{sim}(\theta)\to x$. Any
                regular Python callable (i.e. function or class with `__call__` method)
                can be used.
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            distance: Distance function to compare observed and simulated data. Can be
                a custom callable function or one of `l1`, `l2`, `mse`,
                `mmd`, `wasserstein`.
            requires_iid_data: Whether to allow conditioning on iid sampled data or not.
                Typically, this information is inferred by the choice of the distance,
                but in case a custom distance is used, this information is pivotal.
            distance_kwargs: Configurations parameters for the distances. In particular
                useful for the MMD and Wasserstein distance.
            num_workers: Number of parallel workers to use for simulations.
            simulation_batch_size: Number of parameter sets that the simulator
                maps to data x at once. If None, we simulate all parameter sets at the
                same time. If >= 1, the simulator has to process data of shape
                (simulation_batch_size, parameter_dimension).
            distance_batch_size: Number of simulations that the distance function
                evaluates against the reference observations at once. If -1, we evaluate
                all simulations at the same time.
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        self.prior = prior
        self._simulator = simulator
        self._show_progress_bars = show_progress_bars

        self.x_o = None
        self.x_shape = None

        # Select distance function.
        self.distance = Distance(
            distance, requires_iid_data, distance_kwargs, batch_size=distance_batch_size
        )

        self._batched_simulator = lambda theta: simulate_in_batches(
            simulator=self._simulator,
            theta=theta,
            sim_batch_size=simulation_batch_size,
            num_workers=num_workers,
            show_progress_bars=self._show_progress_bars,
        )

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_sass_transform(
        theta: Tensor,
        x: Tensor,
        expansion_degree: int = 1,
        sample_weight: Optional[Tensor] = None,
    ) -> Callable[[Tensor], Tensor]:
        """Return semi-automatic summary statitics function.

        Running weighted linear regressin as in
        Fearnhead & Prandle 2012: https://arxiv.org/abs/1004.1112

        Following implementation in
        https://abcpy.readthedocs.io/en/latest/_modules/abcpy/statistics.html#Identity
        and
        https://pythonhosted.org/abcpy/_modules/abcpy/summaryselections.html#Semiautomatic
        """
        expansion = PolynomialFeatures(degree=expansion_degree, include_bias=False)
        # Transform x, remove intercept.
        x_expanded = expansion.fit_transform(x.numpy())

        # Fit a single multi-output regression model for all parameters at once
        regression_model = LinearRegression(fit_intercept=True)
        regression_model.fit(
            X=x_expanded,
            y=theta.numpy(),  # All parameters at once
            sample_weight=sample_weight.numpy() if sample_weight is not None else None,
        )

        # Get coefficients for all parameters (shape: [n_features, n_parameters])
        sumstats_map = torch.tensor(regression_model.coef_.T, dtype=torch.float32)

        def sumstats_transform(x):
            x_expanded = torch.tensor(expansion.fit_transform(x), dtype=torch.float32)
            return x_expanded.mm(sumstats_map)

        return sumstats_transform

    @staticmethod
    def _run_lra(
        theta: Tensor,
        x: Tensor,
        observation: Tensor,
        sample_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Return parameters adjusted with linear regression adjustment.

        Implementation as in Beaumont et al. 2002: https://arxiv.org/abs/1707.01254
        """

        # Fit a single multi-output regression model
        regression_model = LinearRegression(fit_intercept=True)
        regression_model.fit(
            X=x.numpy(),
            y=theta.numpy(),  # All parameters at once
            sample_weight=sample_weight.numpy() if sample_weight is not None else None,
        )

        # Predict for observation and simulated data
        pred_obs = regression_model.predict(observation.reshape(1, -1).numpy())
        pred_sim = regression_model.predict(x.numpy())

        # Apply adjustment: theta + m(x_o) - m(x)
        theta_adjusted = theta + torch.from_numpy(pred_obs - pred_sim)

        return theta_adjusted
