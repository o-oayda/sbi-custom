# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor, float32
from torch.distributions import (
    Bernoulli,
    Binomial,
    Categorical,
    Distribution,
    Multinomial,
    MultivariateNormal,
    constraints,
)

from sbi.utils.torchutils import process_device


def get_distribution_parameters(
    dist: Distribution, device: Union[str, torch.device]
) -> Dict:
    """Used to get the tensors of the parameters in torch distributions.

    Returns the tensors relocated to device.
    """
    params = {param: getattr(dist, param).to(device) for param in dist.arg_constraints}
    # MultivariateNormal calculates precision matrix from covariance, and stores it in
    # the arg_constraints. When reinstantiating, we must provide only one of them.
    if isinstance(dist, MultivariateNormal):
        params["precision_matrix"] = None
        params["scale_tril"] = None
    # MultivariateNormal calculates logits from probabilities, and stores it in the
    # arg_constraints. When reinstantiating, we must provide only one of them.
    elif isinstance(dist, (Binomial, Bernoulli, Categorical, Multinomial)):
        params["logits"] = None
    return params


class CustomPriorWrapper(Distribution):
    def __init__(
        self,
        custom_prior,
        return_type: Optional[torch.dtype] = float32,
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        validate_args=None,
        arg_constraints: Optional[Dict[str, constraints.Constraint]] = None,
        lower_bound: Optional[Tensor] = None,
        upper_bound: Optional[Tensor] = None,
    ):
        self.custom_arg_constraints = arg_constraints or {}
        self.custom_prior = custom_prior
        self.return_type = return_type

        self.custom_support = build_support(lower_bound, upper_bound)

        self._set_mean_and_variance()

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, value) -> Tensor:
        return torch.as_tensor(
            self.custom_prior.log_prob(value),
            dtype=self.return_type,  # type: ignore
        )

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return torch.as_tensor(
            self.custom_prior.sample(sample_shape),
            dtype=self.return_type,  # type: ignore
        )

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.custom_arg_constraints

    @property
    def support(self) -> constraints.Constraint:
        return self.custom_support

    def _set_mean_and_variance(self):
        """Set mean and variance if available, else estimate from samples."""

        if hasattr(self.custom_prior, "mean"):
            pass
        else:
            self.custom_prior.mean = torch.mean(
                torch.as_tensor(self.custom_prior.sample((1000,))), dim=0
            )
            warnings.warn(
                "Prior is lacking mean attribute, estimating prior mean from samples.",
                UserWarning,
                stacklevel=2,
            )
        if hasattr(self.custom_prior, "variance"):
            pass
        else:
            self.custom_prior.variance = (
                torch.std(torch.as_tensor(self.custom_prior.sample((1000,))), dim=0)
                ** 2
            )
            warnings.warn(
                "Prior is lacking variance attribute, estimating prior variance from "
                "samples.",
                UserWarning,
                stacklevel=2,
            )

    def _move_support_to_device(self, device: Union[str, torch.device]) -> None:
        '''
        If the support ranges were put on a different device originally,
        then they will remain on that device. We need to access the support
        explicitly and transfer to the right device, otherwise there will be a
        device mismatch error when the parameter transform is called.
        '''
        base_constraint = self.support.base_constraint
        base_constraint.lower_bound = base_constraint.lower_bound.to(device)
        base_constraint.upper_bound = base_constraint.upper_bound.to(device)

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Move the distribution to the specified device. Calls custom prior's to
        method, requiring the user to define it.
        """
        assert hasattr(self.custom_prior, "to"), (
            "custom_prior must implement a 'to' method."
        )
        self.custom_prior.to(device)
        self._move_support_to_device(device)
        self.device = device

    @property
    def mean(self):
        return torch.as_tensor(
            self.custom_prior.mean,
            dtype=self.return_type,  # type: ignore
        )

    @property
    def variance(self):
        return torch.as_tensor(
            self.custom_prior.variance,
            dtype=self.return_type,  # type: ignore
        )


class PytorchReturnTypeWrapper(Distribution):
    """Wrap PyTorch Distribution to return a given return type."""

    def __init__(
        self,
        prior: Distribution,  # type: ignore
        return_type: Optional[torch.dtype] = float32,
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        validate_args=None,
    ):
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=(
                prior._validate_args if validate_args is None else validate_args
            ),
        )

        self.prior = prior
        self.device = None
        self.return_type = return_type

    def log_prob(self, value) -> Tensor:
        return torch.as_tensor(
            self.prior.log_prob(value),
            dtype=self.return_type,  # type: ignore
        )

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return torch.as_tensor(
            self.prior.sample(sample_shape),
            dtype=self.return_type,  # type: ignore
        )

    @property
    def mean(self):
        return torch.as_tensor(
            self.prior.mean,
            dtype=self.return_type,  # type: ignore
        )

    @property
    def variance(self):
        return torch.as_tensor(
            self.prior.variance,
            dtype=self.return_type,  # type: ignore
        )

    @property
    def support(self):
        return self.prior.support

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Move the distribution to the specified device.

        Moves the distribution parameters to the specific device
        and updates the device attribute.

        Args:
            device: device to move the distribution to.
        """
        params = get_distribution_parameters(self.prior, device)
        self.prior = type(self.prior)(**params)
        self.device = device


class MultipleIndependent(Distribution):
    """Wrap a sequence of PyTorch distributions into a joint PyTorch distribution."""

    def __init__(
        self,
        dists: Sequence[Distribution],
        validate_args: Optional[bool] = None,
        arg_constraints: Optional[Dict[str, constraints.Constraint]] = None,
        device: Optional[str] = None,
    ):
        """Joint distribution of multiple independent :class:`torch.distributions`.

        Every element of the sequence is treated as independent from the \
        other elements. Single elements can be multivariate with dependent dimensions.

        Args:
            dists: Sequence of PyTorch distributions.
            validate_args (Optional): If True, the distribution checks its parameters.
            arg_constraints (Optional): Dictionary of constraints for the parameters \
                of the distribution.
            device (Optional): Device to move the distribution to. If None, \
                the distribution is moved to the CPU.

        Example:
        --------

        ::

            import torch
            from torch.distributions import Gamma, Beta, MultivariateNormal
            from sbi.utils.user_input_checks_utils import MultipleIndependent

            prior = MultipleIndependent([
                Gamma(torch.zeros(1), torch.ones(1)),
                Beta(torch.zeros(1), torch.ones(1)),
                MultivariateNormal(torch.ones(2), torch.tensor([[1, .1], [.1, 1.]]))
            ])
        """
        self._check_distributions(dists)
        if validate_args is not None:
            [d.set_default_validate_args(validate_args) for d in dists]

        self.dists = dists
        self.device = process_device(device or "cpu")
        self.to(self.device)
        # numel() instead of event_shape because for all dists both is possible,
        # event_shape=[1] or batch_shape=[1]
        self.dims_per_dist = [d.sample().numel() for d in self.dists]

        self.ndims = int(torch.sum(torch.as_tensor(self.dims_per_dist)).item())
        self.custom_arg_constraints = arg_constraints or {}
        self.validate_args = validate_args

        super().__init__(
            batch_shape=torch.Size([]),  # batch size was ensured to be <= 1 above.
            event_shape=torch.Size([
                self.ndims
            ]),  # Event shape is the sum of all ndims.
            validate_args=validate_args,
        )

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """Return argument constraints."""
        return self.custom_arg_constraints

    def _check_distributions(self, dists):
        """Check if dists is Sequence and longer 1 and check every member."""
        assert isinstance(
            dists, Sequence
        ), f"""The combination of independent priors must be of type Sequence, is
               {type(dists)}."""
        assert len(dists) > 1, "Provide at least 2 distributions to combine."
        # Check every element of the sequence.
        [self._check_distribution(d) for d in dists]

    def _check_distribution(self, dist: Distribution):
        """Check type and shape of a single input distribution."""

        assert not isinstance(dist, (MultipleIndependent, Sequence)), (
            "Nesting of combined distributions is not possible."
        )
        assert isinstance(
            dist, Distribution
        ), """priors passed to MultipleIndependent must be PyTorch distributions. Make \
            sure to process custom priors individually using :func:`process_prior` \
            before passing them in a list to :func:`process_prior`."""
        # Make sure batch shape is smaller or equal to 1.
        assert dist.batch_shape in (
            torch.Size([1]),
            torch.Size([0]),
            torch.Size([]),
        ), "The batch shape of every distribution must be smaller or equal to 1."

        assert (
            len(dist.batch_shape) > 0 or len(dist.event_shape) > 0
        ), """One of the distributions you passed is defined over a scalar only. Make
        sure pass distributions with one of event_shape or batch_shape > 0: For example
            - instead of Uniform(0.0, 1.0) pass Uniform(torch.zeros(1), torch.ones(1))
            - instead of Beta(1.0, 2.0) pass Beta(tensor([1.0]), tensor([2.0])).
        """

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        # Sample from every sub distribution and concatenate samples.
        sample = torch.cat([d.sample(sample_shape) for d in self.dists], dim=-1)

        # This reshape is needed to cover the case .sample() vs. .sample((n, )).
        if sample_shape == torch.Size():
            sample = sample.reshape(self.ndims)
        else:
            sample = sample.reshape(-1, self.ndims)

        return sample

    def log_prob(self, value) -> Tensor:
        value = self._prepare_value(value)

        # Evaluate value per distribution, taking into account that individual
        # distributions can be multivariate.
        num_samples = value.shape[0]
        log_probs = []
        dims_covered = 0
        for idx, d in enumerate(self.dists):
            ndims = int(self.dims_per_dist[idx])
            v = value[:, dims_covered : dims_covered + ndims]
            # Reshape here to ensure all returned log_probs are 2D for concatenation.
            log_probs.append(d.log_prob(v).reshape(num_samples, 1))
            dims_covered += ndims

        # Sum accross last dimension to get joint log prob over all distributions.
        return torch.cat(log_probs, dim=1).sum(-1)

    def _prepare_value(self, value) -> Tensor:
        """Return input value with fixed shape.

        Raises:
            AssertionError: if value has more than 2 dimensions or invalid size in
                2nd dimension.
        """

        if value.ndim < 2:
            value = value.unsqueeze(0)

        assert value.ndim == 2, (
            f"value in log_prob must have ndim <= 2, it is {value.ndim}."
        )

        batch_shape, num_value_dims = value.shape

        assert num_value_dims == self.ndims, (
            f"Number of dimensions must match dimensions of this joint: {self.ndims}."
        )

        return value

    @property
    def mean(self) -> Tensor:
        return torch.cat([d.mean for d in self.dists])

    @property
    def variance(self) -> Tensor:
        return torch.cat([d.variance for d in self.dists])

    @property
    def support(self):
        # First, we remove all `independent` constraints. This applies to e.g.
        # `MultivariateNormal`. An `independent` constraint returns a 1D `[True]`
        # when `.support.check(sample)` is called, whereas distributions that are
        # not `independent` (e.g. `Gamma`), return a 2D `[[True]]`. When such
        # constraints would be combined with the `constraint.cat(..., dim=1)`, it
        # fails because the `independent` constraint returned only a 1D `[True]`.
        supports = []
        for d in self.dists:
            if isinstance(d.support, constraints.independent):
                supports.append(d.support.base_constraint)
            else:
                supports.append(d.support)
        # Wrap as `independent` in order to have the correct shape of the
        # `log_abs_det`, i.e. summed over the parameter dimensions.
        return constraints.independent(
            constraints.cat(supports, dim=-1, lengths=self.dims_per_dist),
            reinterpreted_batch_ndims=1,
        )

    def to(self, device: Union[str, torch.device]) -> None:
        """Move the distribution to the specified device.

        If the distribution has the `to` method, it is used. Otherwise, the
        parameters of the distribution are moved to the specified device.

        Args:
            device: device to move the distribution to.
        """
        for i in range(len(self.dists)):
            # ignoring because it is related to torch and not sbi
            if hasattr(self.dists[i], "to"):
                self.dists[i].to(device)  # type: ignore
            else:
                params = get_distribution_parameters(self.dists[i], device)
                self.dists[i] = type(self.dists[i])(**params)  # type: ignore
        self.device = device


def build_support(
    lower_bound: Optional[Tensor] = None, upper_bound: Optional[Tensor] = None
) -> constraints.Constraint:
    """Return support for prior distribution, depending on available bounds.

    Args:
        lower_bound: lower bound of the prior support, can be None
        upper_bound: upper bound of the prior support, can be None

    Returns:
        support: Pytorch constraint object.
    """

    # Support is real if no bounds are passed.
    if lower_bound is None and upper_bound is None:
        support = constraints.real
        warnings.warn(
            "No prior bounds were passed, consider passing lower_bound "
            "and / or upper_bound if your prior has bounded support.",
            stacklevel=2,
        )
    # Only lower bound is specified.
    elif upper_bound is None:
        num_dimensions = lower_bound.numel()  # type: ignore
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.greater_than(lower_bound),
                1,
            )
        else:
            support = constraints.greater_than(lower_bound)
    # Only upper bound is specified.
    elif lower_bound is None:
        num_dimensions = upper_bound.numel()
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.less_than(upper_bound),
                1,
            )
        else:
            support = constraints.less_than(upper_bound)

    # Both are specified.
    else:
        num_dimensions = lower_bound.numel()
        assert num_dimensions == upper_bound.numel(), (
            "There must be an equal number of independent bounds."
        )
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.interval(lower_bound, upper_bound),
                1,
            )
        else:
            support = constraints.interval(lower_bound, upper_bound)

    return support


class OneDimPriorWrapper(Distribution):
    """Wrap batched 1D distributions to get rid of the batch dim of `.log_prob()`.

    1D pytorch distributions such as `torch.distributions.Exponential`, `.Uniform`, or
    `.Normal` do not, by default return __any__ sample or batch dimension. E.g.:
    ```python
    dist = torch.distributions.Exponential(torch.tensor(3.0))
    dist.sample((10,)).shape  # (10,)
    ```

    `sbi` will raise an error that the sample dimension is missing. A simple solution is
    to add a batch dimension to `dist` as follows:
    ```python
    dist = torch.distributions.Exponential(torch.tensor([3.0]))
    dist.sample((10,)).shape  # (10, 1)
    ```

    Unfortunately, this `dist` will return the batch dimension also for `.log_prob():
    ```python
    dist = torch.distributions.Exponential(torch.tensor([3.0]))
    samples = dist.sample((10,))
    dist.log_prob(samples).shape  # (10, 1)
    ```

    This will lead to unexpected errors in `sbi`. The point of this class is to wrap
    those batched 1D distributions to get rid of their batch dimension in `.log_prob()`.
    """

    def __init__(self, prior: Distribution, validate_args=None) -> None:
        super().__init__(
            batch_shape=prior.batch_shape,
            event_shape=prior.event_shape,
            validate_args=(
                prior._validate_args if validate_args is None else validate_args
            ),
        )
        self.prior = prior
        self.device = None

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Move the distribution to the specified device.

        Moves the distribution parameters to the specific device
        and updates the device attribute.

        Args:
            device: device to move the distribution to.
        """
        params = get_distribution_parameters(self.prior, device)
        self.prior = type(self.prior)(**params)
        self.device = device

    def sample(self, *args, **kwargs) -> Tensor:
        return self.prior.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs) -> Tensor:
        """Override the log_prob method to get rid of the additional batch dimension."""
        return self.prior.log_prob(*args, **kwargs)[..., 0]

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.prior.arg_constraints

    @property
    def support(self):
        return self.prior.support

    @property
    def mean(self) -> Tensor:
        return self.prior.mean

    @property
    def variance(self) -> Tensor:
        return self.prior.variance
