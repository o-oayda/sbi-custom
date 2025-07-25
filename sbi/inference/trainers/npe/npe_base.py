# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union
from warnings import warn

import torch
from torch import Tensor, ones
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Self

from sbi.inference.posteriors import (
    DirectPosterior,
)
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials import posterior_estimator_based_potential
from sbi.inference.potentials.posterior_based_potential import PosteriorBasedPotential
from sbi.inference.trainers.base import NeuralInference, check_if_proposal_has_default_x
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.sbi_types import TorchTransform
from sbi.utils import (
    RestrictedPrior,
    check_estimator_arg,
    handle_invalid_x,
    nle_nre_apt_msg_on_invalid_x,
    npe_msg_on_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
)
from sbi.utils.sbiutils import ImproperEmpirical, mask_sims_from_prior
from sbi.utils.torchutils import assert_all_finite


class PosteriorEstimatorTrainer(NeuralInference, ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        """Base class for Sequential Neural Posterior Estimation methods.

        We note that the different subclasses (NPE_A, NPE_B, NPE_C) have the same
        loss function in the first round and only differ when they are being run in
        a sequential scheme.

        Args:
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = posterior_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        self._proposal_roundwise = []
        self.use_non_atomic_loss = False

    @abstractmethod
    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor: ...

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> Self:
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            proposal: The distribution that the parameters $\theta$ were sampled from.
                Pass `None` if the parameters were sampled from the prior. If not
                `None`, it will trigger a different loss-function.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. For single-round SNPE, it is fine to discard invalid
                simulations, but for multi-round SNPE (atomic), discarding invalid
                simulations gives systematically wrong results. If `None`, it will
                be `True` in the first round and `False` in later rounds.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.

        Returns:
            NeuralInference object (returned so that this function is chainable).
        """
        if (
            proposal is None
            or proposal is self._prior
            or (
                isinstance(proposal, RestrictedPrior) and proposal._prior is self._prior
            )
        ):
            # The `_data_round_index` will later be used to infer if one should train
            # with MLE loss or with atomic loss (see, in `train()`:
            # self._round = max(self._data_round_index))
            current_round = 0
        else:
            if not self._data_round_index:
                # This catches a pretty specific case: if, in the first round, one
                # passes data that does not come from the prior.
                current_round = 1
            else:
                current_round = max(self._data_round_index) + 1

        if exclude_invalid_x is None:
            exclude_invalid_x = current_round == 0

        if data_device is None:
            data_device = self._device

        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x, exclude_invalid_x=exclude_invalid_x
        )

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        if (
            type(self).__name__ == "SNPE_C"
            and current_round > 0
            and not self.use_non_atomic_loss
        ):
            nle_nre_apt_msg_on_invalid_x(
                num_nans,
                num_infs,
                exclude_invalid_x,
                "Multiround SNPE-C (atomic)",
            )
        else:
            npe_msg_on_invalid_x(
                num_nans, num_infs, exclude_invalid_x, "Single-round NPE"
            )

        self._check_proposal(proposal)

        self._data_round_index.append(current_round)
        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._proposal_roundwise.append(proposal)

        if self._prior is None or isinstance(self._prior, ImproperEmpirical):
            if proposal is not None:
                raise ValueError(
                    "You had not passed a prior at initialization, but now you "
                    "passed a proposal. If you want to run multi-round SNPE, you have "
                    "to specify a prior (set the `.prior` argument or re-initialize "
                    "the object with a prior distribution). If the samples you passed "
                    "to `append_simulations()` were sampled from the prior, you can "
                    "run single-round inference with "
                    "`append_simulations(..., proposal=None)`."
                )
            theta_prior = self.get_simulations()[0].to(self._device)
            self._prior = ImproperEmpirical(
                theta_prior, ones(theta_prior.shape[0], device=self._device)
            )

        return self

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect
                to the simulations `x` (optional). See Lueckmann, Gonçalves et al.,
                NeurIPS 2017. If `None`, no calibration is used.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss or resume_training, (
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "SNPE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:

            def default_calibration_kernel(x):
                return ones([len(x)], device=self._device)

            calibration_kernel = default_calibration_kernel

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds as of now.
        # SNPE-A can, by construction of the algorithm, only use samples from the last
        # round. SNPE-A is the only algorithm that has an attribute `_ran_final_round`,
        # so this is how we check for whether or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )
        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            theta = reshape_to_sample_batch_event(
                theta.to("cpu"), self._neural_net.input_shape
            )
            x = reshape_to_batch_event(x.to("cpu"), self._neural_net.condition_shape)
            test_posterior_net_for_multi_d_x(self._neural_net, theta, x)

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not resume_training:
            self.optimizer = Adam(list(self._neural_net.parameters()), lr=learning_rate)
            self.epoch, self._val_loss = 0, float("Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            # Train for a single epoch.
            self._neural_net.train()
            train_loss_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch, masks_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )

                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )
                train_loss = torch.mean(train_losses)
                train_loss_sum += train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            self.epoch += 1

            train_loss_average = train_loss_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_loss"].append(train_loss_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch, masks_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation loss for every epoch.
            self._summary["validation_loss"].append(self._val_loss)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def build_posterior(
        self,
        density_estimator: Optional[ConditionalDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct"
        ] = "direct",
        mcmc_method: Literal[
            "slice_np",
            "slice_np_vectorized",
            "hmc_pyro",
            "nuts_pyro",
            "slice_pymc",
            "hmc_pymc",
            "nuts_pymc",
        ] = "slice_np_vectorized",
        vi_method: Literal["rKL", "fKL", "IW", "alpha"] = "rKL",
        direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        importance_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> NeuralPosterior:
        r"""Build posterior from the neural density estimator.

        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`direct` | `mcmc` | `rejection` | `vi` | `importance`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`,
                `slice_np_vectorized`, `hmc_pyro`, `nuts_pyro`, `slice_pymc`,
                `hmc_pymc`, `nuts_pymc`. `slice_np` is a custom
                numpy implementation of slice sampling. `slice_np_vectorized` is
                identical to `slice_np`, but if `num_chains>1`, the chains are
                vectorized for `slice_np_vectorized` whereas they are run sequentially
                for `slice_np`. The samplers ending on `_pyro` are using Pyro, and
                likewise the samplers ending on `_pymc` are using PyMC.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            direct_sampling_parameters: Additional kwargs passed to `DirectPosterior`.
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.
            importance_sampling_parameters: Additional kwargs passed to
                `ImportanceSamplingPosterior`.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """
        return super().build_posterior(
            density_estimator,
            prior,
            sample_with,
            mcmc_method=mcmc_method,
            vi_method=vi_method,
            mcmc_parameters=mcmc_parameters,
            vi_parameters=vi_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            importance_sampling_parameters=importance_sampling_parameters,
            direct_sampling_parameters=direct_sampling_parameters,
        )

    def _get_potential_function(
        self,
        prior: Distribution,
        estimator: ConditionalDensityEstimator,
    ) -> Tuple[PosteriorBasedPotential, TorchTransform]:
        r"""Gets the potential for posterior-based methods.

        It also returns a transformation that can be used to transform the potential
        into unconstrained space.

        The potential is the same as the log-probability of the `posterior_estimator`,
        but it is set to $-\inf$ outside of the prior bounds.

        Args:
            prior: The prior distribution.
            estimator: The neural network modelling the posterior.

        Returns:
            The potential function and a transformation that maps
            to unconstrained space.
        """
        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator=estimator,
            prior=prior,
            x_o=None,
        )
        return potential_fn, theta_transform

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
    ) -> Tensor:
        """Return loss with proposal correction (`round_>0`) or without it (`round_=0`).

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
        """
        if self._round == 0 or force_first_round_loss:
            theta = reshape_to_batch_event(
                theta, event_shape=self._neural_net.input_shape
            )
            x = reshape_to_batch_event(x, event_shape=self._neural_net.condition_shape)
            # Use posterior log prob (without proposal correction) for first round.
            loss = self._neural_net.loss(theta, x)
        else:
            # Currently only works for `DensityEstimator` objects.
            # Must be extended ones other Estimators are implemented. See #966,
            loss = -self._log_prob_proposal_posterior(theta, x, masks, proposal)

        assert_all_finite(loss, "NPE loss")
        return calibration_kernel(x) * loss

    def _check_proposal(self, proposal):
        """
        Check for validity of the provided proposal distribution.

        If the proposal is a `NeuralPosterior`, we check if the default_x is set.
        If the proposal is **not** a `NeuralPosterior`, we warn since it is likely that
        the user simply passed the prior, but this would still trigger atomic loss.
        """
        if proposal is not None:
            check_if_proposal_has_default_x(proposal)

            if isinstance(proposal, RestrictedPrior):
                if proposal._prior is not self._prior:
                    warn(
                        "The proposal you passed is a `RestrictedPrior`, but the "
                        "proposal distribution it uses is not the prior (it can be "
                        "accessed via `RestrictedPrior._prior`). We do not "
                        "recommend to mix the `RestrictedPrior` with multi-round "
                        "SNPE.",
                        stacklevel=2,
                    )
            elif (
                not isinstance(proposal, NeuralPosterior)
                and proposal is not self._prior
            ):
                warn(
                    "The proposal you passed is neither the prior nor a "
                    "`NeuralPosterior` object. If you are an expert user and did so "
                    "for research purposes, this is fine. If not, you might be doing "
                    "something wrong: feel free to create an issue on Github.",
                    stacklevel=2,
                )
        elif self._round > 0:
            raise ValueError(
                "A proposal was passed but no prior was passed at initialisation. When "
                "running multi-round inference, a prior needs to be specified upon "
                "initialisation. Potential fix: setting the `._prior` attribute or "
                "re-initialisation. If the samples passed to `append_simulations()` "
                "were sampled from the prior, single-round inference can be performed "
                "with `append_simulations(..., proprosal=None)`."
            )
