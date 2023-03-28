import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy
import torch
from botorch.utils.sampling import manual_seed
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.integration.botorch import _get_default_candidates_func
from optuna.samplers import BaseSampler, IntersectionSearchSpace, QMCSampler
from optuna.samplers._base import _CONSTRAINTS_KEY, _process_constraints_after_trial
from optuna.study import Study, StudyDirection
from optuna.trial import FrozenTrial, TrialState


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))

    def update_state(self, Y_next):
        if max(Y_next) > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, max(Y_next).item())
        if self.length < self.length_min:
            self.restart_triggered = True


class BoTorchSampler(BaseSampler):
    """A sampler that uses BoTorch, a Bayesian optimization library built on top of PyTorch.

    This sampler allows using BoTorch's optimization algorithms from Optuna to suggest parameter
    configurations. Parameters are transformed to continuous space and passed to BoTorch, and then
    transformed back to Optuna's representations. Categorical parameters are one-hot encoded.

    .. seealso::
        See an `example <https://github.com/optuna/optuna-examples/blob/main/multi_objective/
        botorch_simple.py>`_ how to use the sampler.

    .. seealso::
        See the `BoTorch <https://botorch.org/>`_ homepage for details and for how to implement
        your own ``candidates_func``.

    .. note::
        An instance of this sampler *should not be used with different studies* when used with
        constraints. Instead, a new instance should be created for each new study. The reason for
        this is that the sampler is stateful keeping all the computed constraints.

    Args:
        candidates_func:
            An optional function that suggests the next candidates. It must take the training
            data, the objectives, the constraints, the search space bounds and return the next
            candidates. The arguments are of type ``torch.Tensor``. The return value must be a
            ``torch.Tensor``. However, if ``constraints_func`` is omitted, constraints will be
            :obj:`None`. For any constraints that failed to compute, the tensor will contain
            NaN.

            If omitted, it is determined automatically based on the number of objectives. If the
            number of objectives is one, Quasi MC-based batch Expected Improvement (qEI) is used.
            If the number of objectives is either two or three, Quasi MC-based
            batch Expected Hypervolume Improvement (qEHVI) is used. Otherwise, for larger number
            of objectives, the faster Quasi MC-based extended ParEGO (qParEGO) is used.

            The function should assume *maximization* of the objective.

            .. seealso::
                See :func:`optuna.integration.botorch.qei_candidates_func` for an example.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraint is violated. A value equal to or smaller than 0 is considered feasible.

            If omitted, no constraints will be passed to ``candidates_func`` nor taken into
            account during suggestion.
        n_startup_trials:
            Number of initial trials, that is the number of trials to resort to independent
            sampling.
        independent_sampler:
            An independent sampler to use for the initial trials and for parameters that are
            conditional.
        seed:
            Seed for random number generator.
        device:
            A ``torch.device`` to store input and output data of BoTorch. Please set a CUDA device
            if you fasten sampling.
    """

    def __init__(
        self,
        *,
        candidates_func: Optional[
            Callable[
                [
                    "torch.Tensor",
                    "torch.Tensor",
                    Optional["torch.Tensor"],
                    "torch.Tensor",
                ],
                "torch.Tensor",
            ]
        ] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        n_startup_trials: int = 10,
        independent_sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
        device: Optional["torch.device"] = None,
        acqf="ts",
    ):
        self._candidates_func = candidates_func
        self._constraints_func = constraints_func

        self._n_startup_trials = n_startup_trials
        self._seed = seed

        self._study_id: Optional[int] = None
        self._search_space = IntersectionSearchSpace()
        self._device = device or torch.device("cpu")

        # custom
        self.turbo_dim = None
        self.acqf = acqf
        self._independent_sampler = independent_sampler or QMCSampler(seed=seed)
        self.restart_count = 0
        self.restart_pos = 0
        print(f"Sampling method : {self.acqf}")

    def restart(self, n_trials: int):
        self._search_space = IntersectionSearchSpace()
        self.turbo_state = TurboState(dim=self.turbo_dim, batch_size=1)
        self.restart_count += 1
        self.restart_pos += n_trials
        print(f"\n\n Turbo Restart : {self.restart_count}")

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        search_space: Dict[str, BaseDistribution] = OrderedDict()
        for name, distribution in self._search_space.calculate(study, ordered_dict=True).items():
            if distribution.single():
                # built-in `candidates_func` cannot handle distributions that contain just a
                # single value, so we skip them. Note that the parameter values for such
                # distributions are sampled in `Trial`.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, OrderedDict)
        if len(search_space) == 0:
            return {}

        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        trials = trials[self.restart_pos :]  # restart時のデータから使用する

        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives = len(study.directions)
        values: Union[numpy.ndarray, torch.Tensor] = numpy.empty((n_trials, n_objectives), dtype=numpy.float64)
        params: Union[numpy.ndarray, torch.Tensor]
        con: Optional[Union[numpy.ndarray, torch.Tensor]] = None
        bounds: Union[numpy.ndarray, torch.Tensor] = trans.bounds
        params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)

        for trial_idx, trial in enumerate(trials):
            params[trial_idx] = trans.transform(trial.params)
            assert len(study.directions) == len(trial.values)

            for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                assert value is not None
                if direction == StudyDirection.MINIMIZE:  # BoTorch always assumes maximization.
                    value *= -1
                values[trial_idx, obj_idx] = value

            if self._constraints_func is not None:
                constraints = study._storage.get_trial_system_attrs(trial._trial_id).get(_CONSTRAINTS_KEY)
                if constraints is not None:
                    n_constraints = len(constraints)

                    if con is None:
                        con = numpy.full((n_trials, n_constraints), numpy.nan, dtype=numpy.float64)
                    elif n_constraints != con.shape[1]:
                        raise RuntimeError(f"Expected {con.shape[1]} constraints but received {n_constraints}.")

                    con[trial_idx] = constraints

        if self._constraints_func is not None:
            if con is None:
                warnings.warn(
                    "`constraints_func` was given but no call to it correctly computed "
                    "constraints. Constraints passed to `candidates_func` will be `None`."
                )
            elif numpy.isnan(con).any():
                warnings.warn(
                    "`constraints_func` was given but some calls to it did not correctly compute "
                    "constraints. Constraints passed to `candidates_func` will contain NaN."
                )

        values = torch.from_numpy(values).to(self._device)
        params = torch.from_numpy(params).to(self._device)
        if con is not None:
            con = torch.from_numpy(con).to(self._device)
        bounds = torch.from_numpy(bounds).to(self._device)

        if con is not None:
            if con.dim() == 1:
                con.unsqueeze_(-1)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(n_objectives=n_objectives)

        if not self.turbo_dim:
            # Turbo stateの次元はここで定義
            self.turbo_dim = bounds.size(-1)
            self.turbo_state = TurboState(dim=self.turbo_dim, batch_size=1)

        with manual_seed(self._seed):
            # `manual_seed` makes the default candidates functions reproducible.
            # `SobolQMCNormalSampler`'s constructor has a `seed` argument, but its behavior is
            # deterministic when the BoTorch's seed is fixed.
            candidates = self._candidates_func(params, values, con, bounds, self.turbo_state, self.acqf, self._device)

            if self._seed is not None:
                self._seed += 1

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            # Batch size is one. Get rid of the batch dimension.
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )

        if self.turbo_state.restart_triggered:
            # Restart
            # https://tech.preferred.jp/ja/blog/bbo-challenge-2020/#sec-restart
            # https://github.com/optuna/bboc-optuna-developers/blob/main/submissions/mksturbo/optimizer.py
            self.restart(n_trials)

        return trans.untransform(candidates.cpu().numpy())

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(2 ** 60)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._independent_sampler.after_trial(study, trial, state, values)
