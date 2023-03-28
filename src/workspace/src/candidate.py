from typing import Optional

import gpytorch
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from optuna import logging
from torch.quasirandom import SobolEngine

_logger = logging.get_logger(__name__)
max_cholesky_size = float("inf")


def generate_batch(
    state,
    model,  # GP model
    train_x,  # Evaluated points on the domain [0, 1]^d
    train_y,  # Function values
    batch_size,
    device,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",
):
    """
    Generate a batch of candidate points using either Thompson sampling or expected improvement.
    For more information, see:
    https://github.com/pytorch/botorch/blob/main/tutorials/turbo_1.ipynb
    https://botorch.org/tutorials/turbo_1
    """
    assert train_x.min() >= 0.0 and train_x.max() <= 1.0 and torch.all(torch.isfinite(train_y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * train_x.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = train_x[train_y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = train_x.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=torch.double, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=torch.double, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            candidates = thompson_sampling(X_cand, num_samples=batch_size)
    elif acqf == "ei":
        ei = qExpectedImprovement(model=model, best_f=train_y.max(), sampler=SobolQMCNormalSampler(1024))
        candidates, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    else:
        assert acqf in ("ts", "ei"), "Expected improvement(ei) or Thompson sampling(ts)"
        raise Exception

    return candidates


def turbo_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    state,
    acqf: str,
    device: "torch.device",
) -> "torch.Tensor":
    """
    Args:
        train_x : Tried parameter configuration, in the format [number of trials, number of parameters]
        train_obj : observations, in the form [number of observations, 1]
        train_con : Constraint, None if not used
        bounds : parameter bounds
    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        if train_obj_feas.numel() == 0:
            # TODO(hvy): Do not use 0state as the best observation.
            _logger.warning("No objective values are feasible. Using 0 as the best objective in qEI.")
            best_f = torch.zeros(())
        else:
            best_f = train_obj_feas.max()

        n_constraints = train_con.size(1)
        objective = ConstrainedMCObjective(
            objective=lambda Z: Z[..., 0],
            constraints=[(lambda Z, i=i: Z[..., -n_constraints + i]) for i in range(n_constraints)],
        )
    else:
        train_y = train_obj

        best_f = train_obj.max()

        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)  # normalized

    dim = bounds.size(-1)  # optimied parameters are 5
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
    model = SingleTaskGP(train_x, train_y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):

        fit_gpytorch_model(mll)

        candidates = generate_batch(
            state=state,
            model=model,
            train_x=train_x,
            train_y=train_y,
            batch_size=1,
            n_candidates=256,
            num_restarts=20,
            raw_samples=1024,
            acqf=acqf,
            device=device,
        )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    state.update_state(Y_next=train_obj)

    print(f"{len(train_x)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")

    if len(train_obj) >= 3 and [train_obj[-1]] * 3 == train_obj[-3:]:
        state.restart_triggered = True

    return candidates
