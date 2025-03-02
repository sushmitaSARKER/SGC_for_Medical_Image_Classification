import torch
import numpy as np
from scipy import integrate
from models.utils import from_flattened_numpy, to_flattened_numpy

def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""
    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
    return div_fn

def ode_likelihood(x, y, score_model, sde_fn, device='cuda', rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5, hutchinson_type='Gaussian'):
    """
    Compute the unbiased log-likelihood estimate of a given data point using a probability flow ODE.

    Args:
        x: Input data.
        y: Additional input (e.g., labels or conditions).
        score_model: A score model.
        sde_fn: A function representing the forward SDE.
        device: Device to use ('cuda' or 'cpu').
        rtol: Relative tolerance for the ODE solver.
        atol: Absolute tolerance for the ODE solver.
        method: Algorithm for the ODE solver (e.g., 'RK45').
        eps: Integration limit for numerical stability.
        hutchinson_type: Type of noise for Hutchinson-Skilling trace estimator ('Gaussian' or 'Rademacher').

    Returns:
        A function that computes the log-likelihood, latent code, and number of function evaluations.
    """

    def drift_fn(score_model, x, t, y):
        """The drift function of the reverse-time SDE."""
        rsde = sde_fn.reverse(score_model, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(score_model, x, t, noise, y):
        """Compute the divergence of the drift function."""
        return get_div_fn(lambda xx, tt: drift_fn(score_model, xx, tt, y))(x, t, noise)

    def likelihood_fn(model, data):
        """
        Compute an unbiased estimate of the log-likelihood in bits/dim.

        Args:
            model: A score model.
            data: A PyTorch tensor.

        Returns:
            z: Latent representation of `data` under the probability flow ODE.
            nfe: Number of function evaluations used by the ODE solver.
            logp: Log-likelihood of the data.
        """
        with torch.no_grad():
            shape = data.shape

            # Generate Hutchinson-Skilling noise
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} not supported.")

            def ode_func(t, x):
                """ODE function for the probability flow."""
                sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = to_flattened_numpy(drift_fn(model, sample.reshape(shape), vec_t, y))
                logp_grad = to_flattened_numpy(div_fn(model, sample.reshape(shape), vec_t, epsilon, y))
                return np.concatenate([drift, logp_grad], axis=0)

            # Solve the ODE
            init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, sde_fn.T), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]

            # Extract results
            z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
            delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
            prior_logp = sde_fn.prior_logp(z)
            logp = prior_logp + delta_logp
            return z, nfe, logp
        
    return likelihood_fn