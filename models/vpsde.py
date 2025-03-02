import numpy as np
import torch

class VPSDE:
    """Variance Preserving Stochastic Differential Equation (SDE)."""

    def __init__(self, beta_min=0.1, beta_max=20):
        """
        Construct a Variance Preserving SDE.

        Args:
            beta_min (float): Value of beta(0). Default is 0.1.
            beta_max (float): Value of beta(1). Default is 20.
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    @property
    def T(self):
        """End time of the SDE."""
        return 1.0

    def sde(self, x, t):
        """
        Compute the drift and diffusion coefficients of the SDE.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            drift (torch.Tensor): Drift coefficient.
            diffusion (torch.Tensor): Diffusion coefficient.
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """
        Compute the mean and standard deviation of the marginal distribution at time `t`.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            mean (torch.Tensor): Mean of the marginal distribution.
            std (torch.Tensor): Standard deviation of the marginal distribution.
        """
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_logp(self, z):
        """
        Compute the log probability of the prior distribution (standard Gaussian).

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            logps (torch.Tensor): Log probability of the prior distribution.
        """
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def reverse(self, score_fn, probability_flow=True):
        """
        Create the reverse-time SDE/ODE.

        Args:
            score_fn: A time-dependent score-based model that takes `x` and `t` and returns the score.
            probability_flow (bool): If `True`, create the reverse-time ODE for probability flow sampling.
                                     Default is `True`.

        Returns:
            RSDE: A class representing the reverse-time SDE/ODE.
        """
        T = self.T
        sde_fn = self.sde

        class RSDE(self.__class__):
            def __init__(self):
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """
                Compute the drift and diffusion functions for the reverse SDE/ODE.

                Args:
                    x (torch.Tensor): Input tensor.
                    t (torch.Tensor): Time tensor.

                Returns:
                    drift (torch.Tensor): Drift coefficient.
                    diffusion (torch.Tensor): Diffusion coefficient.
                """
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.0)
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion
        return RSDE()