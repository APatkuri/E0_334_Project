"""Entropy-SGD.

Chaudhari et al. (ICLR 2017), "Entropy-SGD: Biasing Gradient Descent Into Wide Valleys".
It minimizes the negative local entropy

    -F(x, gamma) = -log ∫ exp(-f(x') - gamma/2 ||x - x'||^2) dx'

whose gradient is gamma (x - <x'>) (Eq. 7), where <x'> is the mean of the Gibbs distribution
focused around x (Eq. 8). Alg. 1 estimates <x'> with L steps of SGLD, each on a new mini-batch:

    x', mu <- x
    repeat L times:
        dx' = grad f(x') - gamma (x - x')
        x'  = x' - eta' dx' + sqrt(eta') eps N(0, I)
        mu  = (1 - alpha) mu + alpha x'
    x <- opt(x, x - mu)

The outer gradient is x - mu, as used in practice (Sec. 4.3). `scale_by_gamma=True` gives
gamma (x - mu), as in Alg. 1 line 7. Scoping (Sec. 4.3): gamma_t = gamma0 (1 + gamma1)^t
for the t-th update, where `gamma` is gamma0 and `gamma_growth` is gamma1.

The SGLD steps use `sgld_optimizer` (SGD by default; Sec. 4.2 also uses Nesterov momentum or
Adam there). It is rebuilt at every outer step, so its state starts fresh each time. Using Adam
as `base_optimizer` gives Entropy-Adam (Sec. 5.4.2).
"""
import math

import torch

from .base import EPS, EntropyBase, norm


class EntropySGD(EntropyBase):
    def __init__(self, params, base_optimizer, sgld_steps=5, gamma=0.03, gamma_growth=0.0,
                 sgld_lr=0.1, noise=1e-4, alpha=0.75, sgld_optimizer=torch.optim.SGD,
                 sgld_kwargs=None, scale_by_gamma=False, **kwargs):
        assert sgld_steps >= 1, "need at least one SGLD step"
        super().__init__(params, base_optimizer, **kwargs)
        self.sgld_steps = sgld_steps
        self.gamma0 = gamma
        self.gamma_growth = gamma_growth
        self.sgld_lr = sgld_lr
        self.noise = noise
        self.alpha = alpha
        self.sgld_optimizer = sgld_optimizer
        self.sgld_kwargs = sgld_kwargs or {}
        self.scale_by_gamma = scale_by_gamma
        self.gamma = gamma        # scope used in the latest step
        self.grad_ratio = None    # ||outer gradient|| / ||grad f(x)||, for tuning gamma (Sec. 4.2)

    @torch.no_grad()
    def step(self, closure):
        """`closure()` is called `sgld_steps` times and should draw a new mini-batch each time.
        Returns the loss of the first call, which is the loss at the current weights."""
        params = self._params()
        self.gamma = gamma = self.gamma0 * (1.0 + self.gamma_growth) ** self.num_steps
        x = [p.detach().clone() for p in params]
        mu = [p.detach().clone() for p in params]
        sgld = self.sgld_optimizer(params, lr=self.sgld_lr, **self.sgld_kwargs)
        noise_std = math.sqrt(self.sgld_lr) * self.noise

        # 1) SGLD around x: the weights themselves play the role of x'
        loss = g0_norm = None
        for _ in range(self.sgld_steps):
            f, g = self._grad(closure, params)
            if loss is None:
                loss, g0_norm = f, norm(g)
            for p, g_i, x_i in zip(params, g, x):
                p.grad = g_i.sub_(x_i - p, alpha=gamma)  # grad f(x') - gamma (x - x')
            sgld.step()
            for p, m in zip(params, mu):
                if noise_std > 0:
                    p.add_(torch.randn_like(p), alpha=noise_std)
                m.mul_(1.0 - self.alpha).add_(p, alpha=self.alpha)

        # 2) local-entropy gradient, applied at the original weights
        self._restore(params, x)
        c = gamma if self.scale_by_gamma else 1.0
        d = [(x_i - m) * c for x_i, m in zip(x, mu)]
        self.grad_ratio = (norm(d) / (g0_norm + EPS)).item()
        self._base_step(params, d)
        return loss
