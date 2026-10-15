"""Eigen-SAM: explicit top-eigenvalue regularization.

Luo et al. (2025), "Explicit Eigenvalue Regularization Improves Sharpness-Aware Minimization",
Alg. 1-2 and Eq. 7-8. Every `eig_every` (p) steps, the top Hessian eigenvector v is estimated
with `power_iters` (q) steps of power iteration. The SAM perturbation is then pushed toward v:

    v      = v_par + v_perp                       (decomposed w.r.t. the gradient g)
    eps    = g / ||g|| + alpha * sign(<g, v>) * v_perp
    w_adv  = w + rho * eps
    update = grad f(w_adv)

The paper uses p = 100, q = 5, alpha = 0.2. alpha = 0 recovers SAM.
"""
import torch

from .base import SAMBase, add, dot, normalize, scale


class EigenSAM(SAMBase):
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.2,
                 eig_every=100, power_iters=5, **kwargs):
        super().__init__(params, base_optimizer, **kwargs)
        self.rho = rho
        self.alpha = alpha
        self.eig_every = eig_every
        self.power_iters = power_iters
        self.v = None
        self.top_eigenvalue = None  # latest estimate, handy for logging sharpness

    @torch.no_grad()
    def step(self, closure):
        params = self._params()

        if self.num_steps % self.eig_every == 0:
            self.v, self.top_eigenvalue = self._top_eigenvector(closure, params, self.power_iters)

        loss, g = self._grad(closure, params)
        g_unit = normalize(g)
        c = dot(self.v, g_unit)
        v_perp = add(self.v, g_unit, -c)
        sign = 1.0 if c >= 0 else -1.0  # pick the eigenvector sign closer to the gradient
        direction = add(g_unit, v_perp, self.alpha * sign)

        backup = self._perturb(params, scale(direction, self.rho))
        _, g_adv = self._grad(closure, params)
        self._restore(params, backup)

        self._base_step(params, g_adv)
        return loss
