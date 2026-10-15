"""GSAM: Surrogate Gap Guided SAM.

Zhuang et al. (ICLR 2022), "Surrogate Gap Minimization Improves Sharpness-Aware Training", Alg. 1:

    rho_t   = rho_min + (rho_max - rho_min) * (lr - lr_min) / (lr_max - lr_min)
    w_adv   = w + rho_t * g / ||g||
    g_p     = grad f(w_adv)
    g       = g_par + g_perp           (decomposed w.r.t. g_p)
    update  = g_p - alpha * g_perp     (minimizes the perturbed loss and the surrogate gap)

If `rho_min` or `lr_max` is None, rho stays fixed at `rho`.
"""
import torch

from .base import EPS, SAMBase, add, dot, norm, scale


class GSAM(SAMBase):
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.01,
                 rho_min=None, lr_max=None, lr_min=0.0, **kwargs):
        super().__init__(params, base_optimizer, **kwargs)
        self.rho = rho  # rho_max when the schedule is used
        self.alpha = alpha
        self.rho_min, self.lr_max, self.lr_min = rho_min, lr_max, lr_min

    def current_rho(self):
        if self.rho_min is None or self.lr_max is None or self.lr_max == self.lr_min:
            return self.rho
        lr = self.param_groups[0]["lr"]
        return self.rho_min + (self.rho - self.rho_min) * (lr - self.lr_min) / (self.lr_max - self.lr_min)

    @torch.no_grad()
    def step(self, closure):
        params = self._params()

        loss, g = self._grad(closure, params)
        backup = self._perturb(params, scale(g, self.current_rho() / (norm(g) + EPS)))
        _, g_p = self._grad(closure, params)
        self._restore(params, backup)

        # Component of the clean gradient orthogonal to the perturbed gradient.
        g_par = scale(g_p, dot(g, g_p) / (dot(g_p, g_p) + EPS))
        g_perp = add(g, g_par, -1.0)

        self._base_step(params, add(g_p, g_perp, -self.alpha))
        return loss
