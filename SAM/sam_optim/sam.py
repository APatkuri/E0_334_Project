"""SAM: Sharpness-Aware Minimization.

Foret et al. (2021), in the form used by Bahri et al. (2022),
"Sharpness-Aware Minimization Improves Language Model Generalization", Alg. 1:

    w_adv = w + rho * g_M / ||g_M||      g_M: gradient on the ascent micro-batch M
    w     = opt(w, grad L_B(w_adv))      L_B: loss on the full batch B

Bahri et al. use rho = 0.15 and |M| = |B| / 4 (pass `ascent_closure`).
"""
import torch

from .base import EPS, SAMBase, norm, scale


class SAM(SAMBase):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        super().__init__(params, base_optimizer, **kwargs)
        self.rho = rho

    @torch.no_grad()
    def step(self, closure, ascent_closure=None):
        """`ascent_closure` (optional) returns the loss on a micro-batch subset and is
        used only for the ascent step. Returns the loss of the first (clean) forward pass."""
        params = self._params()

        # 1) ascent step
        loss, g = self._grad(ascent_closure or closure, params)
        backup = self._perturb(params, scale(g, self.rho / (norm(g) + EPS)))

        # 2) gradient at the adversarial point, applied at the original weights
        _, g_adv = self._grad(closure, params)
        self._restore(params, backup)
        self._base_step(params, g_adv)
        return loss
