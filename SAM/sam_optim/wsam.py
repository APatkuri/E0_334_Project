"""WSAM: Weighted SAM.

Yue et al. (KDD 2023), "Sharpness-Aware Minimization Revisited: Weighted Sharpness as a
Regularization Term". The objective is

    L_WSAM = L + gamma / (1 - gamma) * (L_SAM - L),     gamma in [0, 1)

gamma = 0 is the vanilla loss, gamma = 0.5 is SAM, gamma > 0.5 weights sharpness more.

Alg. 2 (decoupled, default): the base optimizer only sees the clean gradient g~, and the
sharpness gradient is applied separately, like decoupled weight decay:

    w <- base_step(w, g~) - lr * gamma / (1 - gamma) * (g_adv - g~)

`decouple=False` gives Coupled-WSAM (Alg. 4): the base optimizer sees
g~ + gamma / (1 - gamma) * (g_adv - g~).
"""
import torch

from .base import EPS, SAMBase, add, norm, scale


class WSAM(SAMBase):
    def __init__(self, params, base_optimizer, rho=0.05, gamma=0.9, decouple=True, **kwargs):
        assert 0.0 <= gamma < 1.0, "gamma must be in [0, 1)"
        super().__init__(params, base_optimizer, **kwargs)
        self.rho = rho
        self.gamma = gamma
        self.decouple = decouple

    @torch.no_grad()
    def step(self, closure):
        params = self._params()

        loss, g_clean = self._grad(closure, params)
        backup = self._perturb(params, scale(g_clean, self.rho / (norm(g_clean) + EPS)))
        _, g_adv = self._grad(closure, params)
        self._restore(params, backup)

        k = self.gamma / (1.0 - self.gamma)
        g_sharp = add(g_adv, g_clean, -1.0)  # gradient of the sharpness term L_SAM - L

        if not self.decouple:
            self._base_step(params, add(g_clean, g_sharp, k))
            return loss

        lrs = self._lrs()  # read before the step so the sharpness term uses the same lr
        self._base_step(params, g_clean)
        for p, s, lr in zip(params, g_sharp, lrs):
            p.add_(s, alpha=-lr * k)
        return loss
