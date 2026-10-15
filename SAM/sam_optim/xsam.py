"""X-SAM: SAM with dominant-eigenvector gradient correction.

Duan et al. (2026), "X-SAM: Boosting Sharpness-Aware Minimization with Dominant-Eigenvector
Gradient Correction", Alg. 2. Every `eig_every` (p) steps, the top Hessian eigenvector v is
estimated with power iteration (same as Eigen-SAM). Then the SAM gradient is corrected:

    w_adv  = w + rho * g / ||g||
    h      = grad f(w_adv) / ||grad f(w_adv)||        (unit gradient, Eq. 9)
    update = h - alpha * <h, v> v                     (= (I - alpha v v^T) h, Thm. 5.1)

This shrinks the update along the sharpest direction. Alg. 2 writes the correction as
alpha * sign(<h, v>) * h_par. Because h_par = <h, v> v does not depend on the sign of v, we
use the projector form from Thm. 5.1, which always shrinks that component.

The paper normalizes the gradient (Eq. 9), so with SGD every step has length lr.
Set `normalize_grad=False` to keep the gradient's scale.
The paper uses alpha = 0.2 and the same rho as SAM.
"""
import torch

from .base import EPS, SAMBase, add, dot, norm, normalize, scale


class XSAM(SAMBase):
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.2,
                 eig_every=100, power_iters=5, normalize_grad=True, **kwargs):
        super().__init__(params, base_optimizer, **kwargs)
        self.rho = rho
        self.alpha = alpha
        self.eig_every = eig_every
        self.power_iters = power_iters
        self.normalize_grad = normalize_grad
        self.v = None
        self.top_eigenvalue = None

    @torch.no_grad()
    def step(self, closure):
        params = self._params()

        if self.num_steps % self.eig_every == 0:
            self.v, self.top_eigenvalue = self._top_eigenvector(closure, params, self.power_iters)

        loss, g = self._grad(closure, params)
        backup = self._perturb(params, scale(g, self.rho / (norm(g) + EPS)))
        _, g_adv = self._grad(closure, params)
        self._restore(params, backup)

        h = normalize(g_adv) if self.normalize_grad else g_adv
        corrected = add(h, self.v, -self.alpha * dot(h, self.v))

        self._base_step(params, corrected)
        return loss
