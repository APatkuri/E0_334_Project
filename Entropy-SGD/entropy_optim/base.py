"""Shared machinery for the local-entropy optimizers.

Same interface as the SAM variants: each optimizer wraps a base optimizer (SGD, AdamW, ...)
and exposes

    loss = optimizer.step(closure)

where `closure()` runs a forward pass and returns the loss. The closure must NOT call
`backward()`. Unlike SAM, the closure is called several times per step and should draw a
NEW mini-batch on every call (one per inner Langevin iteration).
"""
import torch

EPS = 1e-12


# ---- helpers on lists of tensors (one tensor per parameter) ----------------

def dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def norm(a):
    return torch.sqrt(dot(a, a))


# ---- base class --------------------------------------------------------------

class EntropyBase(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, **base_kwargs):
        super().__init__(params, dict(base_kwargs))
        self.base_optimizer = base_optimizer(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups  # shared, so LR schedulers work
        self.defaults.update(self.base_optimizer.defaults)
        self.num_steps = 0

    def _params(self):
        return [p for g in self.param_groups for p in g["params"] if p.requires_grad]

    def _grad(self, closure, params):
        """Run the closure and return (loss, gradient list) at the current weights."""
        with torch.enable_grad():
            loss = closure()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
        return loss.detach(), grads

    @torch.no_grad()
    def _restore(self, params, backup):
        for p, b in zip(params, backup):
            p.copy_(b)

    def _base_step(self, params, grads):
        """Hand the final gradient to the base optimizer."""
        for p, g in zip(params, grads):
            p.grad = g
        self.base_optimizer.step()
        self.num_steps += 1

    # The base optimizer holds the real state (momentum, Adam moments, ...).
    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
