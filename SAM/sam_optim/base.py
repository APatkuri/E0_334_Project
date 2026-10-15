"""Shared machinery for all SAM variants.

Every optimizer here wraps a base optimizer (SGD, AdamW, ...) and exposes

    loss = optimizer.step(closure)

where `closure()` runs a forward pass on the current mini-batch and returns the
loss. The closure must NOT call `backward()`: each variant computes the
gradients it needs (clean, perturbed, Hessian-vector products) inside `step`.
"""
import torch

EPS = 1e-12


# ---- helpers on lists of tensors (one tensor per parameter) ----------------

def dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def norm(a):
    return torch.sqrt(dot(a, a))


def add(a, b, alpha=1.0):
    """a + alpha * b"""
    return [x + alpha * y for x, y in zip(a, b)]


def scale(a, c):
    return [x * c for x in a]


def normalize(a):
    return scale(a, 1.0 / (norm(a) + EPS))


# ---- base class --------------------------------------------------------------

class SAMBase(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, **base_kwargs):
        super().__init__(params, dict(base_kwargs))
        self.base_optimizer = base_optimizer(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups  # shared, so LR schedulers work
        self.defaults.update(self.base_optimizer.defaults)
        self.num_steps = 0

    def _params(self):
        return [p for g in self.param_groups for p in g["params"] if p.requires_grad]

    def _lrs(self):
        return [g["lr"] for g in self.param_groups for p in g["params"] if p.requires_grad]

    def _grad(self, closure, params):
        """Run the closure and return (loss, gradient list) at the current weights."""
        with torch.enable_grad():
            loss = closure()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
        return loss.detach(), grads

    @torch.no_grad()
    def _perturb(self, params, e):
        """w <- w + e. Returns a backup of the original weights."""
        backup = [p.detach().clone() for p in params]
        for p, e_i in zip(params, e):
            p.add_(e_i)
        return backup

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

    def _top_eigenvector(self, closure, params, iters):
        """Power iteration with Hessian-vector products (Eigen-SAM, Alg. 1).

        Returns (unit top eigenvector as a list of tensors, eigenvalue estimate).
        """
        with torch.enable_grad():
            loss = closure()
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            # Gradients that do not depend on the weights contribute zero curvature.
            idx = [i for i, g in enumerate(grads) if g is not None and g.requires_grad]
            outputs = [grads[i] for i in idx]

            v = normalize([torch.randn_like(p) for p in params])
            eigenvalue = torch.zeros(())
            for _ in range(iters):
                hv = torch.autograd.grad(outputs, params, grad_outputs=[v[i] for i in idx],
                                         retain_graph=True, allow_unused=True)
                hv = [torch.zeros_like(p) if h is None else h for p, h in zip(params, hv)]
                eigenvalue = dot(v, hv)  # Rayleigh quotient, v is unit
                v = normalize(hv)
        return [x.detach() for x in v], eigenvalue.item()

    # The base optimizer holds the real state (momentum, Adam moments, ...).
    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
