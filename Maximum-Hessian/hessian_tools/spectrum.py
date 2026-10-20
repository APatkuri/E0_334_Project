"""Curvature of the training loss from Hessian-vector products (Pearlmutter, 1994).

Kaur et al. (2023), "On the Maximum Hessian Eigenvalue and Generalization", measure sharpness as
lambda_max, the largest eigenvalue of the Hessian of the training loss. They compute it on a subset
of the training set with dropout turned off. The functions here work for any model and take

    params   the parameters to differentiate (list of tensors)
    loss_fn  loss_fn(batch) -> mean loss on that batch (scalar tensor)
    batches  a list of batches, ideally of equal size; the loss is their average

The Hessian is never formed. Each Hessian-vector product costs about two backward passes per batch.
"""
import contextlib
import math

import torch
import torch.nn as nn

EPS = 1e-12


# ---- helpers on lists of tensors (one tensor per parameter) ----------------

def dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def normalize(a):
    n = torch.sqrt(dot(a, a))
    return [x / (n + EPS) for x in a]


# ---- Hessian-vector product --------------------------------------------------

def hvp(params, loss_fn, batches, v):
    """H v, where H is the Hessian of the average loss over `batches`."""
    out = [torch.zeros_like(p) for p in params]
    for batch in batches:
        with torch.enable_grad():
            grads = torch.autograd.grad(loss_fn(batch), params, create_graph=True, allow_unused=True)
            # Gradients that do not depend on the weights contribute zero curvature.
            idx = [i for i, g in enumerate(grads) if g is not None and g.requires_grad]
            hv = torch.autograd.grad([grads[i] for i in idx], params,
                                     grad_outputs=[v[i] for i in idx], allow_unused=True)
        for o, h in zip(out, hv):
            if h is not None:
                o.add_(h.detach())
    return [o / len(batches) for o in out]


def _power_iteration(matvec, like, iters, tol, seed):
    g = torch.Generator().manual_seed(seed)
    v = normalize([torch.randn(p.shape, generator=g).to(p) for p in like])
    lam = None
    for _ in range(iters):
        hv = matvec(v)
        new = dot(v, hv).item()  # Rayleigh quotient, v is unit
        v = normalize(hv)
        converged = lam is not None and abs(new - lam) <= tol * max(abs(new), EPS)
        lam = new
        if converged:
            break
    return lam, v


def _tridiag_eigh(a, b):
    """Eigenvalues and eigenvectors of the symmetric tridiagonal matrix with diagonal a, off-diagonal b."""
    T = torch.diag(torch.tensor(a, dtype=torch.float64))
    if b:
        off = torch.tensor(b, dtype=torch.float64)
        T += torch.diag(off, 1) + torch.diag(off, -1)
    return torch.linalg.eigh(T)


def _lanczos_max(matvec, like, iters, tol, seed, min_iters=20):
    """Largest (algebraic) eigenvalue and its unit Ritz vector by plain Lanczos.

    Unlike power iteration on H - shift*I, large negative eigenvalues do not slow it down: Lanczos
    resolves those outliers in its first few steps. Without reorthogonalization they may reappear
    as ghost copies, which does not affect the largest Ritz value. The Ritz vector comes from a
    second pass that replays the recurrence with the stored coefficients, so only a few vectors are
    ever kept in memory.
    """
    gen = torch.Generator().manual_seed(seed)
    q = normalize([torch.randn(p.shape, generator=gen).to(p) for p in like])
    q_prev, a, b, theta = None, [], [], None
    for j in range(iters):
        w = matvec(q)
        a.append(dot(w, q).item())
        w = [wi - a[-1] * qi for wi, qi in zip(w, q)]
        if q_prev is not None:
            w = [wi - b[-1] * pi for wi, pi in zip(w, q_prev)]
        b.append(torch.sqrt(dot(w, w)).item())
        new = _tridiag_eigh(a, b[:-1])[0][-1].item()
        converged = theta is not None and j + 1 >= min_iters and abs(new - theta) <= tol * max(abs(new), EPS)
        theta = new
        if converged or b[-1] < EPS:
            break
        q_prev, q = q, [x / b[-1] for x in w]
    y = _tridiag_eigh(a, b[:-1])[1][:, -1].tolist()
    gen = torch.Generator().manual_seed(seed)  # replay from the same start vector
    q = normalize([torch.randn(p.shape, generator=gen).to(p) for p in like])
    v, q_prev = [y[0] * x for x in q], None
    for j in range(len(a) - 1):
        w = [wi - a[j] * qi for wi, qi in zip(matvec(q), q)]
        if q_prev is not None:
            w = [wi - b[j - 1] * pi for wi, pi in zip(w, q_prev)]
        q_prev, q = q, [x / b[j] for x in w]
        v = [vi + y[j + 1] * qi for vi, qi in zip(v, q)]
    return theta, normalize(v)


def top_eigenvalue(params, loss_fn, batches, iters=20, tol=1e-3, seed=0, shift_iters=None):
    """lambda_max of the Hessian. Returns (lambda_max, unit eigenvector).

    Power iteration finds the eigenvalue of largest magnitude. If that one is negative, Lanczos
    (up to `shift_iters`, default 5 * iters, steps) finds the largest one instead. A shifted power
    iteration on H - lam*I fails here: fine-tuned BERT has negative eigenvalues of 1e4 and more, and
    relative to such a shift all eigenvalues are nearly equal, so it returns about 0.
    """
    matvec = lambda u: hvp(params, loss_fn, batches, u)
    lam, v = _power_iteration(matvec, params, iters, tol, seed)
    if lam < 0:
        lam, v = _lanczos_max(matvec, params, shift_iters or 5 * iters, tol, seed + 1)
    return lam, v


def hessian_trace(params, loss_fn, batches, samples=50, seed=0):
    """Hutchinson's estimator tr(H) = E[z^T H z], z with i.i.d. +-1 entries.

    Returns (estimate, standard error).
    """
    g = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(samples):
        z = [(torch.randint(0, 2, p.shape, generator=g) * 2 - 1).to(p) for p in params]
        values.append(dot(z, hvp(params, loss_fn, batches, z)).item())
    mean = sum(values) / samples
    var = sum((x - mean) ** 2 for x in values) / max(samples - 1, 1)
    return mean, math.sqrt(var / samples)


@torch.no_grad()
def lmax_between(params, loss_fn, batches, start, end, points=8, **kwargs):
    """Largest lambda_max over `points` evenly spaced weights between two iterates.

    For BN networks trained with a small learning rate, lambda_max changes a lot within one step,
    so Kaur et al. follow Cohen et al. (2021) and take the max over 8 points between the final
    iterate and one extra training step. The weights are left at `end`.
    """
    best = -math.inf
    for k in range(points):
        t = k / (points - 1)
        for p, a, b in zip(params, start, end):
            p.copy_(a + t * (b - a))
        best = max(best, top_eigenvalue(params, loss_fn, batches, **kwargs)[0])
    for p, b in zip(params, end):
        p.copy_(b)
    return best


# ---- model mode ----------------------------------------------------------------

@contextlib.contextmanager
def hessian_mode(model):
    """Evaluate the training objective without dropout, and leave the model as it was.

    Without BatchNorm this is simply eval mode. With BatchNorm the training loss uses batch
    statistics, so BN stays in train mode (dropout layers are switched off one by one), and its
    running statistics are restored on exit.
    """
    was_training = model.training
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [{k: b.clone() for k, b in m.named_buffers()} for m in bns]
    try:
        if not bns:
            model.eval()
        else:
            model.train()
            for m in model.modules():
                if isinstance(m, nn.modules.dropout._DropoutNd):
                    m.eval()
        yield model
    finally:
        with torch.no_grad():
            for m, buffers in zip(bns, saved):
                for k, b in m.named_buffers():
                    b.copy_(buffers[k])
        model.train(was_training)
