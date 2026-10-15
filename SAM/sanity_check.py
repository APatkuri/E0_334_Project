"""Quick CPU check that every optimizer runs and reduces the loss (takes a few seconds)."""
import torch
import torch.nn as nn

from sam_optim import GSAM, SAM, WSAM, EigenSAM, XSAM


def check_power_iteration():
    # f(w) = 0.5 * w^T diag(a) w  ->  Hessian = diag(a), top eigenpair = (10, e_0)
    a = torch.tensor([10.0, 3.0, 1.0, 0.5])
    w = nn.Parameter(torch.randn(4))
    opt = SAM([w], torch.optim.SGD, lr=0.1)
    v, lam = opt._top_eigenvector(lambda: 0.5 * (a * w * w).sum(), [w], iters=30)
    assert abs(lam - 10.0) < 1e-3, lam
    assert abs(abs(v[0][0].item()) - 1.0) < 1e-3, v
    print(f"power iteration ok: lambda_max = {lam:.4f}")


def train(make_opt, steps=150):
    torch.manual_seed(0)
    x = torch.randn(512, 20)
    y = (x @ torch.randn(20) > 0).long()
    model = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 2))
    opt = make_opt(model.parameters())
    loss_fn = lambda: nn.functional.cross_entropy(model(x), y)

    first = loss_fn().item()
    for _ in range(steps):
        if isinstance(opt, torch.optim.SGD):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        else:
            opt.step(loss_fn)
    return first, loss_fn().item()


if __name__ == "__main__":
    check_power_iteration()

    SGD, kw = torch.optim.SGD, dict(lr=0.1, momentum=0.9)
    optimizers = {
        "SGD": lambda p: SGD(p, **kw),
        "SAM": lambda p: SAM(p, SGD, rho=0.05, **kw),
        "ASAM": lambda p: SAM(p, SGD, rho=0.5, adaptive=True, **kw),
        "GSAM": lambda p: GSAM(p, SGD, rho=0.05, alpha=0.1, **kw),
        "WSAM": lambda p: WSAM(p, SGD, rho=0.05, gamma=0.9, **kw),
        "WSAM-coupled": lambda p: WSAM(p, SGD, rho=0.05, gamma=0.9, decouple=False, **kw),
        "EigenSAM": lambda p: EigenSAM(p, SGD, rho=0.05, alpha=0.2, eig_every=10, **kw),
        "XSAM": lambda p: XSAM(p, SGD, rho=0.05, alpha=0.2, eig_every=10, **kw),
    }
    for name, make_opt in optimizers.items():
        first, last = train(make_opt)
        status = "ok" if last < first else "FAIL"
        print(f"{name:14s} loss {first:.4f} -> {last:.4f}  [{status}]")
