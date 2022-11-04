import torch, math
from typing import Dict


def primal_value(w: torch.tensor, x: torch.tensor, S: torch.tensor, alpha: float, beta: float):
    # Note that the authors of "Accelerated Graph Learning from Smooth Signals" have an
    # extra scaling on the ||w*e||_1: f + g = I{w ≽ 0} + 2 w⊤e + b∥w∥^2  - a*log(Sw)
    # We exclude that factor of 2 here in favor of the original smooth_
    # objective: min_w ||w * e||_1 - a \alpha log(Sw) + b ||w||_F^2 s.t. w \in R^{N*(N-1)/2} >= 0
    primal_vals = smooth_objective(w=w, x=x, S=S, alpha=alpha, beta=beta)
    primal_vals_dict = {'primal_vals/mean': primal_vals.mean(),
                        'primal_vals/stde': torch.std(primal_vals) / math.sqrt(len(primal_vals))
                        }
    return primal_vals_dict, primal_vals


def smooth_objective(w: torch.tensor, x: torch.tensor, S: torch.tensor, alpha: float, beta: float):
    """
    "How to learn a graph from smooth signals" introduced the objective
    min_W ||W * E||_1 - a log(W1) + b ||W||_F^2 s.t. W \in S_+^N, W>=0, diag(W)=0

    or equivelenty in half-vector form:
    min_w ||w * e||_1 - a log(Sw) + b ||w||_F^2 s.t. w \in R^{N*(N-1)/2} >= 0
    """
    #alpha, beta = params['alpha'], params['beta']
    assert w.ndim == x.ndim == 2
    assert alpha >= 0 and beta >= 0, f"alpha {alpha} and beta {beta} must be non-neg"
    bs, m = w.shape
    # assert torch.all(w >= 0), f'w is assumed to non-negative (min = {w.min()}. Indicator func in primal is inf otherwise.'
    og_primal_vals = \
        (w * x).abs().sum(dim=1) \
        - alpha * torch.log10(S.matmul(w.unsqueeze(-1)) + 1e-12).sum(dim=1).squeeze() \
        + beta * (w ** 2).sum(dim=1)
    term0 = (w * x).abs().sum(dim=1)
    term1 = - alpha * torch.log10(S.matmul(w.unsqueeze(-1)) + 1e-12).sum(dim=1).squeeze()
    term2 = beta * (w ** 2).sum(dim=1)
    f0, f1, f2 = term0 / og_primal_vals, term1 / og_primal_vals, term2 / og_primal_vals
    # print(f'\trelative weights in obj: term0: {f0.mean()*100:.3f} %, term1: {f1.mean()*100:.3f} %, term2: {f2.mean()*100:.3f} %')
    return og_primal_vals

