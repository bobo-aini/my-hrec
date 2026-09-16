"""Numerically guarded Poincare operations used by HREC (curvature -c)."""
import math
import torch


class PoincareBall:
    def __init__(self, c=1.0, eps=1e-5):
        if c <= 0:
            raise ValueError("c must be positive")
        self.c, self.eps = float(c), eps

    def project(self, x):
        radius = (1 - self.eps) / math.sqrt(self.c)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        return x * (radius / norm).clamp_max(1)

    def exp_map_zero(self, v):
        norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        scale = math.sqrt(self.c) * norm
        return self.project(torch.tanh(scale) * v / scale)

    def log_map_zero(self, x):
        x = self.project(x)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        scale = math.sqrt(self.c) * norm
        return torch.atanh(scale.clamp_max(1 - self.eps)) * x / scale

    def mobius_add(self, x, y):
        c = self.c
        x, y = self.project(x), self.project(y)
        xx = x.square().sum(-1, keepdim=True)
        yy = y.square().sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        numerator = (1 + 2*c*xy + c*yy)*x + (1-c*xx)*y
        denominator = 1 + 2*c*xy + c*c*xx*yy
        return numerator / denominator.clamp_min(1e-15)

    def distance(self, x, y):
        displacement = self.mobius_add(-x, y)
        radius = math.sqrt(self.c) * displacement.norm(dim=-1)
        return 2 / math.sqrt(self.c) * torch.atanh(radius.clamp_max(1-self.eps))
