import math
import torch
from torch.optim import Optimizer

class BINGO:
    def __init__(self, optimizer: Optimizer):
        self._optim = optimizer
        self.eps = 1e-10

    def zero_grad(self):
        self._optim.zero_grad()
    @torch.no_grad()
    def pc_backward(self, losses):

        assert isinstance(losses, list)
        num_tasks = len(losses)

        params = []
        shapes = []
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
                    shapes.append(p.shape)
        total_numel = sum(p.numel() for p in params)

        grads = []
        for L in losses:
            g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)

            flat = []
            for p, gi in zip(params, g):
                if gi is None:
                    flat.append(torch.zeros(p.numel(), device=p.device))
                else:
                    flat.append(gi.reshape(-1))
            grads.append(torch.cat(flat))
        grads = torch.stack(grads)        

        projected = grads.clone()
        projected_ = grads.clone()

        for i in [0]:  
            g_i = projected[i]

            for j in range(num_tasks):
                if i == j:
                    continue
                g_j = projected[j]
                ip = torch.dot(g_i, g_j)

                if ip < 0:
                    proj = ip / (g_j.dot(g_j) + 1e-10)
                    g_i -= proj * g_j
                    if(g_j.dot(g_j)!=g_i.dot(g_i)):
                        g_i = g_i * torch.sqrt(torch.clamp((g_i.dot(g_i)+g_j.dot(g_j)+2*ip) / (g_i.dot(g_i)+g_j.dot(g_j)-2*ip + self.eps), min=1e-12))
                    else:
                        g_i = g_i * torch.sqrt(torch.clamp((g_i.dot(g_i)+g_j.dot(g_j)+2*ip) / (g_i.dot(g_i)+g_j.dot(g_j)-2*ip + self.eps), max=1))
            projected[i] = g_i
        for i in [1,2]:  
            g_i = projected[i]
            for j in [0]:
                if i == j:
                    continue

                g_j = projected_[j]
                ip = torch.dot(g_i, g_j)
                if ip > 0:
                    proj = ip / (g_j.dot(g_j) + 1e-10)
                    g_i += proj * g_j

            projected[i] = g_i

        merged_grad = projected.mean(dim=0)
        idx = 0
        for p, shape in zip(params, shapes):
            n = p.numel()
            p.grad = merged_grad[idx:idx+n].view(shape).clone()
            idx += n

    def step(self):
        self._optim.step()

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self._optim.param_groups
