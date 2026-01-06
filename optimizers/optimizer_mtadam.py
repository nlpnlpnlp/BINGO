import torch.nn as nn
import torch
Tensor = torch.Tensor
import math
from torch.optim.optimizer import Optimizer

class MTAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MTAdam, self).__init__(params, defaults)

        self.training_step = 0

    def __setstate__(self, state):
        super(MTAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, loss_array, ranks, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.update_weights(loss_array, ranks)
        return loss

    def update_weights(self, loss_array, ranks):
        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                beta1, beta2, beta3 = group['betas']
                amsgrad = group['amsgrad']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError('MTAdam does not support sparse gradients')

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 1
                        state['norms'] = [torch.ones(1, device=p.device) for _ in loss_array]
                        for j in range(len(loss_array)):
                            state['exp_avg'+str(j)] = torch.zeros_like(p.data)
                            state['exp_avg_sq'+str(j)] = torch.zeros_like(p.data)
                            if amsgrad:
                                state['max_exp_avg_sq'+str(j)] = torch.zeros_like(p.data)

                    # update moving norm
                    if state['step'] == 1:
                        state['norms'][loss_index] = torch.norm(p.grad)
                    else:
                        state['norms'][loss_index] = (
                            state['norms'][loss_index]*beta3
                            + (1-beta3)*torch.norm(p.grad)
                        )

                    # normalize grads using first valid anchor
                    if state['norms'][loss_index] > 1e-10:
                        for anchor_index in range(len(loss_array)):
                            if state['norms'][anchor_index] > 1e-10:
                                p.grad = (ranks[loss_index] * state['norms'][anchor_index]
                                          * p.grad / state['norms'][loss_index])
                                break

                    exp_avg = state['exp_avg'+str(loss_index)]
                    exp_avg_sq = state['exp_avg_sq'+str(loss_index)]
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq'+str(loss_index)]

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    if loss_index == len(loss_array) - 1:
                        state['step'] += 1

                    if group['weight_decay'] != 0:
                        p.grad = p.grad.add(p, alpha=group['weight_decay'])

                    exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                    if amsgrad:
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1

                    if loss_index == 0 or not hasattr(p, 'exp_avg_list'):
                        p.exp_avg_list = [exp_avg.clone()]
                        p.denom_list = [denom.clone()]
                        p.step_size_list = [step_size]
                    else:
                        p.exp_avg_list.append(exp_avg.clone())
                        p.denom_list.append(denom.clone())
                        p.step_size_list.append(step_size)

                    p.grad.detach_()
                    p.grad.zero_()


        for group in self.param_groups:
            for p in group['params']:
                if not hasattr(p, 'exp_avg_list'):
                    continue
                max_denom = p.denom_list[0]
                for d in p.denom_list[1:]:
                    max_denom = torch.max(max_denom, d)

                update = 0
                for exp_avg, step_size in zip(p.exp_avg_list, p.step_size_list):
                    update += -step_size * (exp_avg / max_denom)

                p.add_(update)

 
                p.exp_avg_list.clear()
                p.denom_list.clear()
                p.step_size_list.clear()

        self.training_step += 1
