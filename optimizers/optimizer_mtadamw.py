import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
Tensor = torch.Tensor
import math
from torch.optim.optimizer import Optimizer


class MTAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid lr: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta3: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MTAdamW, self).__init__(params, defaults)
        self.training_step = 0

    def __setstate__(self, state):
        super(MTAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, loss_array, ranks, closure=None):
        """
        loss_array: list of scalar losses (torch tensors)
        ranks: list or tensor of same length: external task weights/scalars
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._mtadam_step(loss_array, ranks)
        return loss

    def _mtadam_step(self, loss_array, ranks):
        num_tasks = len(loss_array)
        device = None

        # 1) For each task: run backward and collect grads clone per param
        # We'll store clones in state[p]['grads_clones'] as a list of length num_tasks.
        # To avoid interfering with existing grads, zero them between tasks.
        for group in self.param_groups:
            for p in group['params']:
                # initialize state containers
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 1
                    # per-task moving norm for this param
                    state['norms'] = [torch.tensor(1.0, device=p.device) for _ in range(num_tasks)]
                    # per-task exponential moment storage
                    for j in range(num_tasks):
                        state['exp_avg_{}'.format(j)] = torch.zeros_like(p.data)
                        state['exp_avg_sq_{}'.format(j)] = torch.zeros_like(p.data)
                        if group['amsgrad']:
                            state['max_exp_avg_sq_{}'.format(j)] = torch.zeros_like(p.data)
                # ensure a temporary list exists for grads
                state['grads_clones'] = [None] * num_tasks

        for t_idx, loss in enumerate(loss_array):
            # zero grads
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
            # backward this loss
            loss.backward(retain_graph=True)
            # collect clones
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError('MTAdamW does not support sparse gradients')
                    state = self.state[p]
                    # clone grad to avoid future in-place edits
                    g_clone = p.grad.detach().clone()
                    state['grads_clones'][t_idx] = g_clone
                    device = p.device
            # zero grads again to prepare next task
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

        # 2) For each parameter, apply PCGrad pairwise projection to its task-grad list,
        #    then apply normalization by moving norms, then update exp averages & param.
        for group in self.param_groups:
            lr = group.get('lr', self.defaults['lr'])
            beta1, beta2, beta3 = group['betas']
            eps = group['eps']
            amsgrad = group['amsgrad']
            wd = group['weight_decay']

            for p in group['params']:
                state = self.state[p]
                grads = state.get('grads_clones', None)
                if grads is None:
                    continue  # nothing collected for this param

                # Build list of available grads (some tasks might have None due to missing grad)
                task_grads = []
                task_indices = []
                for i, g in enumerate(grads):
                    if g is not None:
                        task_grads.append(g)
                        task_indices.append(i)

                if len(task_grads) == 0:
                    continue

                # PCGrad: for each grad_i, project out components that conflict with earlier grads
                # (we can randomize order or use given order; here use given order)
                # Work on clones to avoid mutating stored clones (they're already clones)
                proc_grads = [g.clone() for g in task_grads]

                # pairwise projection
                for i in range(len(proc_grads)):
                    gi = proc_grads[i]
                    for j in range(i):
                        gj = proc_grads[j]
                        # dot product
                        dot = torch.dot(gi.view(-1), gj.view(-1))
                        if dot < 0:
                            # project gi to remove component along gj
                            gj_norm_sq = torch.dot(gj.view(-1), gj.view(-1))
                            if gj_norm_sq.item() > 0:
                                gi = gi - (dot / (gj_norm_sq + 1e-16)) * gj
                    proc_grads[i] = gi

                # Now proc_grads aligns with task_indices
                # Next: optionally normalize each proc_grad to have same magnitude as moving norm
                # Update moving norms and compute normalized grads
                normalized_grads = []
                for local_idx, task_i in enumerate(task_indices):
                    g_proc = proc_grads[local_idx]
                    # compute current grad norm
                    g_norm = torch.norm(g_proc).detach()
                    # update moving average of norm
                    if state['step'] == 1:
                        state['norms'][task_i] = g_norm if g_norm.numel() else state['norms'][task_i]
                    else:
                        state['norms'][task_i] = state['norms'][task_i] * beta3 + (1.0 - beta3) * g_norm
                    # avoid division by zero
                    denom_norm = state['norms'][task_i].item()
                    if denom_norm < 1e-12:
                        denom_norm = 1.0
                    # scale to match moving norm (this keeps magnitude comparable across tasks)
                    scaled = g_proc * (state['norms'][task_i] / (g_norm + 1e-16))
                    normalized_grads.append((task_i, scaled))

                # Now update moments per task using normalized grads and combine updates
                # We will keep task-specific exp_avg and exp_avg_sq in state
                # Also compute per-task denom and step_size (bias corrected)
                step = state['step']
                updates = []
                denoms = []
                step_sizes = []

                for (task_i, g_normed) in normalized_grads:
                    exp_avg = state['exp_avg_{}'.format(task_i)]
                    exp_avg_sq = state['exp_avg_sq_{}'.format(task_i)]
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq_{}'.format(task_i)]

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    # update moments
                    exp_avg.mul_(beta1).add_(g_normed, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_normed, g_normed, value=1.0 - beta2)

                    if amsgrad:
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    step_size = lr / bias_correction1

                    # store for later combination (note: exp_avg and denom are references to state tensors)
                    updates.append((task_i, exp_avg, denom, step_size))
                    denoms.append(denom)
                    step_sizes.append(step_size)

                # increment step once per param after processing all tasks (similar to original behaviour)
                # (Alternatively could keep global counter)
                state['step'] = state['step'] + 1 if state['step'] == step else state['step'] + 1

                # combine denoms: use max denom (similar to original) to avoid too-large step
                max_denom = denoms[0]
                for d in denoms[1:]:
                    max_denom = torch.max(max_denom, d)

                # decoupled weight decay (AdamW): apply to param.data directly
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                # combine per-task updates (weighted by external ranks if provided)
                combined_update = torch.zeros_like(p.data)
                for (task_i, exp_avg, denom, step_size) in updates:
                    # find rank for this task
                    r = ranks[task_i] if hasattr(ranks, '__len__') else ranks[task_i].item()
                    # update_step = - step_size * r * (exp_avg / max_denom)
                    # use broadcasting-safe division
                    update_step = - (step_size * r) * (exp_avg / (max_denom + 1e-16))
                    combined_update.add_(update_step)

                # apply update
                p.add_(combined_update)

                # cleanup temporary clones
                state['grads_clones'] = None

        self.training_step += 1
