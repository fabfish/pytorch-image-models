# fish: hack eva to test

import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
# import kfac.backend as backend

from eva_utils import get_vector_a, get_vector_g
import logging
logger = logging.getLogger()

class KFAC(optim.Optimizer):
    """Accelerate Distributed K-FAC with Sublinear Memory Cost
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.03)
      kl_clip (float): clipping parameter for gradient scaling (kl_clip > 0: kl-clip, kl_clip = 0: re-scale, kl-clip < 0: None)
      factor_decay (float): running average coefficient for KVs
      exclude_vocabulary_size: exclude the pre-softmax linear layer in the Transformer
      hook_enabled (bool): enable the hook events to save the immediate states (a and g)
      exclude_parts='': exclude CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor for time breakdowns
    """
    def __init__(self,
                 model,
                 lr=0.1,
                 damping=0.03,
                 fac_update_freq=1,
                 kfac_update_freq=1,
                 kfac_batch_size=16,
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):

        # For compatibility with `KFACParamScheduler`
        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        kfac_update_freq=kfac_update_freq) 

        super(KFAC, self).__init__(model.parameters(), defaults)

        self.fac_update_freq = fac_update_freq
        self.kfac_batch_size = kfac_batch_size
        self.kl_clip = kl_clip if (kl_clip is not None and kl_clip >= 0) else None
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled
        
        # register hooks
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_a, self.m_g = {}, {}
        self.handles = []
        
        # scheduling results
        self.module_ranks = None

        self.steps = 0

    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _forward_hook_event(self, module, input):
        """Default: hook for saving input (a)"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_a(input[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_a:
                    self.m_a[module] = new
                else:
                    #self.m_a[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_a[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
                    #xi =  math.pow(self.steps+1, -self.factor_decay)
                    #self.m_a[module].mul_(1-xi).add_(new, alpha=xi)
            # if backend.comm.size() > 1:
            #     self.handles.append(backend.comm.allreduce_async_(self.m_a[module], op=backend.comm.Average))

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving gradient w.r.t output (g)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_g(grad_output[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_g:
                    self.m_g[module] = new
                else:
                    #self.m_g[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_g[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
                    #xi =  math.pow(self.steps+1, -self.factor_decay)
                    #self.m_g[module].mul_(1-xi).add_(new, alpha=xi)
            # if backend.comm.size() > 1:
            #     self.handles.append(backend.comm.allreduce_async_(self.m_g[module], op=backend.comm.Average))

    def _register_module_hooks(self, model):
        """Register forard/backward hooks to supported modules"""
        supported_modules = {'Linear', 'Conv2d'}
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in supported_modules:
                if self.exclude_vocabulary_size is not None and classname == 'Linear' and module.out_features == self.exclude_vocabulary_size:
                    continue # exclude the pre-softmax linear layer in the Transformer model
                self.modules.append(module)
                module.register_forward_pre_hook(self._forward_hook_event)
                module.register_backward_hook(self._backward_hook_event)  # used in pytorch1.4, and pytorch1.8 (full_backward_hook is not fired when its grad_input is None)
                #module.register_full_backward_hook(self._backward_hook_event)  # used in pytorch1.10
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1
        # if backend.comm.rank() == 0:
        #     logger.info("#register modules: %s", len(self.modules))

	### Precondition gradients
    def _precondition_grads(self):
        """Compute preconditioned gradients via Eva"""
        g_sum = 0
        v_sum = 0
        vg_sum = 0

        for module in self.modules:
            # get ma, mg, grad
            ma = self.m_a[module].view(-1, 1).half()
            mg = self.m_g[module].view(-1, 1).half()
            # fish: fix precision bug for deit
            # grad = self._get_grad(module)
            grad = self._get_grad(module).half()
            
            #if backend.comm.rank() == 0:
            #    logger.info("mg: %s" % (mg))
            
            # compute intermediate states
            a = (ma.T @ ma).item()
            g = (mg.T @ mg).item()
            # print(mg.T.dtype,mg.dtype,g.dtype)
            # print((mg.T @ grad @ ma).item().dtype)
            # print('*******************')
            # print(mg.T.dtype,grad.dtype,ma.dtype)
            ag = (mg.T @ grad @ ma).item()
            
            #if backend.comm.rank() == 0 and self.steps % 60 == 0:
            #    logger.info("a: %f, g: %f, ag: %f" % (a, g, ag))
            #    logger.info("beta: %f", ag/(a * g + self.damping))

            # compute preconditioned grads
            v = (mg @ ma.T).mul_(-ag/(a * g + self.damping))
            v.add_(grad)
            v.div_(self.damping)

            # weight and bias
            if module.bias is not None:
                weight = v[:, :-1].view(module.weight.grad.data.size())
                bias = v[:, -1:].view(module.bias.grad.data.size())
                # transform preconditioned gradient into gradient scale
                if self.kl_clip is not None:
                    if self.kl_clip > 0:
                        vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                        vg_sum += (bias * module.bias.grad.data * self.lr ** 2).sum().item()
                    else:
                        v_sum += (weight * weight).sum().item()
                        v_sum += (bias * bias).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                        g_sum += (module.bias.grad.data * module.bias.grad.data).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
                module.bias.grad.data.copy_(bias)
                del grad
            else:
                weight = v.view(module.weight.grad.data.size())
                if self.kl_clip is not None:
                    if self.kl_clip > 0:
                        vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                    else:
                        v_sum += (weight * weight).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
            del v

        # scale preconditioned gradient
        if self.kl_clip is not None:
            if self.kl_clip > 0: # kl-clip
                nu = min(1.0, math.sqrt(self.kl_clip / vg_sum)) if vg_sum > 0 else 1.0
            else: # re-scale
                nu = math.sqrt(g_sum / v_sum)

            for module in self.modules:
                module.weight.grad.data.mul_(nu)
                if module.bias is not None:
                    module.bias.grad.data.mul_(nu)

    def _get_grad(self, module):
        """Get gradient with shape [output_dim, input_dim] for module"""
        if module.__class__.__name__ == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)
        else:
            grad = module.weight.grad.data
        if module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad    


    ### Perform one K-FAC step
    @torch.no_grad()
    def step(self, closure=None, epoch=None):
        """Perform one K-FAC step"""

        # update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        # if self.steps % self.fac_update_freq == 0 and backend.comm.size() > 1:
        #     for handle in self.handles:
        #         backend.comm.synchronize(handle)
        #     self.handles = []
        
        self._precondition_grads()

        self.steps += 1

class KFACParamScheduler():
    """Updates KFAC hyper-parameters at each epoch
    Args:
      kfac (KFAC): wrapped KFAC preconditioner
      damping_alpha (float): multiplicative factor of the damping (default: 1)
      damping_schedule (list): list of epochs to multiply the damping by `damping_alpha` (default: None)
      update_freq_alpha (float): multiplicative factor of the KFAC update freq (default: 1)
      update_freq_schedule (list): list of epochs to multiply the KFAC update freq by `update_freq_alpha` (default: None)
      start_epoch (int): starting epoch, for use if resuming training from checkpoint (default: 0)
    """
    def __init__(self,
                 kfac,
                 damping_alpha=1,
                 damping_schedule=None,
                 update_freq_alpha=1,
                 update_freq_schedule=None,
                 start_epoch=0):

        self.kfac = kfac
        params = self.kfac.param_groups[0]

        self.damping_base = params['damping']
        self.damping_alpha = damping_alpha
        self.damping_schedule = damping_schedule
        self.damping_factor_func = \
                self._get_factor_func(self.damping_schedule,
                                     self.damping_alpha)

        self.fac_update_freq_base = params['fac_update_freq']
        self.kfac_update_freq_base = params['kfac_update_freq']
        self.update_freq_alpha = update_freq_alpha
        self.update_freq_schedule = update_freq_schedule
        self.update_freq_factor_func = \
                self._get_factor_func(self.update_freq_schedule,
                                     self.update_freq_alpha)

        self.epoch = start_epoch

    def _get_factor_func(self, schedule, alpha):
        """Returns a function to compute an update factor using the epoch"""
        if schedule is not None:
            schedule.sort(reverse=True)
        else:
            schedule = []

        def factor_func(epoch):
            factor = 1.
            for e in schedule:
                if epoch >= e:
                    factor *= alpha
            return factor

        return factor_func

    def step(self, epoch=None):
        """Update KFAC parameters"""
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        params = self.kfac.param_groups[0]

        params['damping'] = self.damping_base * self.damping_factor_func(self.epoch)

        factor = self.update_freq_factor_func(self.epoch)
        params['fac_update_freq'] = int(self.fac_update_freq_base * factor)
        params['kfac_update_freq'] = int(self.kfac_update_freq_base * factor)
