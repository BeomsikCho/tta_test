"""
Copyright to COME Authors, ICLR 2025
built upon on SAR code.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from ..utils.sam import SAM

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

class SAR_COME(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, lr, batch_size, num_classes, steps=1, episodic=False, e_margin=0.4*math.log(1000), reset_constant=0.2):
        super().__init__()
        self.model, self.optimizer = self.prepare_SAR_model_and_optimizer(model, lr, batch_size)
        self.steps = steps
        
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.margin_e0 = e_margin  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant = reset_constant  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        self.num_classes = num_classes
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs, ema, reset_flag = forward_and_adapt_sar_come(x, self.model, self.optimizer, self.margin_e0, self.reset_constant, self.ema, num_classes=self.num_classes)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    @staticmethod
    def configure_model(model):
        """Configure model for use with SAR_COME."""
        model.train()
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    @staticmethod
    def prepare_SAR_model_and_optimizer(model, lr, batch_size):
        model = SAR_COME.configure_model(model)
        params, param_names = SAR_COME.collect_params(model)
        base_optimizer = torch.optim.SGD
        if batch_size == 1:
            lr = 2 * lr
        optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
        return model, optimizer
    
    @staticmethod
    def check_model(model):
        """Check model for compatability with SAR_COME."""
        is_training = model.training
        assert is_training, "SAR_COME needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "SAR_COME needs params to update: " \
                               "check which require grad"
        assert not has_all_params, "SAR_COME should not update all params: " \
                                   "check which require grad"
        has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
        assert has_norm, "SAR_COME needs normalization layer parameters for its optimization"


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def dirichlet_entropy(x: torch.Tensor, num_classes:int):#key component of COME
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + num_classes)
    uncertainty = num_classes / (torch.sum(torch.exp(x), dim=1, keepdim=True) + num_classes)
    probability = torch.cat([brief, uncertainty], dim=1) + 1e-7
    entropy = -(probability * torch.log(probability)).sum(1)
    return entropy

@torch.enable_grad()   # ensure grads in possible no grad context for testing
def forward_and_adapt_sar_come(x, model, optimizer, margin, reset_constant, ema, num_classes:int = 1000):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    # forward
    outputs = model(x)
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    # COME: replace softmax_entropy with dirichlet_entropy 
    entropys = dirichlet_entropy(outputs, num_classes)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()
    optimizer.first_step(zero_grad=True) 
    entropys2 = dirichlet_entropy(model(x), num_classes)
    entropys2 = entropys2[filter_ids_1]  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  
    loss_second.backward()
    optimizer.second_step(zero_grad=True)
    
    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < reset_constant:
            print(f"ema < reset_constant: {reset_constant}, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
