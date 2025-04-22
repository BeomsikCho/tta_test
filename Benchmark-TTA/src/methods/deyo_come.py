import math
import torch
import torch.nn as nn
import torch.jit
import torchvision
import numpy as np
from einops import rearrange


class DeYO_COME(nn.Module):
    """DeYO online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    Once DeYOed, a model adapts itself by updating on every forward.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_classes: int,
        steps: int = 1,
        deyo_margin: float = 0.5 * math.log(1000),
        margin_e0: float = 0.4 * math.log(1000),
        # 이하 설정들은 원래 args에서 가져오던 항목을 __init__에서 직접 받도록 변경
        aug_type: str = 'occ',
        occlusion_size: int = 16,
        row_start: int = 0,
        column_start: int = 0,
        patch_len: int = 4,
        plpd_threshold: float = 0.0,
        reweight_ent: float = 0.0,
        reweight_plpd: float = 0.0,
        episodic: bool=False
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.steps = steps

        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0

        # __init__에서 받은 설정들 저장
        self.aug_type = aug_type
        self.occlusion_size = occlusion_size
        self.row_start = row_start
        self.column_start = column_start
        self.patch_len = patch_len
        self.plpd_threshold = plpd_threshold
        self.reweight_ent = reweight_ent
        self.reweight_plpd = reweight_plpd

        # 필요하면 reset 용으로 state 백업할 수도 있음
        self.model_state = None
        self.optimizer_state = None
        self.ema = None
        self.episodic = episodic

    def forward(self, x, targets=None, flag=False, group=None):
        if self.episodic:
            self.reset()
            
        # targets 유무, flag에 따라 branch 처리하는 로직은 기존과 동일
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = self.forward_and_adapt_deyo(
                        x, self.model, self.optimizer,
                        self.deyo_margin, self.margin_e0,
                        targets, flag, group, num_classes=self.num_classes
                    )
                else:
                    outputs = self.forward_and_adapt_deyo(
                        x, self.model, self.optimizer,
                        self.deyo_margin, self.margin_e0,
                        targets, flag, group, num_classes=self.num_classes
                    )
            if flag:
                return outputs, backward, final_backward
            else:
                return outputs
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = self.forward_and_adapt_deyo(
                        x, self.model, self.optimizer,
                        self.deyo_margin, self.margin_e0,
                        targets, flag, group, num_classes=self.num_classes
                    )
                else:
                    outputs = self.forward_and_adapt_deyo(
                        x, self.model, self.optimizer,
                        self.deyo_margin, self.margin_e0,
                        targets, flag, group, num_classes=self.num_classes
                    )
            if flag:
                return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            else:
                return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    @staticmethod
    def configure_model(model):
        """Configure model for use with DeYO."""
        # train mode, because DeYO optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what DeYO updates
        model.requires_grad_(False)
        # configure norm for DeYO updates: enable grad + force batch statisics
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
        """Collect the affine scale + shift parameters from norm layers."""
        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation
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

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np_, p in m.named_parameters():
                    if np_ in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np_}")

        return params, names

    @torch.enable_grad()
    def forward_and_adapt_deyo(
        self, x, model, optimizer,
        deyo_margin, margin, targets=None, flag=True, group=None, num_classes:int = 1000
    ):
        """Forward and adapt model input data."""
        outputs = model(x)
        if not flag:
            return outputs

        optimizer.zero_grad()
        entropys = dirichlet_entropy(outputs, num_classes)

        # 1차 필터링 (entropy)
        filter_ids_1 = torch.where((entropys < deyo_margin))
        entropys = entropys[filter_ids_1]
        backward = len(entropys)
        if backward == 0:
            if targets is not None:
                return outputs, 0, 0, 0, 0
            return outputs, 0, 0

        # x_prime 만들기
        x_prime = x[filter_ids_1].detach()
        # aug_type 적용
        if self.aug_type == 'occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
            x_prime[:, :, self.row_start:self.row_start + self.occlusion_size,
                        self.column_start:self.column_start + self.occlusion_size] = occlusion_window

        elif self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(
                ((x.shape[-1] // self.patch_len) * self.patch_len,
                 (x.shape[-1] // self.patch_len) * self.patch_len)
            )
            resize_o = torchvision.transforms.Resize((x.shape[-1], x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(
                x_prime,
                'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w',
                ps1=self.patch_len, ps2=self.patch_len
            )
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(
                x_prime,
                'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)',
                ps1=self.patch_len, ps2=self.patch_len
            )
            x_prime = resize_o(x_prime)

        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(
                x_prime,
                'b c (ps1 ps2) -> b c ps1 ps2',
                ps1=x.shape[-1],
                ps2=x.shape[-1]
            )

        # x_prime에 대해서 한 번 더 forward (no_grad)
        with torch.no_grad():
            outputs_prime = model(x_prime)

        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim=1)
        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - \
               torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
        plpd = plpd.reshape(-1)

        # 2차 필터링 (plpd)
        filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        
        entropys = entropys[filter_ids_2]
        final_backward = len(entropys)

        # 정답(타겟)이 있을 때
        if targets is not None:
            corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()

        if final_backward == 0:
            del x_prime
            del plpd
            if targets is not None:
                return outputs, backward, 0, corr_pl_1, 0
            return outputs, backward, 0

        plpd = plpd[filter_ids_2]

        if targets is not None:
            corr_pl_2 = (targets[filter_ids_1][filter_ids_2] ==
                         prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

        # reweight
        if self.reweight_ent or self.reweight_plpd:
            coeff = (self.reweight_ent * (1 / (torch.exp(entropys.clone().detach() - margin)))
                     + self.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach()))))
            entropys = entropys.mul(coeff)

        loss = entropys.mean(0)

        if final_backward != 0:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        del x_prime
        del plpd

        if targets is not None:
            return outputs, backward, final_backward, corr_pl_1, corr_pl_2
        return outputs, backward, final_backward


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def dirichlet_entropy(x: torch.Tensor, num_classes: int):#key component of COME
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + num_classes)
    uncertainty = num_classes / (torch.sum(torch.exp(x), dim=1, keepdim=True) + num_classes)
    probability = torch.cat([brief, uncertainty], dim=1) + 1e-7
    entropy = -(probability * torch.log(probability)).sum(1)
    return entropy

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
