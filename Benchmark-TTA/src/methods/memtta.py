
""" 참고
Class를 생성할 때, 반드시 지켜야할 원칙:
__init__함수는 반드시 model, optimizer를 입력 받은 후에 **kwrags를 입력받음 
reset()함수가 반드시 존재해야함.

reset(self): None -> domain이 변경되었을 때, 모델을 초기화시키는 방법
forward(self, sample): logits -> domain



그 외에는 optimizer가 update을 수행해줘야할 params를 뽑아주는 함수는 하나 만들어야함.
e.g.
(configure_model
collect_params)

""" 

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from copy import deepcopy

from collections import OrderedDict

class MemTTA(nn.Module):
    def __init__(
        self,
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        num_prompts: int = 4,
        model_dim: int = 512,
        num_heads: int = 8,
        alpha: float = 0.5,
        steps: int = 1
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        
        self.current_instance = 0

        self.num_prompts = num_prompts
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.steps = steps

        # reset을 위한 attribute 저장
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        for _ in range(self.steps): # 이거 거의 고정임.
            outputs = self.forward_and_adapt(x)
        return outputs

    def reset(self): # 이거 거의 고정임
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
    @torch.enable_grad()
    def forward_and_adapt(self, batch_data):
        # forward
        outputs = self.model(batch_data)
        # adapt
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs

    def update_model(self, model, optimizer):  # 이거 거의 고정임
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)


    @staticmethod
    def configure_model(
        model: nn.Module,
        num_prompts: int = 4,
        num_heads: int = 8,
        alpha: float = 0.5
    ) -> nn.Module:
        """
        1) 만약 CrossAttnBlock이면 => 건드리지 않고 그대로 반환
           (이미 우리가 만든 블록이므로, 내부 param은 requires_grad=True 유지)
        2) 그렇지 않은 모듈이면 => 파라미터 전부 requires_grad=False 로 세팅 (동결)
        3) 만약 nn.Sequential이면, 각 블록 뒤에 CrossAttnBlock 삽입 (AttnBlock 파라미터만 True)
        4) 일반 모듈이면 => 자식 모듈에 대해 재귀적으로 configure_model 호출
        5) 최종적으로 변환된 모듈 반환
        """
        # (1) CrossAttnBlock -> 이미 새로 만든 블록이므로 아무것도 안 함
        if isinstance(model, CrossAttnBlock):
            return model

        # (2) 현재 모듈의 파라미터 전부 동결
        for p in model.parameters():
            p.requires_grad = False

        # (3) nn.Sequential -> 내부를 순회하며, 각 블록 뒤에 CrossAttnBlock 삽입
        if isinstance(model, nn.Sequential):
            items = list(model.named_children())  # [(name, module), ...]
            new_items = []
            for i, (blk_name, blk_module) in enumerate(items):
                # 먼저 blk_module 재귀 적용
                new_blk_module = MemTTA.configure_model(
                    blk_module,
                    num_prompts=num_prompts,
                    num_heads=num_heads,
                    alpha=alpha
                )
                # 재귀 결과를 삽입
                new_items.append((blk_name, new_blk_module))

                # blk_module 출력 채널 추론
                out_dim = get_out_channels(new_blk_module)
                if out_dim is not None:
                    # CrossAttnBlock을 새로 추가 (이건 학습가능)
                    attn_block = CrossAttnBlock(
                        model_dim=out_dim,
                        num_prompts=num_prompts,
                        num_heads=num_heads,
                        alpha=alpha
                    )
                    # CrossAttnBlock의 파라미터는 기본적으로 requires_grad=True
                    # => 별도 동결 처리 안 함
                    new_items.append((f"attn_{i}", attn_block))

            replaced = nn.Sequential(OrderedDict(new_items))
            return replaced

        # (4) nn.Sequential이 아닌 일반 모듈 -> 자식 모듈들 재귀 처리
        for child_name, child_module in model.named_children():
            new_child = MemTTA.configure_model(
                child_module,
                num_prompts=num_prompts,
                num_heads=num_heads,
                alpha=alpha
            )
            if new_child is not child_module:
                setattr(model, child_name, new_child)

        return model

    @staticmethod
    def collect_params(model: nn.Module):
        """
        학습해야 할 파라미터와 그 이름들을 모아 반환
        """
        names = []
        params = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)
        return params, names
    
    @staticmethod
    def _freeze_except_attn(module: nn.Module):
        """
        - 우선 현재 모듈의 모든 파라미터를 requires_grad=False
        - 자식(child)을 순회하며,
          만약 CrossAttnBlock이면 해당 블록(및 내부) 전부 unfreeze
          그렇지 않으면 재귀적으로 계속 동결
        """
        # (1) 현재 모듈의 모든 파라미터를 일단 동결
        for p in module.parameters():
            p.requires_grad = False

        # (2) 자식 모듈들을 확인
        for child_name, child_module in module.named_children():
            if isinstance(child_module, CrossAttnBlock):
                # CrossAttnBlock이면, 그 내부까지 포함 전부 requires_grad=True로
                for p in child_module.parameters():
                    p.requires_grad = True
                # 내부로 더 들어가지 않아도 됨 (이미 CrossAttnBlock 자체가 우리가 원하는 메타 블록)
            else:
                # 그 외 모듈은 재귀적으로 또 들어가서 같은 로직 적용
                MemTTA._freeze_except_attn(child_module)
    

class CrossAttnBlock(nn.Module):
    """
    - 기존 feature + 프롬프트 간 Cross-Attention 수행
    - 원본 feature와 attn 결과를 (1 - alpha) : alpha 로 섞어서 출력
    """
    def __init__(self, model_dim: int, num_prompts: int, num_heads: int, alpha: float):
        super().__init__()
        self.model_dim = model_dim
        self.num_prompts = num_prompts
        self.num_heads = num_heads
        self.alpha = alpha

        # (num_prompts, model_dim) 형태의 learnable prompt 파라미터
        self.prompts = nn.Parameter(torch.randn(num_prompts, model_dim))

        # Cross-Attention용 MultiheadAttention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) 형태의 이미지 feature라고 가정
           -> (B, N, C) 변환 후 프롬프트와 Cross-Attn
        """
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, N, C)
        out_reshaped = x.view(B, C, -1).transpose(1, 2)  # (B, N, C)

        # prompts: (num_prompts, model_dim)
        # -> batch 차원 B를 맞추기 위해 expand
        expanded_prompts = self.prompts.unsqueeze(0).expand(B, -1, -1)

        # query = out_reshaped, key = prompts, value = prompts
        attn_out, _ = self.cross_attn(
            query=out_reshaped,
            key=expanded_prompts,
            value=expanded_prompts
        )
        # attn_out: (B, N, C)

        # (B, N, C) -> (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

        # 최종: (1-alpha)*원본 + alpha*attn
        mixed_out = (1 - self.alpha) * x + (self.alpha) * attn_out
        return mixed_out

    
def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module

def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def get_out_channels(module: nn.Module):
    """
    모듈(블록)의 '출력 채널 수'를 추론해서 정수(int)를 반환
    - nn.Conv2d -> out_channels
    - nn.Linear -> out_features
    - (ResNet)BasicBlock -> block.expansion * block.planes (등등)
    - 모듈마다 상이하므로, 필요한 경우 조건문을 확장
    - 모르는 경우(None) 반환
    """
    # (A) Conv2d
    if isinstance(module, nn.Conv2d):
        return module.out_channels

    # (B) Linear
    if isinstance(module, nn.Linear):
        return module.out_features

    # (C) 예: Bottleneck / BasicBlock (ResNet 계열) - 모델마다 구현 다름
    #   - WideResNet, torchvision.ResNet, etc. 구조가 다르니, 
    #     실제 코드 보고 고쳐야 함
    # 여기서는 간단 예시로, "만약 module에 out_channels라는 속성이 있으면 반환"
    # (실제 WideResNet 블록은 'self.block' 등 다른 서브모듈로 감싸져 있을 수도 있음)
    if hasattr(module, 'out_channels'):
        return getattr(module, 'out_channels')

    # (D) nn.Sequential이나 다른 모듈 -> None
    return None




if __name__ == "__main__":
    from src.models.load_model import load_model
    from src.data import load_dataset
    import os

    model_list = ['Hendrycks2020AugMix_ResNeXt', 'resnet50', 'WideResNet', 'officehome_shot', 'domainnet126_shot', 'vit', 'convnext_base', 'efficientnet_b0']
    checkpoint_dir = "./ckpt/"
    source_domain = 'origin' # 이거 도메인마다 다르게바꿔라...
    dataset_list = ['cifar10_c', 'cifar100_c', 'imagenet_c', 'officehome', 'domainnet126']
    
    # for dataset_name in dataset_list:
    #     testset, testloader = load_dataset(dataset) 
    for model_name in model_list:
        model = load_model(model_name, checkpoint_dir=os.path.join(checkpoint_dir, 'models'), domain=source_domain)
        model = MemTTA.configure_model(model)
        param, param_name = MemTTA.collect_params(model)

        breakpoint()