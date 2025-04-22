import torch
from torch.utils.data import DataLoader, Sampler, Subset, ConcatDataset
from typing import List, Tuple
import numpy as np
from numpy.random import dirichlet

from .data import load_dataset


class DomainWrapperDataset(torch.utils.data.Dataset):
    """
    원본 Dataset을 감싸서, __getitem__ 시 (img, label, domain_label)을 반환하도록 하는 래퍼.
    """
    def __init__(self, base_dataset, domain_name: str, severity: int):
        super().__init__()
        self.base_dataset = base_dataset
        # domain label을 문자열로 정의 (예: "domainA2")
        self.domain_label_str = f"{domain_name}{severity}"

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, label = self.base_dataset[idx]
        # (img, label, domain_label)을 tuple로 반환
        return img, label, self.domain_label_str


def load_mixed_dataset(dataset: str,
                       root: str,
                       batch_size: int,
                       workers: int,
                       domain_names_loop: list,  # 예: ["domainA", "domainB", "domainC", ...]
                       severities: list,        # 예: [0, 1, 2, ...]
                       imbalance_ratio: int = 1,  
                       split: str = "all",
                       adaptation: str = None,
                       level: int = None,
                       num_aug: int = 1,
                       transforms=None,
                       ckpt: str = None
                       ) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    """
    여러 도메인(domain)과 severity를 하나로 합쳐 불균형(long-tailed) 데이터셋을 구성하고,
    (img, label, domain_label) 형태로 데이터를 뽑아주는 DataLoader를 반환한다.
    
    Args:
        dataset (str): 사용할 데이터셋 이름.
        root (str): 데이터가 저장된 루트 디렉토리.
        batch_size (int): DataLoader의 batch size.
        workers (int): DataLoader의 num_workers.
        domain_names_loop (list): 사용할 도메인 이름들의 리스트.
        severities (list): 사용할 severity 레벨들의 리스트.
        imbalance_ratio (int): IR(Long-Tailed Imbalance Ratio). 
                               예: 100이면 가장 큰 도메인과 가장 작은 도메인의 샘플 수가 100배 차이.
        split (str): "train", "test" 혹은 "all"처럼 데이터 분할을 선택.
        adaptation (str): 특정 Adaptation 모드 (필요시 사용).
        level (int): 레벨(단일 도메인 severity와 혼동되지 않도록 주의).
        num_aug (int): 데이터 증강 횟수.
        transforms (callable): 데이터셋에 적용할 변환 함수(전처리).
        ckpt (str): 사전학습된 checkpoint 경로 (필요시).
    
    Returns:
        Tuple[torch.utils.data.Dataset, DataLoader]: 
            (하나로 합쳐진 ConcatDataset, 해당 Dataset을 로드하는 DataLoader)
            여기서 각 샘플은 (img, label, domain_label) 형태.
    """
    
    # 1) 도메인×severity별로 dataset_list 만들기
    dataset_list = []
    for i_dom, domain_name in enumerate(domain_names_loop):
        for severity in severities:
            # 기본 (img, label)을 반환하는 원본 Dataset
            ds = load_dataset(
                dataset=dataset,
                root=root,
                batch_size=batch_size,
                workers=workers,
                domain=domain_name,
                level=severity,
                split=split,
                adaptation=adaptation,
                num_aug=num_aug,
                transforms=transforms,
                ckpt=ckpt
            )
            # 도메인 레이블을 추가해주는 래퍼로 감싼다.
            wrapped_ds = DomainWrapperDataset(ds, domain_name, severity)
            dataset_list.append(wrapped_ds)
    
    # 혹시 dataset_list의 길이가 1이라면 그대로 반환해도 무방
    if len(dataset_list) == 1:
        single_loader = DataLoader(dataset_list[0], batch_size=batch_size,
                                   shuffle=True, num_workers=workers)
        return dataset_list[0], single_loader
    
    # 2) Long-tailed를 만들기 위해, 가장 긴 데이터셋 길이를 찾음
    lengths = [len(ds) for ds in dataset_list]
    max_len = max(lengths)
    
    # 3) IR에 따라 각 서브 데이터셋(Subset)을 구성
    #    예: 도메인 i 에 대해, 샘플 수 = max_len * (1/IR)^(i/(n-1))
    #    - i=0: (1/IR)^0 = 1.0 -> 최대 길이 (전체)
    #    - i=n-1: (1/IR)^1 = 1/IR
    #    => i가 커질수록 샘플 수가 지수적으로 줄어드는 전형적인 long-tailed
    n_dom = len(dataset_list)
    subsets = []
    
    for i, ds in enumerate(dataset_list):
        if imbalance_ratio > 1 and n_dom > 1:
            # i의 증가에 따라 샘플 수 비율이 지수 감소
            ratio = (1 / imbalance_ratio) ** (i / (n_dom - 1))
        else:
            # imbalance_ratio == 1이면 균등
            ratio = 1.0
        
        sub_len = int(max_len * ratio)
        sub_len = min(sub_len, len(ds))  # 원본 길이 넘어가지 않도록
        
        # 서브샘플 추출(undersampling). 필요하다면 replace=True로 오버샘플링 가능
        indices = np.random.choice(len(ds), size=sub_len, replace=False)
        subset_ds = Subset(ds, indices)
        
        subsets.append(subset_ds)
    
    # 4) Subset들을 합쳐서 최종 Dataset으로 만들기
    mixed_dataset = ConcatDataset(subsets)
    
    # 5) DataLoader 구성
    mixed_dataloader = DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=True,  # 전체 합친 후 섞기
        num_workers=workers,
        drop_last=False
    )
    
    return mixed_dataset, mixed_dataloader
