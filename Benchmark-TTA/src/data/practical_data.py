import torch
from torch.utils.data import DataLoader, Sampler
from typing import List, Tuple
import numpy as np
from numpy.random import dirichlet

from .data import load_dataset  # 경로에 맞게 수정

# ------------------------------------------------------------
# 2) DatumBase (프로젝트에 이미 있다면 삭제해도 무방)
# ------------------------------------------------------------
class DatumBase:
    def __init__(self, img=None, label=0, domain=0, classname=""):
        self.img = img
        self.label = label
        self.domain = domain
        self.classname = classname

class LabelDirichletSampler(Sampler):
    """
    한 도메인의 샘플을 Dirichlet 분포 기반 time‑slot 으로 섞어
    클래스‑상관(correlated) 시퀀스를 만들어 주는 Sampler
    """
    def __init__(self,
                 data_source: List[DatumBase],
                 gamma: float = 0.5,
                 num_slots: int = None):
        self.data_source = data_source
        self.gamma = gamma

        # 클래스별 인덱스 테이블
        self.class_dict = {}
        for idx, item in enumerate(data_source):
            if isinstance(item, DatumBase):
                label = item.label
            elif isinstance(item, tuple):
                label = item[1]
                if torch.is_tensor(label):
                    label = label.item()
            else:
                raise TypeError(f"Unsupported sample type: {type(item)}")

            self.class_dict.setdefault(label, []).append(idx)

        self.num_class = len(self.class_dict)
        self.num_slots = num_slots if num_slots is not None else min(self.num_class, 100)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        # 슬롯별 컨테이너
        slot_indices = [[] for _ in range(self.num_slots)]

        # (num_class, num_slots) 크기의 Dirichlet 비율표
        dist = dirichlet([self.gamma] * self.num_slots, self.num_class)

        # 클래스별 인덱스를 슬롯으로 분할
        for cls_indexes, slot_ratio in zip(self.class_dict.values(), dist):
            cls_indexes = np.array(cls_indexes)
            # 누적 비율(cumsum) 기준 split
            splits = np.split(
                cls_indexes,
                (np.cumsum(slot_ratio)[:-1] * len(cls_indexes)).astype(int)
            )
            for s, part in enumerate(splits):
                slot_indices[s].append(part)

        # 슬롯 순회하며 인덱스 배출
        final_indices = []
        for s_idx in slot_indices:
            np.random.shuffle(s_idx)  # 슬롯 내부 클래스 순서 무작위
            for part in s_idx:
                final_indices.extend(part)

        return iter(final_indices)

def load_practical_dataset(dataset: str,
                           root: str,
                           domain: str,
                           batch_size: int,
                           workers: int,
                           gamma: float = 0.5,
                           slots: int = None,
                           split: str = "all",
                           adaptation: str = None,
                           level: int = None,
                           num_aug: int = 1,
                           transforms=None,
                           ckpt: str = None
                           ) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    """
    기존 load_dataset 으로 Dataset 을 만든 뒤
    LabelDirichletSampler 를 적용해 Practical‑TTA DataLoader 반환
    """
    # 1) Dataset 객체 획득 (DataLoader는 버리고 Dataset만 사용)
    dataset, _ = load_dataset(dataset=dataset,
                              root=root,
                              batch_size=batch_size,
                              workers=workers,
                              split=split,
                              adaptation=adaptation,
                              domain=domain,
                              level=level,
                              ckpt=ckpt,
                              num_aug=num_aug,
                              transforms=transforms)

    # 2) Dataset 내부 샘플 리스트(List[DatumBase]) 추출
    if hasattr(dataset, "datasets"):  # ConcatDataset
        sample_list: List[DatumBase] = [
            dataset.datasets[i_dataset][i_sample]
            for i_dataset in range(len(dataset.datasets))
            for i_sample in range(len(dataset.datasets[i_dataset]))
        ]
    else:
        sample_list: List[DatumBase] = [dataset[i] for i in range(len(dataset))]

    # 3) Sampler 생성
    sampler = LabelDirichletSampler(
        data_source=sample_list,
        gamma=gamma,
        num_slots=slots
    )

    # 4) Practical‑TTA DataLoader
    practical_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True
    )

    return dataset, practical_loader