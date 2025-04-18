import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import timm
from sklearn.manifold import TSNE
import numpy as np

from utils import PltPreset
PltPreset.base_plot_style()

# 사용할 부식(corruption) 종류 (예시)
corruptions = [
    'gaussian_noise', 
    'defocus_blur', 
    'fog', 
    'pixelate', 
    'saturate'
]

# 사용할 부식 심각도(severity)
severity_levels = [5]

# -----------------------------------------------------------
# 1) timm을 이용해 모델 로드 (최종 분류 레이어 제거)
# -----------------------------------------------------------
DEVICE = 'cuda:6'  # 상황에 맞게 변경
MODEL_NAME = 'resnet50'
DATASET = 'imagenet-c'
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
model = model.to(DEVICE)
model.eval()

# -----------------------------------------------------------
# 2) 이미지 전처리 정의
# -----------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------
# 3) 결과 저장 폴더 준비
# -----------------------------------------------------------
save_dir = f"results/tSNE/{DATASET}/{MODEL_NAME}"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------------------------------------
# 4) 사용할 클래스 5개를 예시로 지정
#    (폴더 이름 -> 사람이 이해하기 쉬운 라벨)
# -----------------------------------------------------------
target_class_dict = {
    'n01443537': 'Goldfish',
    'n02108915': 'French_bulldog',
    'n02124075': 'Egyptian_cat',
    'n02066245': 'grey_whale',
    'n02119789': 'kit_fox'
}

# (도메인 이름 -> index) 사전
domain_to_idx = {corr: i for i, corr in enumerate(corruptions)}

# (클래스 폴더명 -> index) 사전
class_list = list(target_class_dict.keys())  # ['n01443537', 'n02108915', ...]
class_to_idx2 = {cls_id: i for i, cls_id in enumerate(class_list)}

# 시각화에서 쓸 색상 (클래스용), 마커 (도메인용)
domain_colors = [
    'red', 'blue', 'green', 'purple', 'orange'
]
class_markers = [
    'o', '^', 's', 'v', 'P'
]

print("선택된 클래스 목록:")
for cls_id, cls_label in target_class_dict.items():
    print(f" - {cls_id} -> {cls_label}")

# -----------------------------------------------------------
# 5) 모든 도메인 × 모든 severity × 위의 클래스 5개 데이터를 한꺼번에 모으기
# -----------------------------------------------------------
all_features = []
all_domains = []
all_classes = []

for corruption in corruptions:
    for severity in severity_levels:
        data_path = os.path.join(f"./data/{DATASET}", corruption, str(severity))
        
        if not os.path.exists(data_path):
            # 해당 폴더가 없으면 스킵
            continue
        
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        
        # 우리가 관심있는 클래스(폴더)가 있는지 확인
        valid_cls_ids_in_this_set = set(dataset.class_to_idx.keys()).intersection(target_class_dict.keys())
        if len(valid_cls_ids_in_this_set) == 0:
            continue
        
        # 필터링
        filtered_samples = []
        for path, label_idx in dataset.samples:
            # folder_name = ...
            folder_name = list(dataset.class_to_idx.keys())[
                list(dataset.class_to_idx.values()).index(label_idx)
            ]
            if folder_name in target_class_dict:
                filtered_samples.append((path, label_idx))
        
        if len(filtered_samples) == 0:
            continue
        
        # 커스텀 Dataset
        class FilteredDataset(torch.utils.data.Dataset):
            def __init__(self, samples, transform=None):
                self.samples = samples
                self.transform = transform
                self.loader = dataset.loader
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                path, lbl = self.samples[idx]
                img = self.loader(path)
                if self.transform:
                    img = self.transform(img)
                return img, lbl
        
        filtered_dataset = FilteredDataset(filtered_samples, transform=transform)
        
        dataloader = torch.utils.data.DataLoader(
            filtered_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        domain_idx = domain_to_idx[corruption]
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(DEVICE)
                feats = model(images)  # (batch, 768) or (batch, 2048), 모델에 따라 다름
                feats = feats.cpu().numpy()
                labels = labels.cpu().numpy()
                
                for fvec, lbl in zip(feats, labels):
                    folder_name = list(dataset.class_to_idx.keys())[
                        list(dataset.class_to_idx.values()).index(lbl)
                    ]
                    cls_idx = class_to_idx2[folder_name]
                    
                    all_features.append(fvec)
                    all_domains.append(domain_idx)
                    all_classes.append(cls_idx)

all_features = np.array(all_features)
all_domains = np.array(all_domains)
all_classes = np.array(all_classes)

if len(all_features) == 0:
    print("수집된 데이터가 없습니다. 경로 또는 클래스 설정을 확인하세요.")
    exit()

# -----------------------------------------------------------
# 6) t-SNE
# -----------------------------------------------------------
print("\n[INFO] Running a single t-SNE on ALL data...")
tsne = TSNE(n_components=2, verbose=1, random_state=42, n_iter=10000)
tsne_results = tsne.fit_transform(all_features)

# -----------------------------------------------------------
# 7) 시각화 (색상: 클래스, 마커: 도메인)
#    => '클래스별 -> 도메인별' 순서로 그려서
#       범례가 '같은 색상'끼리 연속해서 나오게 함
# -----------------------------------------------------------
plt.figure(figsize=(10, 8))

# 새롭게 범례를 구성하기 위해 handle/label 묶음을 관리
# (scatter 반환값을 이용해 legend를 만들 수도 있지만,
#  여기서는 간단히 label만 넣어주어도 되므로 기본 방식을 사용)
for cls_id, c_idx in class_to_idx2.items():
    marker = class_markers[c_idx]
    cls_label = target_class_dict[cls_id]
    
    for dname, d_idx in domain_to_idx.items():
        color = domain_colors[d_idx]
        
        # 해당 클래스 & 도메인에 해당하는 점들
        mask = (all_classes == c_idx) & (all_domains == d_idx)
        
        if np.sum(mask) == 0:
            # 해당 조합 데이터가 없으면 스킵
            continue
        
        label_name = f"{cls_label}-{dname}"
        
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=color,
            marker=marker,
            label=label_name,
            alpha=0.7,
            s=10
        )

plt.title(f"All Classes & Domains in One t-SNE ({MODEL_NAME})")

# 이제 legend가 "Goldfish-gaussian_noise, Goldfish-shot_noise, ..." 식으로
# 동일 클래스(동일 색)끼리 연속해서 기록됨.
plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

save_path = os.path.join(save_dir, "all_in_one.jpg")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"[INFO] Saved t-SNE to {save_path}")
