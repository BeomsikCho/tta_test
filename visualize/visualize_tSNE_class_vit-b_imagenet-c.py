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

# 사용할 부식 종류 (예시)
corruptions = [
    'gaussian_noise', 
    'defocus_blur', 
    'fog', 
    'pixelate', 
    'saturate'
]

# 예시로 사용할 부식 강도
severity_levels = [5]

# -----------------------------------------------------------
# 1) timm을 이용해 resnet50 모델 로드 (분류 헤드를 제거해 feature만 받도록 설정)
# -----------------------------------------------------------
DEVICE = 'cuda:6'  # 필요한 경우 다른 GPU 또는 'cpu'로 변경
MODEL_NAME = 'vit_base_patch16_224'
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
save_root = f"results/tSNE/{DATASET}/{MODEL_NAME}/class/"
os.makedirs(save_root, exist_ok=True)

# -----------------------------------------------------------
# 4) 실제로 존재하는 클래스 중 5개를 예시로 지정
#    (폴더 이름 -> 사람이 이해하기 쉬운 라벨)
# -----------------------------------------------------------
target_class_dict = {
    'n01443537': 'Goldfish',
    'n02108915': 'French_bulldog',
    'n02124075': 'Egyptian_cat',
    'n02066245': 'grey_whale',
    'n02119789': 'kit_fox'
}

# 도메인(corruption)별로 색을 구분할 때 사용하기 위해 index 할당
domain_to_idx = {corr: i for i, corr in enumerate(corruptions)}

print("선택된 클래스 목록:")
for cls_id, cls_label in target_class_dict.items():
    print(f" - {cls_id} -> {cls_label}")

# -----------------------------------------------------------
# 5) '각 클래스'별로 loop -> 데이터 로드 -> t-SNE
#    (도메인을 색으로 표현)
# -----------------------------------------------------------
for cls_id, cls_label in target_class_dict.items():
    print(f"\n[INFO] Processing class: {cls_id} ({cls_label})")
    
    # 모든 (embedding), (domain) 레이블 저장용
    all_features = []
    all_domains = []  # 정수로 표현 (0 ~ len(corruptions)-1)
    
    # -------------------------------------------------------
    # 모든 corruption과 severity를 순회하며, 해당 클래스 이미지를 모은다
    # -------------------------------------------------------
    for corruption in corruptions:
        for severity in severity_levels:
            data_path = os.path.join(f"./data/{DATASET}", corruption, str(severity))
            
            # ImageFolder로 전체 이미지를 불러온다
            dataset = datasets.ImageFolder(root=data_path, transform=transform)
            
            # 이 데이터셋에서 'cls_id' 폴더만 있는지 확인
            if cls_id not in dataset.class_to_idx:
                # 해당 부식, 해당 severity에 cls_id 폴더 자체가 없을 수 있음
                continue
            
            valid_class_idx = dataset.class_to_idx[cls_id]

            # 해당 클래스에만 해당하는 샘플을 골라낸다
            filtered_samples = [
                (path, idx) for (path, idx) in dataset.samples 
                if idx == valid_class_idx
            ]
            
            # 만약 이미지가 없다면 스킵
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
                def __getitem__(self, index):
                    path, label_idx = self.samples[index]
                    img = self.loader(path)
                    if self.transform:
                        img = self.transform(img)
                    return img, label_idx
            
            filtered_dataset = FilteredDataset(filtered_samples, transform=transform)
            
            # DataLoader
            dataloader = torch.utils.data.DataLoader(
                filtered_dataset, 
                batch_size=32, 
                shuffle=False, 
                num_workers=4
            )
            
            # --------------------------------------------------
            # feature 추출 (model: 분류 레이어 없이 임베딩 출력)
            # --------------------------------------------------
            domain_idx = domain_to_idx[corruption]
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(DEVICE)
                    feats = model(images)  
                    # feats.shape: (batch_size, 2048) 예상 (ResNet50 기준)
                    
                    all_features.append(feats.cpu().numpy())
                    # all_domains에 (batch_size만큼) domain_idx를 append
                    all_domains.append(np.full(len(labels), domain_idx))
    
    # --------------------------------------------------------
    # 해당 클래스에 대한 모든 data가 모여 있으면 t-SNE
    # --------------------------------------------------------
    if len(all_features) == 0:
        print(" - 해당 클래스 이미지가 하나도 없어서 스킵합니다.")
        continue

    all_features = np.concatenate(all_features, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)  # domain index

    print(" - Running t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, random_state=42, n_iter=10000)
    tsne_results = tsne.fit_transform(all_features)

    # --------------------------------------------------
    # 6) 시각화 (domain별로 scatter 색 구분)
    # --------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    for corruption in corruptions:
        domain_idx = domain_to_idx[corruption]
        idxs = (all_domains == domain_idx)
        plt.scatter(tsne_results[idxs, 0], 
                    tsne_results[idxs, 1], 
                    label=corruption, alpha=0.6, s=10)

    plt.legend()
    plt.title(f"[{cls_label}] Class t-SNE across different corruptions")
    
    # 결과 저장
    save_path = f"{save_root}/{cls_label}.jpg"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" - Saved t-SNE visualization to {save_path}")