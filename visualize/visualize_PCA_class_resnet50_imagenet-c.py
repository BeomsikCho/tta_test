import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import timm
# from sklearn.manifold import TSNE  # t-SNE는 사용하지 않으므로 주석 처리
from sklearn.decomposition import PCA
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

# 예시로 사용할 부식 강도(severity)
severity_levels = [5]

# -----------------------------------------------------------
# 1) timm을 이용해 모델 로드 (분류 헤드를 제거해 feature만 받도록 설정)
# -----------------------------------------------------------
DEVICE = 'cuda:5'  # 상황에 맞게 변경 ('cuda:0' 또는 'cpu' 등)
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
#    (t-SNE -> PCA로 대체, 하위 폴더명만 변경)
# -----------------------------------------------------------
save_root = f"results/PCA/{DATASET}/{MODEL_NAME}/class/"
os.makedirs(save_root, exist_ok=True)

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

# 도메인(corruption)별로 color를 구분하기 위해 인덱스 부여
domain_to_idx = {corr: i for i, corr in enumerate(corruptions)}
# 색상이나 마커가 필요하면 미리 정의해도 됨 (예시로 색상 리스트)
domain_colors = ['red', 'blue', 'green', 'orange', 'purple']  # corruption 수와 맞춰야 함

print("선택된 클래스 목록:")
for cls_id, cls_label in target_class_dict.items():
    print(f" - {cls_id} -> {cls_label}")

# -----------------------------------------------------------
# 5) '각 클래스'별로 loop -> 데이터 로드 -> PCA
#    (domain을 색으로 표현)
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
            
            if not os.path.exists(data_path):
                # 해당 폴더가 없으면 스킵
                continue

            dataset = datasets.ImageFolder(root=data_path, transform=transform)
            
            # 이 데이터셋에서 'cls_id' 폴더가 존재하는지 확인
            if cls_id not in dataset.class_to_idx:
                # 폴더가 없는 경우 스킵
                continue
            
            valid_class_idx = dataset.class_to_idx[cls_id]

            # 해당 클래스에만 해당하는 샘플을 골라낸다
            filtered_samples = [
                (path, idx) for (path, idx) in dataset.samples 
                if idx == valid_class_idx
            ]
            
            if len(filtered_samples) == 0:
                # 실제 이미지가 없으면 스킵
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
                    feats = model(images)  # (batch_size, feature_dim)
                    
                    all_features.append(feats.cpu().numpy())
                    all_domains.append(np.full(len(labels), domain_idx))
    
    # --------------------------------------------------------
    # 해당 클래스의 모든 data가 모여 있으면 PCA 진행
    # --------------------------------------------------------
    if len(all_features) == 0:
        print(" - 해당 클래스 이미지가 하나도 없어서 스킵합니다.")
        continue

    all_features = np.concatenate(all_features, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)  # domain index

    print(" - Running PCA...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(all_features)

    # --------------------------------------------------
    # 6) 시각화 (domain별로 scatter 색 구분)
    # --------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    for i, corruption in enumerate(corruptions):
        domain_idx = domain_to_idx[corruption]
        idxs = (all_domains == domain_idx)

        # domain_colors가 corruptions 개수만큼 정의됐다고 가정
        color = domain_colors[domain_idx]  
        
        if np.sum(idxs) == 0:
            continue  # 해당 도메인 데이터 없으면 스킵
        
        plt.scatter(
            pca_results[idxs, 0],
            pca_results[idxs, 1],
            label=corruption,
            color=color,
            alpha=0.6,
            s=10
        )

    plt.legend()
    plt.title(f"[{cls_label}] Class PCA across different corruptions")
    
    # 결과 저장
    save_path = os.path.join(save_root, f"{cls_label}.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" - Saved PCA visualization to {save_path}")
