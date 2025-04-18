import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import timm
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
severity_levels = [1, 2, 3, 4, 5]

# -----------------------------------------------------------
# 1) timm을 이용해 모델 로드 (분류 헤드를 제거해 feature만 받도록 설정)
# -----------------------------------------------------------
DEVICE = 'cuda:5'  # 상황에 맞게 변경
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
# 3) 결과 저장 폴더 준비 (도메인 기준 시각화)
# -----------------------------------------------------------
save_root = f"results/PCA/{DATASET}/{MODEL_NAME}/domain/"
os.makedirs(save_root, exist_ok=True)

# -----------------------------------------------------------
# 4) 사용할 클래스(5개)를 예시로 지정
#    (폴더 이름 -> 사람이 이해하기 쉬운 라벨)
# -----------------------------------------------------------
target_class_dict = {
    'n01443537': 'Goldfish',
    'n02108915': 'French_bulldog',
    'n02124075': 'Egyptian_cat',
    'n02066245': 'grey_whale',
    'n02119789': 'kit_fox'
}
class_list = list(target_class_dict.keys())

# 시각화 시 클래스별 색깔 할당
class_colors = ['red', 'blue', 'green', 'orange', 'purple']
class_to_idx = {cls_id: i for i, cls_id in enumerate(class_list)}

print("선택된 클래스 목록:")
for cls_id, cls_label in target_class_dict.items():
    print(f" - {cls_id} -> {cls_label}")

# -----------------------------------------------------------
# 5) '각 도메인'별로 데이터 로드 -> PCA -> 시각화
#    (클래스를 색으로 표현)
# -----------------------------------------------------------
for domain_name in corruptions:
    print(f"\n[INFO] Processing domain: {domain_name}")
    
    # 모든 (embedding), (class) 레이블 저장용
    all_features = []
    all_classes = []  # (0 ~ len(target_class_dict)-1) 범위 정수
    
    # -------------------------------------------------------
    # 해당 도메인에서 모든 severity를 순회하면서 5개 클래스를 모은다
    # -------------------------------------------------------
    for severity in severity_levels:
        data_path = os.path.join(f"./data/{DATASET}", domain_name, str(severity))
        
        if not os.path.exists(data_path):
            # 해당 폴더가 없으면 스킵
            continue
        
        # ImageFolder 로 전체 이미지를 불러온다
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        
        # (데이터셋에 존재하는) 5개 클래스 각각에 대해 필터링
        for cls_id in target_class_dict.keys():
            if cls_id not in dataset.class_to_idx:
                # 해당 폴더가 없으면 스킵
                continue
            valid_class_idx_in_dataset = dataset.class_to_idx[cls_id]
            
            # 해당 클래스 샘플만 필터링
            filtered_samples = [
                (path, idx) for (path, idx) in dataset.samples
                if idx == valid_class_idx_in_dataset
            ]
            
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
            # feature 추출
            # --------------------------------------------------
            class_idx = class_to_idx[cls_id]  # 0~4 (5개 클래스)
            
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(DEVICE)
                    feats = model(images)  # (batch_size, feature_dim)
                    
                    all_features.append(feats.cpu().numpy())
                    # all_classes에 (batch_size만큼) 동일한 class_idx를 append
                    all_classes.append(np.full(len(labels), class_idx))
    
    # --------------------------------------------------------
    # 해당 도메인에서 모은 전체 데이터가 있으면 PCA 수행
    # --------------------------------------------------------
    if len(all_features) == 0:
        print(" - 해당 도메인에 해당 클래스 이미지가 하나도 없어서 스킵합니다.")
        continue

    all_features = np.concatenate(all_features, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)
    
    print(" - Running PCA...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(all_features)

    # --------------------------------------------------
    # 6) 시각화 (클래스별로 scatter 색 구분)
    # --------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    for cls_id, cls_label in target_class_dict.items():
        c_idx = class_to_idx[cls_id]
        color = class_colors[c_idx]
        
        # 해당 클래스에 해당하는 점들
        mask = (all_classes == c_idx)
        if np.sum(mask) == 0:
            continue
        
        plt.scatter(
            pca_results[mask, 0],
            pca_results[mask, 1],
            color=color,
            label=cls_label,
            alpha=0.6,
            s=10
        )

    plt.legend()
    plt.title(f"[{domain_name}] Domain PCA across different classes")
    
    # 결과 저장
    save_path = os.path.join(save_root, f"{domain_name}.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" - Saved PCA visualization to {save_path}")
