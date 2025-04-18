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
MODEL_NAME = 'resnet50'
DATASET = 'imagenet-c'
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)  
# num_classes=0 => 최종 분류 레이어 제거 (임베딩만 뽑기)
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
save_root = f"results/tSNE/{DATASET}/{MODEL_NAME}/domain/"
os.makedirs(save_root, exist_ok=True)

# -----------------------------------------------------------
# 4) 실제로 존재하는 클래스 중 6개를 자동 선택하는 로직
#    (아래는 첫 corruption, 첫 severity만 이용해 샘플링)
# -----------------------------------------------------------
sample_corruption = corruptions[0]
sample_severity = severity_levels[0]
sample_data_path = os.path.join(f"./data/{DATASET}", sample_corruption, str(sample_severity))

# ImageFolder로 로드하여 존재하는 클래스(폴더) 이름을 확인
sample_dataset = datasets.ImageFolder(root=sample_data_path, transform=transform)
all_classes = sorted(list(sample_dataset.class_to_idx.keys()))
# 만약 class가 6개보다 적으면 에러가 날 수 있으므로, 실제로 6개 이상 있는지 확인
if len(all_classes) < 6:
    raise ValueError(f"폴더 내 클래스가 6개 미만입니다. 현재 클래스 수: {len(all_classes)}")

# 앞에서 6개 골라서 target_class_dict 만들기 (딱히 의미있는 이름이 없다면 그냥 폴더명 사용)
# selected_classes = all_classes[:6]
# target_class_dict = {cls_name: cls_name for cls_name in selected_classes}

target_class_dict = {
    'n01443537': 'Goldfish',
    'n02108915': 'French_bulldog',
    'n02124075': 'Egyptian_cat',
    'n02066245': 'grey_whale',
    "n02119789": 'kit_fox'
}



print("선택된 5개 클래스(폴더명) - (클래스명)")
for c in target_class_dict:
    print(f" - {c} - {target_class_dict[c]}")

# -----------------------------------------------------------
# 5) 각 corruption, severity마다 선택된 6개 클래스만 모아서 feature 추출 & 시각화
# -----------------------------------------------------------
for corruption in corruptions:
    print(f"\n[INFO] Processing corruption: {corruption}")
    
    # 클래스별 feature, label 모음
    all_features = []
    all_labels = []
    
    for severity in severity_levels:
        data_path = os.path.join("./data/imagenet-c", corruption, str(severity))
        
        # 전체 폴더(클래스)를 불러옴
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        
        # dataset.class_to_idx는 {폴더명: index} 형식
        # 우리가 선택한 6개 클래스에 해당하는 index만 필터링
        valid_class_idxs = []
        for class_name, class_idx in dataset.class_to_idx.items():
            if class_name in target_class_dict:
                valid_class_idxs.append(class_idx)
        
        # 유효 클래스에 해당하는 샘플만 고른다
        filtered_samples = [
            (path, idx) 
            for (path, idx) in dataset.samples 
            if idx in valid_class_idxs
        ]
        
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
        # feature 추출 (timm 모델: 분류 헤드를 제거했으므로 바로 임베딩이 나옴)
        # --------------------------------------------------
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(DEVICE)
                feats = model(images)  # model 자체가 임베딩을 출력
                # feats.shape: (batch_size, 2048) 예상 (ResNet50의 경우)
                
                all_features.append(feats.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    
    # 하나의 corruption 내, 모든 severity 레벨의 데이터를 합친다
    if len(all_features) == 0:
        print(" - 선택된 클래스에 해당하는 이미지가 없어서 스킵합니다.")
        continue

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # --------------------------------------------------
    # 6) t-SNE 적용 (2차원)
    # --------------------------------------------------
    print(" - Running t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, random_state=42, n_iter=10000)
    tsne_results = tsne.fit_transform(all_features)

    # --------------------------------------------------
    # 7) 시각화 (클래스별로 scatter plot)
    # --------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    # index -> class_name으로 역매핑
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    for class_name in target_class_dict:
        class_idx = dataset.class_to_idx[class_name]
        idxs = (all_labels == class_idx)
        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], 
                    label=target_class_dict[class_name], alpha=0.5, s=10)

    plt.legend()
    plt.title(f"t-SNE for corruption: {corruption} (6 selected classes)")
    
    # 결과 저장
    save_path = f"{save_root}/{cls_label}.jpg"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" - Saved t-SNE visualization to {save_path}")
