from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

from utils import PltPreset, setup_deterministic

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import timm

class Visualizer(object):
    device = 'cuda:6'

    @staticmethod
    def prepare():
        # 재현성(determinism) 및 기본 plot 스타일 설정
        setup_deterministic(seed=2025)
        PltPreset.base_plot_style()

    @classmethod
    def visualize(
            cls,
            model='resnet50',
            dataset='imagenet-c',
            method='tSNE',
            focus='domain'
        ):
        cls.prepare()
        """
        Args:
            model   : ['resnet50', 'resnet50_gn', 'vit_base_patch16_224'] 중 선택
            dataset : ['imagenet-c', 'domainnet'] 중 선택
            method  : ['tSNE', 'PCA'] 중 선택
            focus   : ['domain', 'class', 'all'] 중 선택
        """
        # 간단한 에러 처리
        if model not in ['resnet50', 'resnet50_gn', 'vit_base_patch16_224']:
            print("[WARN] 아직 구현 안됨 - model:", model)
            return
        if dataset not in ['imagenet-c', 'domainnet']:
            print("[WARN] 아직 구현 안됨 - dataset:", dataset)
            return
        if method not in ['tSNE', 'PCA']:
            print("[WARN] 아직 구현 안됨 - method:", method)
            return
        if focus not in ['domain', 'class', 'all']:
            print("[WARN] 아직 구현 안됨 - focus:", focus)
            return

        if method == 'tSNE':
            cls.visualize_tSNE(model, dataset, focus)
        else:  # 'PCA'
            cls.visualize_PCA(model, dataset, focus)

    # ----------------------------------------------------------------
    # t-SNE
    # ----------------------------------------------------------------
    @classmethod
    def visualize_tSNE(cls, model_name, dataset_name, focus):
        print(f"\n[INFO] visualize_tSNE -> model={model_name}, dataset={dataset_name}, focus={focus}")

        # 모델 로드
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model = model.to(cls.device)
        model.eval()

        # 전처리
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 모든 (feature, domain, class) 한꺼번에 로드
        if dataset_name == 'imagenet-c':
            features, domains, classes, domain_list, class_list = cls._load_imagenet_c_data(
                model, transform
            )
        else:  # domainnet
            features, domains, classes, domain_list, class_list = cls._load_domainnet_data(
                model, transform
            )

        if len(features) == 0:
            print("[WARN] No data collected. Exiting.")
            return

        # t-SNE
        print(f"[INFO] Running t-SNE (focus={focus})...")
        tsne = TSNE(n_components=2, verbose=1, random_state=42, n_iter=2000)
        embeds_2d = tsne.fit_transform(features)

        # 시각화 함수
        cls._visualize_and_save(
            method='tSNE',
            dataset=dataset_name,
            model=model_name,
            focus=focus,
            embeds_2d=embeds_2d,
            domains=domains,
            classes=classes,
            domain_list=domain_list,
            class_list=class_list
        )

    # ----------------------------------------------------------------
    # PCA
    # ----------------------------------------------------------------
    @classmethod
    def visualize_PCA(cls, model_name, dataset_name, focus):
        print(f"\n[INFO] visualize_PCA -> model={model_name}, dataset={dataset_name}, focus={focus}")

        # 모델 로드
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model = model.to(cls.device)
        model.eval()

        # 전처리
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 데이터 로드
        if dataset_name == 'imagenet-c':
            features, domains, classes, domain_list, class_list = cls._load_imagenet_c_data(
                model, transform
            )
        else:  # domainnet
            features, domains, classes, domain_list, class_list = cls._load_domainnet_data(
                model, transform
            )

        if len(features) == 0:
            print("[WARN] No data collected. Exiting.")
            return

        # PCA
        print(f"[INFO] Running PCA (focus={focus})...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeds_2d = pca.fit_transform(features)

        # 시각화
        cls._visualize_and_save(
            method='PCA',
            dataset=dataset_name,
            model=model_name,
            focus=focus,
            embeds_2d=embeds_2d,
            domains=domains,
            classes=classes,
            domain_list=domain_list,
            class_list=class_list
        )

    # ----------------------------------------------------------------
    # (A) ImageNet-C 데이터 로드 -> (features, domain_idx, class_idx) 반환
    # ----------------------------------------------------------------
    @classmethod
    def _load_imagenet_c_data(cls, model, transform):
        """
        Returns:
          features: shape (N, D)
          domains:  shape (N,)  # domain index
          classes:  shape (N,)  # class index
          domain_list: list of domain names
          class_list : list of class names
        """
        # 예시 corruptions, severity, 클래스
        corruptions = [
            'gaussian_noise', 
            'shot_noise', 
            'impulse_noise', 
            'defocus_blur', 
            'glass_blur'
        ]
        severity_levels = [1, 2, 3, 4, 5]
        target_class_dict = {
            'n01443537': 'Goldfish',
            'n02108915': 'French_bulldog',
            'n02124075': 'Egyptian_cat',
            'n02066245': 'grey_whale',
            'n02119789': 'kit_fox'
        }
        # domain_list = corruptions
        # class_list  = list of keys in target_class_dict
        domain_list = corruptions
        class_list = list(target_class_dict.keys())

        domain_to_idx = {d: i for i, d in enumerate(domain_list)}
        class_to_idx = {c: i for i, c in enumerate(class_list)}

        all_feats = []
        all_domains = []
        all_classes = []

        root_dir = "./data/imagenet-c"
        DEVICE = next(model.parameters()).device

        for domain_name in domain_list:
            for severity in severity_levels:
                data_path = os.path.join(root_dir, domain_name, str(severity))
                if not os.path.exists(data_path):
                    continue

                dataset = datasets.ImageFolder(root=data_path, transform=transform)
                valid_set = set(dataset.class_to_idx.keys()).intersection(class_list)
                if len(valid_set) == 0:
                    continue

                # 필터링
                filtered_samples = []
                for (path, idx_label) in dataset.samples:
                    folder_name = list(dataset.class_to_idx.keys())[
                        list(dataset.class_to_idx.values()).index(idx_label)
                    ]
                    if folder_name in class_to_idx:
                        filtered_samples.append((path, idx_label))

                if len(filtered_samples) == 0:
                    continue

                class FilteredDataset(torch.utils.data.Dataset):
                    def __init__(self, samples, transform=None):
                        self.samples = samples
                        self.transform = transform
                        self.loader = dataset.loader
                    def __len__(self):
                        return len(self.samples)
                    def __getitem__(self, i):
                        path, l = self.samples[i]
                        img = self.loader(path)
                        if self.transform: 
                            img = self.transform(img)
                        return img, l

                fds = FilteredDataset(filtered_samples, transform=transform)
                loader = torch.utils.data.DataLoader(fds, batch_size=32, shuffle=False, num_workers=4)

                domain_idx = domain_to_idx[domain_name]

                with torch.no_grad():
                    for imgs, lbls in loader:
                        imgs = imgs.to(DEVICE)
                        feats = model(imgs)
                        feats = feats.cpu().numpy()
                        lbls = lbls.cpu().numpy()
                        for fv, lb in zip(feats, lbls):
                            folder_name = list(dataset.class_to_idx.keys())[
                                list(dataset.class_to_idx.values()).index(lb)
                            ]
                            cls_idx = class_to_idx[folder_name]
                            all_feats.append(fv)
                            all_domains.append(domain_idx)
                            all_classes.append(cls_idx)

        if len(all_feats) == 0:
            return np.array([]), np.array([]), np.array([]), domain_list, class_list

        return (np.array(all_feats), 
                np.array(all_domains), 
                np.array(all_classes),
                domain_list,
                class_list)

    # ----------------------------------------------------------------
    # (B) DomainNet 데이터 로드 -> (features, domain_idx, class_idx) 반환
    # ----------------------------------------------------------------
    @classmethod
    def _load_domainnet_data(cls, model, transform):
        domains = ["clipart","infograph","painting","quickdraw","real","sketch"]
        target_class_dict = {
            "apple": "apple",
            "bicycle": "bicycle",
            "car": "car",
            "dog": "dog",
            "mug": "mug"
        }
        domain_list = domains
        class_list = list(target_class_dict.keys())

        domain_to_idx = {d: i for i, d in enumerate(domain_list)}
        class_to_idx = {c: i for i, c in enumerate(class_list)}

        all_feats = []
        all_domains = []
        all_classes = []

        root_dir = "./data/domainnet"
        DEVICE = next(model.parameters()).device

        for dom_name in domain_list:
            domain_path = os.path.join(root_dir, dom_name)
            if not os.path.isdir(domain_path):
                continue

            dataset = datasets.ImageFolder(root=domain_path, transform=transform)
            valid_set = set(dataset.class_to_idx.keys()).intersection(class_list)
            if len(valid_set) == 0:
                continue

            # 필터링
            filtered_samples = []
            for (path, idx_label) in dataset.samples:
                folder_name = list(dataset.class_to_idx.keys())[
                    list(dataset.class_to_idx.values()).index(idx_label)
                ]
                if folder_name in class_list:
                    filtered_samples.append((path, idx_label))

            if len(filtered_samples) == 0:
                continue

            class FilteredDataset(torch.utils.data.Dataset):
                def __init__(self, samples, transform=None):
                    self.samples = samples
                    self.transform = transform
                    self.loader = dataset.loader
                def __len__(self):
                    return len(self.samples)
                def __getitem__(self, i):
                    path, l = self.samples[i]
                    img = self.loader(path)
                    if self.transform:
                        img = self.transform(img)
                    return img, l

            fds = FilteredDataset(filtered_samples, transform=transform)
            loader = torch.utils.data.DataLoader(fds, batch_size=32, shuffle=False, num_workers=4)

            dom_idx = domain_to_idx[dom_name]

            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs = imgs.to(DEVICE)
                    feats = model(imgs)
                    feats = feats.cpu().numpy()
                    lbls = lbls.cpu().numpy()

                    for fv, lb in zip(feats, lbls):
                        folder_name = list(dataset.class_to_idx.keys())[
                            list(dataset.class_to_idx.values()).index(lb)
                        ]
                        cls_idx = class_to_idx[folder_name]
                        all_feats.append(fv)
                        all_domains.append(dom_idx)
                        all_classes.append(cls_idx)

        if len(all_feats) == 0:
            return np.array([]), np.array([]), np.array([]), domain_list, class_list

        return (np.array(all_feats),
                np.array(all_domains),
                np.array(all_classes),
                domain_list,
                class_list)

    # ----------------------------------------------------------------
    # 시각화 & 저장: 
    #   focus='domain' -> 도메인별로 별도 figure (각 figure에는 해당 domain의 모든 클래스)
    #   focus='class'  -> 클래스별로 별도 figure (각 figure에는 해당 클래스의 모든 도메인)
    #   focus='all'    -> 하나의 figure (도메인×클래스 전부)
    # ----------------------------------------------------------------
    @classmethod
    def _visualize_and_save(cls,
                            method,
                            dataset,
                            model,
                            focus,
                            embeds_2d,
                            domains,
                            classes,
                            domain_list,
                            class_list):
        """
        embeds_2d : shape (N, 2)
        domains   : shape (N,)
        classes   : shape (N,)
        domain_list: e.g. ['gaussian_noise', ...] or ['clipart', ...]
        class_list : e.g. ['n01443537', ...] or ['apple', ...]
        """
        # 색상/마커
        domain_colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink'
        ]
        class_colors = [
            'cyan', 'magenta', 'gold', 'lime', 'navy', 'orange', 'grey'
        ]
        # marker는 도메인/클래스 구분 시 여러 개 필요할 수도 있음
        markers = ['o','^','s','v','P','D','X','>','<']

        # 결과 저장 루트
        # focus마다 폴더(subdir) 달라짐. ('domain', 'class', 'all')
        # 단, 아래 요구사항에서는 폴더를 더 세분화해서,
        #   results/{method}/{dataset}/{model}/domain/
        #   results/{method}/{dataset}/{model}/class/
        #   results/{method}/{dataset}/{model}/all.jpg
        # 형태로 저장해야 한다.
        base_save_dir = f"results/{method}/{dataset}/{model}"
        os.makedirs(base_save_dir, exist_ok=True)

        if focus == 'domain':
            # 도메인마다 하나씩 그림
            out_dir = os.path.join(base_save_dir, "domain")
            os.makedirs(out_dir, exist_ok=True)

            # domain idx -> 실제 데이터들만
            for d_idx, d_name in enumerate(domain_list):
                mask_domain = (domains == d_idx)
                if np.sum(mask_domain) == 0:
                    # 해당 도메인 데이터 없음
                    continue

                # 이 도메인에 속한 (x,y)
                x = embeds_2d[mask_domain, 0]
                y = embeds_2d[mask_domain, 1]
                c_classes = classes[mask_domain]  # shape (?)
                
                plt.figure(figsize=(8,6))
                # 클래스별로 색/marker 다르게
                for c_idx, c_name in enumerate(class_list):
                    mask_class = (c_classes == c_idx)
                    if np.sum(mask_class) == 0:
                        continue
                    plt.scatter(
                        x[mask_class],
                        y[mask_class],
                        c=class_colors[c_idx % len(class_colors)],
                        marker=markers[c_idx % len(markers)],
                        label=c_name,  # 폴더명 그대로 or custom
                        alpha=0.7,
                        s=12
                    )
                plt.title(f"{dataset} {method} - domain [{d_name}] (model={model})")
                plt.legend(fontsize=8)
                plt.tight_layout()

                # 저장 경로: f"results/{method}/{dataset}/{model}/domain/{domain_name}.jpg"
                save_path = os.path.join(out_dir, f"{d_name}.jpg")
                plt.savefig(save_path, dpi=300)
                plt.close()

        elif focus == 'class':
            # 클래스마다 하나씩 그림
            out_dir = os.path.join(base_save_dir, "class")
            os.makedirs(out_dir, exist_ok=True)

            for c_idx, c_name in enumerate(class_list):
                mask_class = (classes == c_idx)
                if np.sum(mask_class) == 0:
                    continue

                x = embeds_2d[mask_class, 0]
                y = embeds_2d[mask_class, 1]
                c_domains = domains[mask_class]

                plt.figure(figsize=(8,6))
                # 도메인별로 색/marker
                for d_idx, d_name in enumerate(domain_list):
                    mask_dom = (c_domains == d_idx)
                    if np.sum(mask_dom) == 0:
                        continue
                    plt.scatter(
                        x[mask_dom],
                        y[mask_dom],
                        c=domain_colors[d_idx % len(domain_colors)],
                        marker=markers[d_idx % len(markers)],
                        label=d_name,
                        alpha=0.7,
                        s=12
                    )
                plt.title(f"{dataset} {method} - class [{c_name}] (model={model})")
                plt.legend(fontsize=8)
                plt.tight_layout()

                # 저장 경로: f"results/{method}/{dataset}/{model}/class/{class_name}.jpg"
                save_path = os.path.join(out_dir, f"{c_name}.jpg")
                plt.savefig(save_path, dpi=300)
                plt.close()

        else:  # focus == 'all'
            # 하나의 그림에 전부
            plt.figure(figsize=(10,8))
            # color=class, marker=domain (혹은 반대로) 해도 됨
            # 여기서는 class=color, domain=marker 예시
            N = len(embeds_2d)
            x = embeds_2d[:, 0]
            y = embeds_2d[:, 1]

            for c_idx, c_name in enumerate(class_list):
                # mask_class
                mask_c = (classes == c_idx)
                if np.sum(mask_c) == 0:
                    continue
                for d_idx, d_name in enumerate(domain_list):
                    mask_dom = mask_c & (domains == d_idx)
                    if np.sum(mask_dom) == 0:
                        continue
                    plt.scatter(
                        x[mask_dom],
                        y[mask_dom],
                        c=class_colors[c_idx % len(class_colors)],
                        marker=markers[d_idx % len(markers)],
                        label=f"{c_name}-{d_name}",
                        alpha=0.7,
                        s=12
                    )
            plt.title(f"{dataset} {method} - all (model={model})")
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # 저장 경로: f"results/{method}/{dataset}/{model}/all.jpg"
            save_path = os.path.join(base_save_dir, "all.jpg")
            plt.savefig(save_path, dpi=300)
            plt.close()

        print(f"[INFO] Saved {focus} visualization under results/{method}/{dataset}/{model}/")

# ----------------------------------------------------------------
# 실행 예시 (필요하다면 main문 사용)
# ----------------------------------------------------------------
if __name__ == "__main__":
    model_names = ['resnet50', 'resnet50_gn', 'vit_base_patch16_224']
    data_names = ['imagenet-c', 'domainnet']
    method_names = ['tSNE', 'PCA']
    focus_types = ['domain', 'class', 'all']

    for model in model_names:
        for data in data_names:
            for method in method_names:
                for focus in focus_types:
                    Visualizer.visualize(model, data, method, focus)