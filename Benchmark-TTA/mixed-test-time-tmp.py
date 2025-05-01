import logging
import os
import time

import numpy as np

from src.data import load_mixed_dataset
from src.methods import *
from src.models.load_model import load_model
from src.utils import get_accuracy, get_args, get_accuracy_with_figure, get_mixed_accuracy_with_figure
from src.utils.conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence

logger = logging.getLogger(__name__)


import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Union, Dict
from sklearn.manifold import TSNE

def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """
    문자열로 주어진 모듈 경로('model.model.block1' 등)를 따라가 
    해당 모듈 객체를 반환하는 보조 함수
    """
    parts = module_name.split(".")
    submodule = model
    for p in parts:
        submodule = getattr(submodule, p)
    return submodule


def get_mixed_accuracy_tSNE_with_figure(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    save_dir: Union[Path, str],
    device: Optional[torch.device] = None,
    make_plots: bool = True,
    module_names: Optional[List[str]] = None  # ← (추가) hook을 달 모듈들
) -> Dict[str, float]:
    """
    여러 도메인(domain)×severity가 섞여 있는 Mixed DataLoader에 대해,
    (img, label, domain_label) 형태로 데이터를 받아서:
      1) 도메인별 정확도 계산 + batch accuracy 추이
      2) 전체 데이터셋에 대한 accuracy
      3) 하나의 Figure에 도메인별 다른 색깔로 batch accuracy 그래프
      4) 하나의 Figure에 도메인별 다른 색깔로 pseudo-label 분포 (scatter)
      5) 하나의 Figure에 도메인별 다른 색깔로 ground-truth 분포 (scatter)
      6) CSV 저장 (모든 샘플에 대해 iteration, pred, label, domain_label)

    + (추가)
      - forward hook을 사용하여 module_names 모듈들의 출력(feature)을 
        배치마다 저장하고, domain별로 다른 색으로 t-SNE를 시각화

    최종적으로 
        acc_dict = {
            f"{domain_label}": domain별 accuracy,
            "all": 전체 accuracy
        }
    형태의 dict를 리턴한다.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ---------------- (1) Hook 등록을 위한 준비 ----------------
    features_dict = {}  # { module_name: [Tensor(batch_outputs), ...], ... }
    hooks = []

    def save_output_hook(module, input, output, key):
        # output: shape (B, feature_dim, ...)
        # 후에 torch.cat을 위해 CPU로 옮겨서 저장
        features_dict[key].append(output.detach().cpu())

    # module_names가 지정되어 있다면, 해당 모듈들에 hook 등록
    if module_names is not None:
        for mname in module_names:
            features_dict[mname] = []
            submodule = get_module_by_name(model, mname)
            h = submodule.register_forward_hook(
                lambda mod, inp, out, name=mname: save_output_hook(mod, inp, out, name)
            )
            hooks.append(h)

    # ---------------- (2) 도메인별 통계용 자료구조 ----------------
    domain_info = {}  # domain_label -> dict(...)
    all_data = {
        "iteration": [],
        "predicted_class": [],
        "ground_truth_class": [],
        "domain_label": []
    }

    # ---------------- (A) 배치 순회하며 도메인별 통계 수집 ----------------
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # (imgs, labels, domain_labels) 형태를 가정
            if len(batch_data) == 3:
                imgs, labels, domain_labels = batch_data
            else:
                raise ValueError("Mixed loader batch must be (imgs, labels, domain_labels).")

            # 모델 forward
            if isinstance(imgs, list):
                imgs = [img.to(device) for img in imgs]
                outputs = model(imgs)
            else:
                outputs = model(imgs.to(device))

            preds = outputs.argmax(dim=1)
            labels = labels.to(device)

            # 넘파이/리스트 변환
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            domain_labels = list(domain_labels)  # 예: ["domainA0", "domainB2", ...]

            # 전체(all) 샘플 레벨 정보 누적 -> 나중에 한 장의 scatter에서 사용
            bs = len(labels_cpu)
            all_data["iteration"].extend([batch_idx]*bs)
            all_data["predicted_class"].extend(preds_cpu.tolist())
            all_data["ground_truth_class"].extend(labels_cpu.tolist())
            all_data["domain_label"].extend(domain_labels)

            # 한 배치 안에 여러 도메인이 섞여 있을 수 있으므로,
            # 도메인별로 인덱스 그룹화해서 accuracy 집계
            unique_doms = set(domain_labels)
            for dlabel in unique_doms:
                idxs = [i for i, dl in enumerate(domain_labels) if dl == dlabel]
                count = len(idxs)
                if count == 0:
                    continue
                correct_count = (preds_cpu[idxs] == labels_cpu[idxs]).sum()

                # domain_info 초기화
                if dlabel not in domain_info:
                    domain_info[dlabel] = {
                        "batch_idxs": [],
                        "batch_accs": [],
                        "total_correct": 0,
                        "total_seen": 0
                    }
                domain_info[dlabel]["batch_idxs"].append(batch_idx)
                domain_info[dlabel]["batch_accs"].append(correct_count / count)
                domain_info[dlabel]["total_correct"] += correct_count
                domain_info[dlabel]["total_seen"]    += count

    # ---------------- (B) 도메인별 accuracy 계산, 전체 accuracy 계산 ----------------
    total_correct_all = 0
    total_seen_all = 0

    acc_dict = {}  # 최종 반환용 (domain_label -> accuracy)
    for dlabel, info in domain_info.items():
        d_correct = info["total_correct"]
        d_seen = info["total_seen"]
        d_acc = d_correct / d_seen if d_seen > 0 else 0.0
        acc_dict[dlabel] = d_acc

        total_correct_all += d_correct
        total_seen_all    += d_seen

    overall_acc = total_correct_all / total_seen_all if total_seen_all > 0 else 0.0
    acc_dict["all"] = overall_acc

    # ---------------- (C) 결과 저장(그래프 / CSV) ----------------
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if make_plots:
        # 컬러맵 설정: 도메인 수만큼 색상이 달라야 하므로
        domain_list = sorted(domain_info.keys())
        n_dom = len(domain_list)
        cmap = plt.cm.get_cmap('rainbow', n_dom)

        # ----- (1) Batch Accuracy Transition: 한 장에 도메인별 다른 색으로 plot -----
        fig_acc, ax_acc = plt.subplots(figsize=(6,4))
        for i, dlabel in enumerate(domain_list):
            info = domain_info[dlabel]
            color = cmap(i)
            ax_acc.plot(info["batch_idxs"], 
                        np.array(info["batch_accs"])*100.0, 
                        label=f"{dlabel} ({acc_dict[dlabel]*100:.2f}%)",
                        color=color, marker='o', markersize=3, linewidth=1)

        ax_acc.set_ylim(0, 105)
        ax_acc.set_xlabel("Batch Index")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Domain-wise Batch Accuracy Transition")
        ax_acc.legend(fontsize=8, loc="lower left", bbox_to_anchor=(1.0, 0))
        fig_acc.tight_layout()
        fig_acc.savefig(save_dir / "domain_batch_accuracy.jpg", dpi=300)
        plt.close(fig_acc)

        # ----- (2) pseudo-label 분포 (scatter) -----
        fig_pred, ax_pred = plt.subplots(figsize=(6,4))
        for i, dlabel in enumerate(domain_list):
            color = cmap(i)
            # 해당 도메인만 필터
            idxs = [idx for idx, dl in enumerate(all_data["domain_label"]) if dl == dlabel]
            x = [all_data["iteration"][j] for j in idxs]
            y = [all_data["predicted_class"][j] for j in idxs]
            ax_pred.scatter(x, y, s=4, alpha=0.6, label=dlabel, color=color)
        ax_pred.set_xlabel("Batch Index")
        ax_pred.set_ylabel("Predicted Class")
        ax_pred.set_title("Pseudo‑Label Distribution (color by domain)")
        ax_pred.legend(fontsize=8, loc="lower left", bbox_to_anchor=(1.0, 0))
        fig_pred.tight_layout()
        fig_pred.savefig(save_dir / "pseudo_label_scatter.jpg", dpi=300)
        plt.close(fig_pred)

        # ----- (3) ground-truth 분포 (scatter) -----
        fig_gt, ax_gt = plt.subplots(figsize=(6,4))
        for i, dlabel in enumerate(domain_list):
            color = cmap(i)
            # 해당 도메인만 필터
            idxs = [idx for idx, dl in enumerate(all_data["domain_label"]) if dl == dlabel]
            x = [all_data["iteration"][j] for j in idxs]
            y = [all_data["ground_truth_class"][j] for j in idxs]
            ax_gt.scatter(x, y, s=4, alpha=0.6, label=dlabel, color=color)
        ax_gt.set_xlabel("Batch Index")
        ax_gt.set_ylabel("Ground‑Truth Class")
        ax_gt.set_title("Ground‑Truth Class Distribution (color by domain)")
        ax_gt.legend(fontsize=8, loc="lower left", bbox_to_anchor=(1.0, 0))
        fig_gt.tight_layout()
        fig_gt.savefig(save_dir / "ground_truth_scatter.jpg", dpi=300)
        plt.close(fig_gt)

        # ----- (추가) (4) TSNE by domain: 각 모듈 출력(feature)에 대해 -----
        if module_names is not None:
            # 전체 도메인 라벨 목록(샘플 순서 그대로) → numpy array
            all_domains = np.array(all_data["domain_label"], dtype=object)

            for mname in module_names:
                # (B, feature_dim, ...) 형태 텐서들이 리스트에 쌓여있으므로, concat
                cat_features = torch.cat(features_dict[mname], dim=0)  # shape (N, ...)
                # 필요시 Flatten (ex. (N, C, H, W) -> (N, C×H×W))
                if len(cat_features.shape) > 2:
                    cat_features = cat_features.view(cat_features.size(0), -1)

                # t-SNE
                tsne = TSNE(n_components=2, random_state=0)
                cat_features_np = cat_features.numpy()
                embedding = tsne.fit_transform(cat_features_np)  # shape (N, 2)

                # scatter: domain별 다른 색
                fig_tsne, ax_tsne = plt.subplots(figsize=(5,5))
                unique_domains = sorted(np.unique(all_domains))
                n_doms = len(unique_domains)
                cmap_tsne = plt.cm.get_cmap('rainbow', n_doms)

                for idx_d, dlabel in enumerate(unique_domains):
                    color_d = cmap_tsne(idx_d)
                    sel = (all_domains == dlabel)
                    ax_tsne.scatter(embedding[sel, 0],
                                    embedding[sel, 1],
                                    s=4, alpha=0.6,
                                    color=color_d, label=dlabel)

                ax_tsne.set_title(f"{mname} domain-tSNE")
                ax_tsne.legend(fontsize=8, loc="best")
                fig_tsne.tight_layout()
                fig_tsne.savefig(save_dir / f"{mname}-domain_tSNE.jpg", dpi=300)
                plt.close(fig_tsne)

    # ----- (5) CSV (모든 샘플) 저장 -----
    df = pd.DataFrame(all_data)
    df.to_csv(save_dir / "all_samples.csv", index=False)

    # ----- (6) Hook 제거 -----
    for h in hooks:
        h.remove()

    # ----- (D) 결과 반환: domain별 acc + 전체 acc -----
    return acc_dict



def evaluate(cfg):
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = load_model(model_name=cfg.MODEL.ARCH, checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                            domain=cfg.CORRUPTION.SOURCE_DOMAIN)
    base_model = base_model.cuda()

    logger.info(f"Setting up test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    if cfg.MODEL.ADAPTATION == "source":  # BN--0
        model, param_names = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "t3a":
        model, param_names = setup_t3a(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "norm_test":  # BN--1
        model, param_names = setup_test_norm(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
        model, param_names = setup_alpha_norm(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "memo":
        model, param_names = setup_memo(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "tent":
        model, param_names = setup_tent(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "cotta":
        model, param_names = setup_cotta(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "lame":
        model, param_names = setup_lame(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "adacontrast":
        model, param_names = setup_adacontrast(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "eata":
        model, param_names = setup_eata(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "sar":
        model = setup_sar(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "deyo": # 내가 추가함 (deyo)
        model, param_names = setup_deyo(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "rotta": # 내가 추가함 (rotta)
        model, param_names = setup_rotta(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "tent_come": # 내가 추가함 (come)
        model, param_names = setup_tent_come(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "eata_come": # 내가 추가함 (come)
        model, param_names = setup_eata_come(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "sar_come": # 내가 추가함 (come)
        model = setup_sar_come(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "deyo_come": # 내가 추가함 (come)
        model = setup_deyo_come(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == 'memtta':
        model, param_names = setup_memtta(base_model, cfg, num_classes)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"domainnet126", "officehome"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.SOURCE_DOMAIN)
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = dom_names_all

    # setup the severities for the gradual setting

    severities = cfg.CORRUPTION.SEVERITY

    accs = []

    try:
        model.reset()
        logger.info("resetting model")
    except:
        logger.warning("not resetting model")
    mixedset, mixed_loader = load_mixed_dataset(
        cfg.CORRUPTION.DATASET,
        cfg.DATA_DIR,
        cfg.TEST.BATCH_SIZE,
        split='all',
        domain_names_loop=dom_names_loop,
        severities = severities,
        workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
        ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
        num_aug=cfg.TEST.N_AUGMENTATIONS,
        # 여기서 부터는 mxied_dataset을 위한 config
        imbalance_ratio=cfg.TEST.IMBALANCE_RATIO
    )
    for epoch in range(cfg.TEST.EPOCH):
        # bscho가 만듦
        from pathlib import Path
        log_path = Path(cfg.OUTPUT) / cfg.LOG_DEST
        log_dir = log_path.with_suffix("")
        log_dir.mkdir(parents=True, exist_ok=True)
                
        # 다시 원래 코드 (이 부분도 바꿔야할 듯?)
        module_names = ['model.block1', 'model.block2', 'model.block3']

        acc = get_mixed_accuracy_tSNE_with_figure(
            model=model,
            data_loader=mixed_loader,
            # domain_names_loop=dom_names_loop,
            # severities=severities,
            save_dir=log_dir,
            make_plots=True,
            module_names=module_names)
        if cfg.TEST.EPOCH > 1:
            print(f"epoch: {epoch}, acc: {acc:.2%}")
            # logger.info(f"epoch: {epoch}, acc: {acc:.2%}")

            accs.append(acc)

            # logger.info(
            #     f"{cfg.CORRUPTION.DATASET} accuracy % [{domain_name}{severity}][#samples={len(testset)}]: {acc:.2%}")

        logger.info(f"mean accuracy: {np.mean(accs):.2%}")
    return accs


if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'online_evaluation'
    load_cfg_fom_args(args.cfg, args.output_dir)
    logger.info(cfg)
    start_time = time.time()
    accs = []
    for domain in cfg.CORRUPTION.SOURCE_DOMAINS:
        logger.info("#" * 50 + f'evaluating domain {domain}' + "#" * 50)
        cfg.CORRUPTION.SOURCE_DOMAIN = domain
        acc = evaluate(cfg)
        accs.extend(acc)

    logger.info("#" * 50 + 'fianl result' + "#" * 50)
    logger.info(f"total mean accuracy: {np.mean(accs):.2%}")

    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time / 3600)
    minutes = int((run_time - hours * 3600) / 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    logger.info(f"total run time: {hours}h {minutes}m {seconds}s")
