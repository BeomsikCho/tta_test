import logging
import os
import time

import numpy as np

from src.methods import *
from src.models.load_model import load_model
from src.utils import get_accuracy, get_args, get_accuracy_with_figure
from src.utils.conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence

logger = logging.getLogger(__name__)


import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
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

def get_accuracy_tSNE_with_figure(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    domain_name: str,                         # ex) "glass_blur"
    severity: int | str,                      # ex) 5
    save_dir: Path | str,                     # ex) Path("./results")
    device: Optional[torch.device] = None,
    make_plots: bool = True,
    module_names: Optional[List[str]] = None  # 추가된 인자
):
    """
    · 전체 accuracy 계산
    · Figure 1: 배치별 평균 정확도 (선 그래프, 0~100 %, 4×4 inch, 500 DPI)
    · Figure 2: pseudo‑label 분포 (scatter)
    · Figure 3: ground‑truth 분포 (scatter)
    · CSV: iteration, predicted_class, ground_truth_class 저장

    · (추가) module hook을 통해 지정된 module_names의 intermediate feature를 수집하고
      전체 데이터셋 처리 후 t‑SNE로 시각화하여
      {module_name}-tSNE.jpg 파일로 저장
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ------ (1) Hook 등록을 위한 준비 ------
    # features_dict: { module_name: [tensor(batch_outputs), ...], ... }
    features_dict = {}
    hooks = []

    # Hook 함수: forward에서 해당 모듈의 출력을 저장
    def save_output_hook(module, input, output, key):
        # output: shape (B, feature_dim, ...)
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

    # ------ (2) Accuracy / 통계 계산 ------
    total_correct, total_seen = 0, 0
    batch_accs: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_iters: list[int] = []

    with torch.no_grad():
        for itr, (imgs, labels) in enumerate(data_loader):
            if isinstance(imgs, list):
                imgs = [img.to(device) for img in imgs]
                outputs = model(imgs)
            else:
                outputs = model(imgs.to(device))

            preds  = outputs.argmax(dim=1)
            labels = labels.to(device)

            correct_this = (preds == labels).float().sum().item()
            bs = labels.size(0)

            total_correct += correct_this
            total_seen    += bs
            batch_accs.append(correct_this / bs)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_iters.extend([itr] * bs)

    accuracy = total_correct / total_seen

    # ------ (3) Hook 제거 ------
    for h in hooks:
        h.remove()

    # ------ (4) 저장 경로 설정 ------
    save_dir = Path(save_dir)
    target_dir = save_dir / f"{domain_name}{severity}"
    target_dir.mkdir(parents=True, exist_ok=True)

    FIGSIZE = (4, 4)
    DPI = 500
    title = getattr(data_loader, "name", "undefined_loader")

    # ------ (5) 기존 figure 들 (accuracy 등) ------
    if make_plots:
        # Figure 1: 배치별 평균 정확도
        fig1, ax1 = plt.subplots(figsize=FIGSIZE)
        ax1.plot(np.arange(len(batch_accs)),
                 np.array(batch_accs) * 100.0,
                 linewidth=1.5)
        ax1.set_ylim(0, 100)
        ax1.set_title(title)
        ax1.set_xlabel("Online Batch")
        ax1.set_ylabel("Average Accuracy (%)")
        fig1.tight_layout()
        fig1.savefig(target_dir / "accuracy_transition.jpg", dpi=DPI)
        plt.close(fig1)

        # Figure 2: pseudo‑label 분포
        fig2, ax2 = plt.subplots(figsize=FIGSIZE)
        ax2.scatter(all_iters, all_preds, s=3, alpha=0.6)
        ax2.set_title(title)
        ax2.set_xlabel("Online Batch")
        ax2.set_ylabel("Predicted Class")
        fig2.tight_layout()
        fig2.savefig(target_dir / "pseudo_label.jpg", dpi=DPI)
        plt.close(fig2)

        # Figure 3: ground‑truth 분포
        fig3, ax3 = plt.subplots(figsize=FIGSIZE)
        ax3.scatter(all_iters, all_labels, s=3, alpha=0.6, c="tab:green")
        ax3.set_title(title)
        ax3.set_xlabel("Online Batch")
        ax3.set_ylabel("Ground‑Truth Class")
        fig3.tight_layout()
        fig3.savefig(target_dir / "ground_truth.jpg", dpi=DPI)
        plt.close(fig3)

    # ------ (6) (추가) t-SNE 시각화 ------
    # module_names가 있다면, 해당 모듈 feature들에 대해 t-SNE
    if module_names is not None and make_plots:
        all_labels_np = np.array(all_labels)
        unique_labels = np.unique(all_labels_np)

        for mname in module_names:
            # features_dict[mname]에 (B, ...) 형식의 텐서 여러 개가 들어있음
            # -> (N, ...) 로 이어붙이기
            all_features = torch.cat(features_dict[mname], dim=0)

            # 만약 채널수가 많으면 (예: (N, C, H, W)) -> Flatten 등 필요
            # 여기서는 단순히 (N, feature_dim)이라고 가정
            if len(all_features.shape) > 2:
                all_features = all_features.view(all_features.size(0), -1)

            # t-SNE 변환
            tsne = TSNE(n_components=2, random_state=0)
            all_features_np = all_features.numpy()
            embedding = tsne.fit_transform(all_features_np)

            # 시각화
            fig_tsne, ax_tsne = plt.subplots(figsize=(5, 5))
            # label별 다른 색깔
            for lab in unique_labels:
                idx = (all_labels_np == lab)
                ax_tsne.scatter(
                    embedding[idx, 0], embedding[idx, 1],
                    s=4, alpha=0.6, label=f"Class {lab}"
                )
            ax_tsne.set_title(f"{mname} t-SNE")
            ax_tsne.legend(loc="best")
            fig_tsne.tight_layout()

            # 저장
            fig_tsne.savefig(target_dir / f"{mname}-tSNE.jpg", dpi=DPI)
            plt.close(fig_tsne)

    # ------ (7) CSV 저장 ------
    df = pd.DataFrame({
        "iteration": all_iters,
        "predicted_class": all_preds,
        "ground_truth_class": all_labels
    })
    df.to_csv(target_dir / f"{domain_name}{severity}.csv", index=False)

    return accuracy
















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

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        try:
            model.reset()
            logger.info("resetting model")
        except:
            logger.warning("not resetting model")

        for severity in severities:
            testset, test_loader = load_dataset(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                                cfg.TEST.BATCH_SIZE,
                                                split='all', domain=domain_name, level=severity,
                                                adaptation=cfg.MODEL.ADAPTATION,
                                                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                                ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                                num_aug=cfg.TEST.N_AUGMENTATIONS)

            for epoch in range(cfg.TEST.EPOCH):
                # bscho가 만듦
                from pathlib import Path
                log_path = Path(cfg.OUTPUT) / cfg.LOG_DEST
                log_dir = log_path.with_suffix("")
                log_dir.mkdir(parents=True, exist_ok=True)

                test_loader.name = f"{cfg.CORRUPTION.DATASET}-{domain_name}{severity}"
                
                # tSNE때문에 추가된 부분
                module_names = ['model.block1', 'model.block2', 'model.block3']


                # 다시 원래 코드
                acc = get_accuracy_tSNE_with_figure(
                    model=model,
                    data_loader=test_loader,
                    domain_name = domain_name,
                    severity=severity,
                    save_dir=log_dir,
                    make_plots=True,
                    module_names=module_names)
                if cfg.TEST.EPOCH > 1:
                    print(f"epoch: {epoch}, acc: {acc:.2%}")
                    # logger.info(f"epoch: {epoch}, acc: {acc:.2%}")


            accs.append(acc)

            logger.info(
                f"{cfg.CORRUPTION.DATASET} accuracy % [{domain_name}{severity}][#samples={len(testset)}]: {acc:.2%}")

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
