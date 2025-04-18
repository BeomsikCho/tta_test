import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import List

from visualize.utils import PltPreset


def reproduce_from_csv(
    save_dir: Path | str,
    domain_names: List[str],
    severities: List[int | str],
    output_name: str = "combined_timeline",
    dpi: int = 500
):
    PltPreset.base_plot_style()

    save_dir = Path(save_dir)
    assert len(domain_names) == len(severities), "도메인과 severity 길이가 다릅니다."

    segments = []          # [(iters, preds, truths, label), ...]
    boundary_xpos = []     # 각 도메인 시작 iteration

    offset = 0
    for dname, sev in zip(domain_names, severities):
        csv_path = save_dir / f"{dname}{sev}" / f"{dname}{sev}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV가 없습니다: {csv_path}")

        df = pd.read_csv(csv_path)
        iters = df["iteration"] + offset
        label = f"{dname}{sev}"

        segments.append(
            (iters.to_numpy(),
             df["predicted_class"].to_numpy(),
             df["ground_truth_class"].to_numpy(),
             label)
        )

        boundary_xpos.append(offset)     # 도메인 시작점 기록
        offset = iters.max() + 1

    # 색상 팔레트 (tab20 → 최대 20개 후 반복)
    palette = list(cm.get_cmap('tab20').colors)
    color_for = lambda idx: palette[idx % len(palette)]

    # batch accuracy 계산
    all_df = pd.concat(
        [pd.DataFrame({"iteration": seg[0], "pred": seg[1], "truth": seg[2]})
         for seg in segments],
        ignore_index=True
    )
    acc_per_iter = (
        all_df.groupby("iteration")
              .apply(lambda x: (x["pred"] == x["truth"]).mean() * 100.0)
              .sort_index()
    )

    # x‑tick 위치: 0 과 모든 도메인 시작 iteration (경계)
    xticks = boundary_xpos + [segments[-1][0].max()]  # 마지막 끝도 tick
    xtick_labels = [str(x) for x in xticks]

    # 세로 점선 함수
    def draw_boundaries(ax):
        for xpos in boundary_xpos[1:]:     # 첫 번째(0)는 제외
            ax.axvline(x=xpos, linestyle='--', color='gray', linewidth=1)

    FIGSIZE = (10, 4)

    # ---- 1) pseudo‑label ---------------------------------
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    for idx, (iters, preds, _, label) in enumerate(segments):
        ax1.scatter(iters, preds, s=4, alpha=0.6,
                    color=color_for(idx), label=label)
    draw_boundaries(ax1)
    ax1.set_xticks(xticks, xtick_labels, rotation=0, fontsize=7)
    ax1.set_xlabel("Online Batch (Cumulative)")
    ax1.set_ylabel("Predicted Class")
    ax1.set_title("Pseudo‑Label Distribution")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig1.tight_layout()
    fig1.savefig(save_dir / f"{output_name}_pseudo_label.jpg", dpi=dpi)
    plt.close(fig1)

    # ---- 2) ground‑truth ---------------------------------
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    for idx, (iters, _, truths, label) in enumerate(segments):
        ax2.scatter(iters, truths, s=4, alpha=0.6,
                    color=color_for(idx), label=label)
    draw_boundaries(ax2)
    ax2.set_xticks(xticks, xtick_labels, rotation=0, fontsize=7)
    ax2.set_xlabel("Online Batch (Cumulative)")
    ax2.set_ylabel("Ground‑Truth Class")
    ax2.set_title("Ground‑Truth Distribution")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig2.tight_layout()
    fig2.savefig(save_dir / f"{output_name}_ground_truth.jpg", dpi=dpi)
    plt.close(fig2)

    # ---- 3) batch accuracy -------------------------------
    fig3, ax3 = plt.subplots(figsize=FIGSIZE)
    for idx, (iters, _, _, _) in enumerate(segments):
        domain_acc = acc_per_iter.loc[iters]
        ax3.plot(iters, domain_acc,
                 color=color_for(idx), linewidth=1.2,
                 label=segments[idx][3])
    ax3.set_ylim(0, 100)
    draw_boundaries(ax3)
    ax3.set_xticks(xticks, xtick_labels, rotation=0, fontsize=7)
    ax3.set_xlabel("Online Batch (Cumulative)")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Batch Accuracy over Online Batches")
    ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig3.tight_layout()
    fig3.savefig(save_dir / f"{output_name}_accuracy.jpg", dpi=dpi)
    plt.close(fig3)

    print("✔️  모든 그림 저장 완료")



if __name__ == "__main__":
    # domain_names = {
    #     'cifar10_c': ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    #     'cifar100_c': ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    #     'imagenet_c': ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    #     'imagenet_c_vit': ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    #     'officehome': ["Clipart", "Painting", "Sketch", "Real"]
    # }

    # method_names=["tent", "eata", "sar", "t3a", "source", "lame", "memo", "norm", "cotta", "rotta"]
    # dataset_names=["cifar10_c", "cifar100_c", "officehome", "imagenet_c", "imagenet_c_vit", "imagenet_convnet"]

    # for dataset_name in dataset_names:
    #     for method_name in method_names:
    #         save_dir = f"./Benchmark-TTA/output/test-time-evaluation/{dataset_name}/{method_name}/tent250418_201201_55159"
    #         domain_name = domain_names[dataset_name]
    #         severity = [5] * len(domain_names) # 이것도 office-home에 맞도록 나중에 수정
            
    #         reproduce_from_csv(
    #             save_dir=save_dir,
    #             domain_names=domain_names,
    #             severities=severity,
    #             output_name="mixed_domains_timeline.jpg"
    #         )


    save_dir = "./Benchmark-TTA/output/test-time-evaluation/cifar10_c/tent/tent250418_201201_55159"
    domain_names = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
    severity = [5] * len(domain_names)

    reproduce_from_csv(
        save_dir=save_dir,
        domain_names=domain_names,
        severities=severity,
        output_name="mixed_domains_timeline.jpg"
    )