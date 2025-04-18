import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

def reproduce_from_csv(
    save_dir: Path | str,
    domain_names: List[str],
    severities: List[int | str],
    output_name: str = "combined_timeline.jpg",
    dpi: int = 500
):
    """
    CSV 결과를 이용해 하나의 Online Batch 축으로 예측/정답 분포를 재현.

    Parameters
    ----------
    save_dir : str | Path
        get_accuracy_with_figure() 에서 저장했던 최상위 폴더
    domain_names : list[str]
        예: ["glass_blur", "snow"]
    severities : list[int|str]
        예: [5, 3]
    output_name : str
        저장될 파일명 (폴더는 save_dir)
    dpi : int
        그림 해상도
    """
    save_dir = Path(save_dir)
    assert len(domain_names) == len(severities), "도메인과 severity 개수가 다릅니다."

    global_iters   = []
    global_preds   = []
    global_truths  = []
    boundary_xpos  = []   # 도메인 경계 위치
    boundary_labels = []  # annotation 텍스트

    offset = 0
    for dname, sev in zip(domain_names, severities):
        csv_path = save_dir / f"{dname}{sev}" / f"{dname}{sev}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV가 없습니다: {csv_path}")

        df = pd.read_csv(csv_path)

        # iteration 보정(누적)
        iters = df["iteration"] + offset
        global_iters.extend(iters)
        global_preds.extend(df["predicted_class"])
        global_truths.extend(df["ground_truth_class"])

        # 다음 도메인 시작 지점 (현재 도메인 마지막 + 1)
        offset = iters.max() + 1
        boundary_xpos.append(offset)
        boundary_labels.append(f"{dname}{sev}")

    # 마지막 도메인의 끝은 경계선 불필요 → pop
    boundary_xpos.pop()            # 맨 끝 제거
    boundary_labels.pop()

    # ---------- 그림 ----------
    FIGSIZE = (10, 4)   # 가로로 좀 더 넓게
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.scatter(global_iters, global_preds,   s=4, alpha=0.6, label="Predicted")
    ax.scatter(global_iters, global_truths,  s=4, alpha=0.6, label="Ground Truth", c="tab:green")

    # 도메인 경계선과 annotation
    ymin, ymax = ax.get_ylim()
    for xpos, label in zip(boundary_xpos, boundary_labels):
        ax.axvline(x=xpos, linestyle='--', color='gray', linewidth=1)
        ax.text(xpos, ymax - 0.05*(ymax - ymin), label, rotation=90,
                va='top', ha='right', fontsize=8, color='gray')

    ax.set_xlabel("Online Batch (cumulative)")
    ax.set_ylabel("Class Index")
    ax.set_title("Domain Sequence over Online Batches")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = save_dir / output_name
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"✔️  그림 저장 완료: {out_path}")




if __name__ == "__main__":
    save_dir = "./Benchmark-TTA/output/test-time-evaluation/cifar10_c/tent/tent250418_193706_62003"
    domain_names = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_sblur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
    severity = [5] * len(domain_names)

    reproduce_from_csv(
        save_dir=save_dir,
        domain_names=domain_names,
        severities=severity,
        output_name="mixed_domains_timeline.jpg"
    )