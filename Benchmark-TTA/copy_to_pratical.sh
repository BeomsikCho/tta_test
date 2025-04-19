#!/usr/bin/env bash
# best_cfgs/Online_TTA → best_cfgs/Practical_TTA 로 복사하면서
# TEST 블록에 GAMMA:0.3, SLOTS:10 추가

set -euo pipefail

SRC_ROOT="./best_cfgs/Online_TTA"
DST_ROOT="./best_cfgs/Practical_TTA"

datasets=(
  cifar10_c cifar100_c domainnet126
  imagenet_c imagenet_c_vit imagenet_convnet
  imagenet_efn officehome
)

methods=(
  source.yaml tent.yaml eata.yaml sar.yaml cotta.yaml
  adacontrast.yaml t3a.yaml memo.yaml rotta.yaml lame.yaml
)

for dataset in "${datasets[@]}"; do
  for cfg in "${methods[@]}"; do
    src="$SRC_ROOT/$dataset/$cfg"
    dst_dir="$DST_ROOT/$dataset"
    dst="$dst_dir/$cfg"

    # 원본이 실제로 존재할 때만 처리
    [[ -f "$src" ]] || continue

    mkdir -p "$dst_dir"
    cp "$src" "$dst"

    # 이미 GAMMA가 있으면 중복 삽입 방지
    if ! grep -q '^[[:space:]]*GAMMA:' "$dst"; then
      # BATCH_SIZE: 라인 바로 아래에 두 줄 삽입
      sed -i '/^[[:space:]]*BATCH_SIZE:/a\
  GAMMA: 0.3\n  SLOTS: 10' "$dst"
    fi
  done
done

echo "✅ 모든 설정 파일을 Practical_TTA로 복사·패치 완료."
