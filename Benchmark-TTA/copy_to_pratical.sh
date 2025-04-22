#!/usr/bin/env bash
# best_cfgs/Online_TTA → best_cfgs/Practical_TTA 로 복사하면서
#   - TEST 블록에 GAMMA: 0.3, SLOTS: 10 삽입
# 그리고 best_cfgs/Online_TTA → best_cfgs/Mixed_TTA 로 복사하면서
#   - TEST 블록에 IMBALANCE_RATIO: 16 삽입
# 그리고 best_cfgs/Online_TTA → best_cfgs/Persistent_TTA 로 복사하면서
#   - TEST 블록에 LOOPS: 50 삽입

set -euo pipefail

SRC_ROOT="./best_cfgs/Online_TTA"
DST_ROOT_PRACTICAL="./best_cfgs/Practical_TTA"
DST_ROOT_MIXED="./best_cfgs/Mixed_TTA"
DST_ROOT_PERSISTENT="./best_cfgs/Persistent_TTA"

datasets=(
  cifar10_c
  cifar100_c
  domainnet126
  imagenet_c
  imagenet_c_vit
  imagenet_convnet
  imagenet_efn
  officehome
)

methods=(
  source.yaml
  tent.yaml
  eata.yaml
  sar.yaml
  cotta.yaml
  adacontrast.yaml
  t3a.yaml
  rotta.yaml
  lame.yaml
  deyo.yaml
  tent_come.yaml
  eata_come.yaml
  sar_come.yaml
  deyo_come.yaml
  memTTA.yaml
)

for dataset in "${datasets[@]}"; do
  for cfg in "${methods[@]}"; do
    src="$SRC_ROOT/$dataset/$cfg"

    # 원본 파일이 실제로 존재할 때만 처리
    [[ -f "$src" ]] || continue

    # ========== 1) Practical_TTA로 복사 (GAMMA / SLOTS 추가) ==========
    dst_dir_p="$DST_ROOT_PRACTICAL/$dataset"
    dst_p="$dst_dir_p/$cfg"
    mkdir -p "$dst_dir_p"
    cp "$src" "$dst_p"

    if ! grep -q '^[[:space:]]*GAMMA:' "$dst_p"; then
      sed -i '/^[[:space:]]*BATCH_SIZE:/a\
  GAMMA: 0.3\n  SLOTS: 10' "$dst_p"
    fi

    # ========== 2) Mixed_TTA로 복사 (IMBALANCE_RATIO 추가) ==========
    dst_dir_m="$DST_ROOT_MIXED/$dataset"
    dst_m="$dst_dir_m/$cfg"
    mkdir -p "$dst_dir_m"
    cp "$src" "$dst_m"

    if ! grep -q '^[[:space:]]*IMBALANCE_RATIO:' "$dst_m"; then
      sed -i '/^[[:space:]]*BATCH_SIZE:/a\
  IMBALANCE_RATIO: 16' "$dst_m"
    fi

    # ========== 3) Persistent_TTA로 복사 (LOOPS 추가) ==========
    dst_dir_t="$DST_ROOT_PERSISTENT/$dataset"
    dst_t="$dst_dir_t/$cfg"
    mkdir -p "$dst_dir_t"
    cp "$src" "$dst_t"

    if ! grep -q '^[[:space:]]*LOOPS:' "$dst_t"; then
      sed -i '/^[[:space:]]*BATCH_SIZE:/a\
  LOOPS: 50' "$dst_t"
    fi

  done
done

echo "✅ 모든 설정 파일을 Practical_TTA, Mixed_TTA, Persistent_TTA로 복사·패치 완료."
