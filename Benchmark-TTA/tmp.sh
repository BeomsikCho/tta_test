#!/usr/bin/env bash
set -euo pipefail

METHODS=(
  "tent"
  "eata"
  "sar"
  "t3a"
  "source"
  "lame"
  "norm"
  "cotta"
  "rotta"
  "deyo"
  "tent_come"
  "eata_come"
  "sar_come"
  "deyo_come"
)
DATASETS=(
  "cifar10_c"
  "cifar100_c"
  "officehome"
  "imagenet_c"
  "imagenet_c_vit"
  "imagenet_convnet"
)

# 사용할 GPU 개수 (0~6번 → 7개)
NGPUS=7

# 1) (METHOD, DATASET) 쌍으로 jobs 리스트를 구성
jobs=()
for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    jobs+=("$METHOD $DATASET")
  done
done

# 2) 특정 GPU 번호(gpu_id)에 대해 할당된 jobs를 순서대로 실행하는 함수
run_jobs_on_gpu() {
  local gpu_id="$1"
  local num_gpus="$2"
  local total_jobs="${#jobs[@]}"

  # gpu_id부터 시작해서 num_gpus 간격으로 job을 배정 (예: 0,7,14,...)
  for (( i=gpu_id; i<total_jobs; i+=num_gpus )); do
    # "METHOD DATASET" 문자열을 분리
    IFS=' ' read -r METHOD DATASET <<< "${jobs[$i]}"
    echo "[GPU $gpu_id] METHOD=$METHOD, DATASET=$DATASET"

    # 실제 실행 커맨드
    CUDA_VISIBLE_DEVICES="$gpu_id" python persistent-test-time.py \
      --cfg "best_cfgs/Persistent_TTA/${DATASET}/${METHOD}.yaml" \
      --output_dir "persistent-test-time-evaluation/${DATASET}/${METHOD}"
  done
}

# 3) GPU 0~6번 각각을 백그라운드로 실행하고, 모든 실행이 끝날 때까지 대기
for (( gid=0; gid<NGPUS; gid++ )); do
  run_jobs_on_gpu "$gid" "$NGPUS" &
done

wait

echo "✅ 모든 작업이 종료되었습니다."
