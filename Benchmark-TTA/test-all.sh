#!/bin/bash

########################
# 0. 공통 설정
########################
METHODS=("deyo")
# METHODS=("tent" "eata" "sar" "t3a" "source" "lame" "memo" "norm" "cotta" "rotta")

# 세 가지 스크립트(파이썬 파일)별 DATASETS
DATASETS_TESTTIME=("cifar10_c" "cifar100_c" "officehome" "imagenet_c" "imagenet_c_vit" "imagenet_convnet")
DATASETS_CONTINUAL=("cifar10_c" "cifar100_c" "officehome" "imagenet_c" "imagenet_c_vit" "imagenet_convnet")
DATASETS_PRACTICAL=("cifar10_c" "cifar100_c" "officehome" "imagenet_c" "imagenet_c_vit" "imagenet_convnet")
# 필요하면 위 배열에 더 추가 가능

# 사용할 GPU 리스트 (예: 0~6)
GPU_LIST=(0 1 2 3 4 5 6)

# 최대 동시 실행 프로세스 수 (GPU 수만큼 or 원하는 병렬 개수)
N_JOBS=7

########################
# 1. 작업 실행 함수
########################
run_job() {
  local script_type="$1"   # "test-time", "continual", "practical"
  local dataset="$2"
  local method="$3"
  local gpu_id="$4"

  echo "===== GPU:$gpu_id | $script_type | DATASET:$dataset | METHOD:$method ====="
  
  if [ "$script_type" == "test-time" ]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" python test-time.py \
      --cfg "best_cfgs/Online_TTA/${dataset}/${method}.yaml" \
      --output_dir "test-time-evaluation/${dataset}/${method}"
  
  elif [ "$script_type" == "continual" ]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" python continual-test-time.py \
      --cfg "best_cfgs/Online_TTA/${dataset}/${method}.yaml" \
      --output_dir "continual-test-time-evaluation/${dataset}/${method}"
  
  elif [ "$script_type" == "practical" ]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" python practical-test-time.py \
      --cfg "best_cfgs/Practical_TTA/${dataset}/${method}.yaml" \
      --output_dir "practical-test-time-evaluation/${dataset}/${method}"
  fi
}

########################
# 2. 실제 병렬 실행 로직
########################

#----------------------------------------------------
# Test-time
#----------------------------------------------------
echo "===== [1/3] Test-time Evaluation 시작 ====="
i=0  # GPU 할당용 인덱스
for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS_TESTTIME[@]}"; do
    # 현재 백그라운드 작업 수가 N_JOBS 이상이면 대기
    while [ "$(jobs -rp | wc -l)" -ge "$N_JOBS" ]; do
      sleep 2
    done

    # 라운드 로빈 방식으로 GPU 할당
    GPU_ID=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
    i=$((i+1))

    run_job "test-time" "$DATASET" "$METHOD" "$GPU_ID" &
  done
done

# 모든 test-time 작업이 끝날 때까지 대기
wait
echo "===== [1/3] Test-time Evaluation 완료 ====="


#----------------------------------------------------
# Continual test-time
#----------------------------------------------------
echo "===== [2/3] Continual Test-time Evaluation 시작 ====="
i=0
for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS_CONTINUAL[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$N_JOBS" ]; do
      sleep 2
    done

    GPU_ID=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
    i=$((i+1))

    run_job "continual" "$DATASET" "$METHOD" "$GPU_ID" &
  done
done

wait
echo "===== [2/3] Continual Test-time Evaluation 완료 ====="


#----------------------------------------------------
# Practical test-time
#----------------------------------------------------
echo "===== [3/3] Practical Test-time Evaluation 시작 ====="
i=0
for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS_PRACTICAL[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$N_JOBS" ]; do
      sleep 2
    done

    GPU_ID=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
    i=$((i+1))

    run_job "practical" "$DATASET" "$METHOD" "$GPU_ID" &
  done
done

wait
echo "===== [3/3] Practical Test-time Evaluation 완료 ====="


echo "==== 모든 작업이 완료되었습니다! ===="
