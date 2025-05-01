METHODS=("tent")
DATASETS=("cifar10_c")
GPU_id=0

for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "DATASET: $DATASET"
    echo "METHOD: $METHOD"
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-tmp.py \
      --cfg best_cfgs/Online_TTA/"${DATASET}"/"${METHOD}".yaml \
      --output_dir "test-time-evaluation/${DATASET}/${METHOD}"
  done
done
