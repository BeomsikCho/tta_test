METHODS=("tent")
DATASETS=("cifar10_c")
GPU_id=1

for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "DATASET: $DATASET"
    echo "METHOD: $METHOD"
    CUDA_VISIBLE_DEVICES="$GPU_id" python mixed-test-time-tmp.py \
      --cfg best_cfgs/Mixed_TTA/"${DATASET}"/"${METHOD}".yaml \
      --output_dir "mixed-test-time-evaluation/${DATASET}/${METHOD}"
  done
done
