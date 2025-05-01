METHODS=(
  "source"
  "tent"
  "cotta"
  "lame"
  "eata"
  "sar"
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
GPU_id=2

for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "DATASET: $DATASET"
    echo "METHOD: $METHOD"
    CUDA_VISIBLE_DEVICES="$GPU_id" python practical-test-time.py \
      --cfg best_cfgs/Practical_TTA/"${DATASET}"/"${METHOD}".yaml \
      --output_dir "practical-test-time-evaluation/${DATASET}/${METHOD}"
  done
done
