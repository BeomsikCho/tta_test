# DATASETS=("cifar10_c" "cifar100_c" "domainnet126" "officehome" "imagenet_c" "imagenet_c_vit" "imagenet_convnet")

METHODS=("cotta" "rotta")
DATASETS=("cifar10_c" "cifar100_c" "officehome" "imagenet_c" "imagenet_c_vit" "imagenet_convnet")
GPU_id=6

for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "DATASET: $DATASET"
    echo "METHOD: $METHOD"
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py \
      --cfg best_cfgs/Online_TTA/"${DATASET}"/"${METHOD}".yaml \
      --output_dir "test-time-evaluation/${DATASET}/${METHOD}"
  done
done
