METHODS=("deyo" "tent_come" "eata_come" "sar_come" "deyo_come")
DATASETS=("cifar10_c" "cifar100_c" "officehome" "imagenet_c" "imagenet_c_vit" "imagenet_c_efn" "imagenet_convnet")
GPU_id=1

for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "DATASET: $DATASET"
    echo "METHOD: $METHOD"
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py \
      --cfg best_cfgs/Online_TTA/"${DATASET}"/"${METHOD}".yaml \
      --output_dir "test-time-evaluation/${DATASET}/${METHOD}"
  done
done
