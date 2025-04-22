METHODS=("tent" "eata" "sar" "t3a" "source" "lame" "memo" "norm" "cotta" "rotta")
DATASETS=("cifar10_c" "cifar100_c" "officehome" "imagenet_c" "imagenet_c_vit" "imagenet_convnet")
GPU_id=0

for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "DATASET: $DATASET"
    echo "METHOD: $METHOD"
    CUDA_VISIBLE_DEVICES="$GPU_id" python mixed-test-time.py \
      --cfg best_cfgs/Mixed_TTA/"${DATASET}"/"${METHOD}".yaml \
      --output_dir "mixed-test-time-evaluation/${DATASET}/${METHOD}"
  done
done
