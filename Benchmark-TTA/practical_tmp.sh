GPU_id=0
DATASET="cifar10_c"
METHOD="tent"

echo "DATASET: $DATASET"
echo "METHOD: $METHOD"
CUDA_VISIBLE_DEVICES="$GPU_id" python practical-test-time.py \
    --cfg best_cfgs/Practical_TTA/"${DATASET}"/"${METHOD}".yaml \
    --output_dir "practical-test-time-evaluation/${DATASET}/${METHOD}"
