DATASET="cifar100_c"     # cifar10_c cifar100_c imagenet_c domainnet126 officehome imagenet_vit imagenet_convnet imagenet_efn
METHOD="tent"        # source norm_test memo eata cotta tent t3a norm_alpha lame adacontrast sar

GPU_id=0

for DATASET in "cifar100_c"
do  
    echo DATASET: $DATASET
    echo METHOD: $METHOD
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py --cfg best_cfgs/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}"
done