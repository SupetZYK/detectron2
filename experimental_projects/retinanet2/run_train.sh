python3 train_net.py --num-gpus 8 \
    --config-file configs/Retinanet2_1x.yaml \
    --resume \
    OUTPUT_DIR /data/training_dir/base_retinanet2  \
    SOLVER.IMS_PER_BATCH 16 SEED 4073263