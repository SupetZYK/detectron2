cfg=FCOS_1x
python3 train_net.py --num-gpus 8 \
    --config-file configs/$cfg.yaml \
    --resume \
    --eval-only \
    OUTPUT_DIR /data/training_dir/$cfg  \
    SOLVER.IMS_PER_BATCH 16 SEED 4073263
