cfg=Retinanet2_D_R50_1x_coco_3
python3 train_net.py --num-gpus 8 \
    --config-file configs/$cfg.yaml \
    OUTPUT_DIR /data/training_dir/$cfg  \
    SOLVER.IMS_PER_BATCH 16 SEED 4073263