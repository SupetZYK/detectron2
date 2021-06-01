
python3 tools/train_net.py --num-gpus 8 \
    --config-file configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml \
    OUTPUT_DIR training_dir/base_retinanet SOLVER.IMS_PER_BATCH 16 SEED 4073263

