cfg=FCOS_1x
python3 ../../demo/demo.py \
    --config-file configs/$cfg.yaml \
    --input ../../demo/sample.jpg \
    --output /data/training_dir/$cfg/output.jpg \
    --opts MODEL.WEIGHTS /data/training_dir/$cfg/model_final.pth MODEL.FCOS.TEST_SCORE_THRESH 0.5
