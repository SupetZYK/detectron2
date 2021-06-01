cfg=Retinanet2_D_R50_1x
dir=base_retinanet2_d_r50
python3 ../../demo/demo.py \
    --config-file configs/$cfg.yaml \
    --input ../../demo/sample.jpg \
    --output /data/training_dir/$dir/output.jpg \
    --opts MODEL.WEIGHTS /data/training_dir/$dir/model_0009999.pth MODEL.RETINANET2.TEST_SCORE_THRESH 0.5
