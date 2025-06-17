# !/bin/bash
source ~/.bashrc

conda activate deep

cd /home/unicon/AUE8088

python utils/eval/generate_kaist_ann_json.py \
    --textListFile datasets/kaist-rgbt/val.txt

python train_simple.py \
    --img 640 \
    --batch-size 8 \
    --epochs 20 \
    --data data/kaist-rgbt.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 8 \
    --name taehun_val \
    --rgbt \
    --single-cls
