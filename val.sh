python val.py \
    --weights "runs/train/yolov5n-rgbt103/weights/best.pt" \
    --data data/kaist-rgbt.yaml \
    --name "taehun" \
    --task test \
    --save-json \
    --verbose \
    --single-cls

python val.py \
    --weights "runs/train/taehun/weights/best.pt" \
    --data data/kaist-rgbt.yaml \
    --name "taehun" \
    --task val \
    --verbose \
    --single-cls \
    --rgbt