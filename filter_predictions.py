# predictions.json 생성 후 실행
import json

with open('runs/train/taehun24/epoch19_predictions.json', 'r') as f:
    preds = json.load(f)

valid_preds = []
for p in preds:
    w, h = p['bbox'][2], p['bbox'][3]
    if w > 5 and h > 5 and p['score'] > 0.01:  # 유효한 박스만 필터링
        valid_preds.append(p)

with open('filtered_predictions.json', 'w') as f:
    json.dump(valid_preds, f, indent=4)
