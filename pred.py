# 예측 결과 파일 확인
with open('runs/train/yolov5n-rgbt101/epoch47_predictions.json', 'r') as f:
    predictions = json.load(f)

print(f"예측 결과 수: {len(predictions)}")
print(f"샘플 예측 구조: {predictions[0] if predictions else 'None'}")

# 이미지 ID 범위 확인
image_ids = set(pred['image_id'] for pred in predictions)
print(f"예측 결과 이미지 ID 범위: {min(image_ids)} - {max(image_ids)}")
