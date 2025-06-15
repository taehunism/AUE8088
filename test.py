import json

# 어노테이션 파일 로드 및 구조 확인
with open('datasets/kaist-rgbt/KAIST_val-D_annotation.json', 'r') as f:
    annotation = json.load(f)

print("필수 필드 존재 여부:")
print(f"- images: {'존재' if 'images' in annotation else '없음'}")
print(f"- annotations: {'존재' if 'annotations' in annotation else '없음'}")
print(f"- categories: {'존재' if 'categories' in annotation else '없음'}")

print(f"\n이미지 수: {len(annotation.get('images', []))}")
print(f"어노테이션 수: {len(annotation.get('annotations', []))}")

print(f"--------------------------------------------")


# 예측 결과 파일 확인
with open('runs/train/yolov5n-rgbt101/epoch47_predictions.json', 'r') as f:
    predictions = json.load(f)

print(f"예측 결과 수: {len(predictions)}")
print(f"샘플 예측 구조: {predictions[0] if predictions else 'None'}")

# 이미지 ID 범위 확인
image_ids = set(pred['image_id'] for pred in predictions)
print(f"예측 결과 이미지 ID 범위: {min(image_ids)} - {max(image_ids)}")

print(f"--------------------------------------------")

# 어노테이션과 예측 결과의 이미지 ID 매칭 확인
ann_image_ids = set(img['id'] for img in annotation['images'])
pred_image_ids = set(pred['image_id'] for pred in predictions)

print(f"어노테이션 이미지 ID 개수: {len(ann_image_ids)}")
print(f"예측 결과 이미지 ID 개수: {len(pred_image_ids)}")
print(f"공통 ID 개수: {len(ann_image_ids & pred_image_ids)}")
print(f"매칭률: {len(ann_image_ids & pred_image_ids) / len(ann_image_ids) * 100:.2f}%")