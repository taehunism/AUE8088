# 예시: kaisteval.py 내부 함수 사용
from utils.eval.kaisteval import evaluate

# 결과 파일 경로와 annotation 경로 지정
pred_file = 'runs/train/yolov5n-rgbt34/epoch19_predictions.json'  # 예측 결과
gt_file = 'datasets/kaist-rgbt/KAIST_annotation.json'  # ground truth

evaluate(pred_file, gt_file, plot=True)  # plot=True일 때 MR-FPPI 곡선 자동 생성
