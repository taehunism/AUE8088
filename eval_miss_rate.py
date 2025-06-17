import sys
import os
sys.path.append('utils/eval')

from utils.eval.kaisteval import evaluate

def run_miss_rate_evaluation():
    '''L695-L711 코드 기반 Miss Rate 평가 실행'''

    # 1. 예측 결과 파일 경로 설정
    predictions_file = "runs/train/yolov5n-rgbt/epoch19_predictions.json"

    # 2. 결과 저장 폴더 생성
    os.makedirs('results/miss_rate_plots', exist_ok=True)

    # 3. All/Day/Night 3가지 설정으로 평가 실행
    subsets = ['all', 'day', 'night']
    results = {}

    for subset in subsets:
        print(f"\n🔍 {subset.upper()} 서브셋 평가 중...")

        # kaisteval.py의 evaluate_multiple 함수 사용
        mr_result = evaluate(
            result_files=[predictions_file],
            dataset_type='kaist',
            subset=subset,
            output_dir=f'results/miss_rate_plots/{subset}'
        )

        results[subset] = mr_result

        print(f"✅ {subset} Log-average Miss Rate: {mr_result['lamr']:.3f}")

    # 4. 결과 요약 테이블 생성
    print("\n" + "="*50)
    print("📈 MISS RATE 평가 결과 요약")
    print("="*50)
    print(f"{'Subset':<10} {'Log-avg MR':<12} {'개선도':<10}")
    print("-"*32)

    baseline_mr = 0.20  # 베이스라인 가정
    for subset, result in results.items():
        improvement = ((baseline_mr - result['lamr']) / baseline_mr) * 100
        print(f"{subset.capitalize():<10} {result['lamr']:<12.3f} {improvement:>+6.1f}%")

    return results

# 실행
# results = run_miss_rate_evaluation()
