import random

with open('datasets/kaist-rgbt/train-all-04.txt', 'r') as f:
    lines = f.readlines()

# '{}'를 'visible'로 치환
lines = [line.replace('{}', '{}') for line in lines]

random.shuffle(lines)
split_idx = int(len(lines) * 0.8)
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open('datasets/kaist-rgbt/train.txt', 'w') as f:
    f.writelines(train_lines)
with open('datasets/kaist-rgbt/val.txt', 'w') as f:
    f.writelines(val_lines)

# (선택) 결과 미리보기
print("train.txt 샘플:")
print("".join(train_lines[:5]))
print("val.txt 샘플:")
print("".join(val_lines[:5]))

