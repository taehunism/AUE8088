import os

label_dir = 'datasets/kaist-rgbt/train/labels/visible'
empty_labels = []
nonempty_labels = []

for fname in os.listdir(label_dir):
    if fname.endswith('.txt'):
        with open(os.path.join(label_dir, fname)) as f:
            content = f.read().strip()
            if content == "":
                empty_labels.append(fname)
            else:
                nonempty_labels.append(fname)

print(f"비어있는 라벨 파일 개수: {len(empty_labels)}")
print(f"라벨이 있는 파일 개수: {len(nonempty_labels)}")
print("라벨이 들어있는 예시:", nonempty_labels[:5])
