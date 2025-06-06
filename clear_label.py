import os

label_root = 'datasets/kaist-rgbt/train/labels/visible'
for fname in os.listdir(label_root):
    if fname.endswith('.txt'):
        path = os.path.join(label_root, fname)
        with open(path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # 앞의 5개만 사용
                new_lines.append(" ".join(parts[:5]) + '\n')
        with open(path, 'w') as f:
            f.writelines(new_lines)
