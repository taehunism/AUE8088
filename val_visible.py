with open('datasets/kaist-rgbt/val.txt', 'r') as f:
    lines = f.readlines()

lines_fixed = [line.replace("{}", "visible") for line in lines]

with open('datasets/kaist-rgbt/val_visible.txt', 'w') as f:
    f.writelines(lines_fixed)
