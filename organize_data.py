import os
import shutil
from pathlib import Path

# 配置路径
base_dir = ".\datasets"  # 你的根目录
source_dir = os.path.join(base_dir, "tennis_predict")  # 原始数据目录（需要被改变结构的目录）
target_dir = os.path.join(base_dir, "tennis_predict0")  # 目标目录（从中复制Label.csv的目录）

# 检查源目录是否存在
if not os.path.exists(source_dir):
    raise FileNotFoundError(f"Source directory {source_dir} does not exist")

# 检查目标目录是否存在
if not os.path.exists(target_dir):
    raise FileNotFoundError(f"Target directory {target_dir} does not exist")

# 从 tennis_predict0 的 game_1 的 Clip_1 中找到 Label.csv
label_source = None
game1_path = os.path.join(target_dir, "game_1")
if os.path.exists(game1_path):
    clip1_path = os.path.join(game1_path, "Clip_1")
    if os.path.exists(clip1_path):
        potential_label = os.path.join(clip1_path, "Label.csv")
        if os.path.exists(potential_label):
            label_source = potential_label

if not label_source:
    # 如果在 game_1/Clip_1 中没找到，尝试在目标目录的其他地方查找
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower() == 'label.csv':
                label_source = os.path.join(root, file)
                break
        if label_source:
            break

if not label_source:
    raise FileNotFoundError(f"Label.csv not found in target directory {target_dir} or its subdirectories")

print(f"Found Label.csv at: {label_source}")

# 遍历 tennis_predict 下的所有 game_X 文件夹
for game_folder in os.listdir(source_dir):
    game_path = os.path.join(source_dir, game_folder)
    if not os.path.isdir(game_path):
        continue

    # 创建对应的结构：tennis_predict/game_X/Clip_1
    clip_path = os.path.join(game_path, "Clip_1")
    os.makedirs(clip_path, exist_ok=True)

    # 移动图片文件（假设是 img_000000.jpg 到 img_000010.jpg）
    image_files = [f"img_{i:06d}.jpg" for i in range(11)]
    for img_file in image_files:
        src_img = os.path.join(game_path, img_file)  # 从原始 game 文件夹中找图片
        dst_img = os.path.join(clip_path, img_file)  # 移动到 Clip_1 子文件夹中
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)  # 使用 move 而不是 copy2，这样会移动而不是复制
            print(f"Moved {src_img} to {dst_img}")
        else:
            print(f"Warning: {src_img} not found, skipping.")

    # 从 tennis_predict0 复制 Label.csv 到当前 game 的 Clip_1 文件夹
    dst_label = os.path.join(clip_path, "Label.csv")
    shutil.copy2(label_source, dst_label)
    print(f"Copied {label_source} to {dst_label}")

print("文件夹结构改变完成！")