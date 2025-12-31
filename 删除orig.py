import os
import shutil
from pathlib import Path

def remove_orig_folders(base_path):
    """
    删除base_path下所有子文件夹中的wasb/orig/和yolo/orig/文件夹
    """
    base_path = Path(base_path)
    
    # 遍历base_path下的所有子文件夹
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            # 检查wasb子文件夹中是否有orig文件夹
            wasb_path = subfolder / "wasb"
            if wasb_path.exists() and wasb_path.is_dir():
                orig_path = wasb_path / "orig"
                if orig_path.exists() and orig_path.is_dir():
                    print(f"正在删除: {orig_path}")
                    shutil.rmtree(orig_path)
            
            # 检查yolo子文件夹中是否有orig文件夹
            yolo_path = subfolder / "yolo"
            if yolo_path.exists() and yolo_path.is_dir():
                orig_path = yolo_path / "orig"
                if orig_path.exists() and orig_path.is_dir():
                    print(f"正在删除: {orig_path}")
                    shutil.rmtree(orig_path)

if __name__ == "__main__":
    # 设置基础路径
    base_path = "reports/yolo_only_review"
    
    # 执行删除操作
    remove_orig_folders(base_path)
    print("删除操作完成！")