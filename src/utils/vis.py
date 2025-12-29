import os
import os.path as osp
from typing import Tuple
from tqdm import tqdm
import cv2

from utils import Center

def draw_frame(img_or_path,
               center: Center,
               color: Tuple,
               radius : int = 5,
               thickness : int = -1,
    ):
        if osp.isfile(img_or_path):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        xy   = center.xy
        visi = center.is_visible
        if visi:
            x, y = xy
            x, y = int(x), int(y)
            img  = cv2.circle(img, (x,y), radius, color, thickness=thickness)
        
        return img
        
# def gen_video(video_path, 
#               vis_dir, 
#               resize=1.0, 
#               fps=30.0, 
#               fourcc='mp4v'
# ):

#     fnames = os.listdir(vis_dir)
#     fnames.sort()
#     h,w,_   = cv2.imread(osp.join(vis_dir, fnames[0])).shape
#     im_size = (int(w*resize), int(h*resize))
#     fourcc  = cv2.VideoWriter_fourcc(*fourcc)
#     out     = cv2.VideoWriter(video_path, fourcc, fps, im_size)

#     for fname in tqdm(fnames):
#         im_path = osp.join(vis_dir, fname)
#         im      = cv2.imread(im_path)
#         im = cv2.resize(im, None, fx=resize, fy=resize)
#         out.write(im)

def gen_video(video_path, 
              vis_dir, 
              resize=1.0, 
              fps=30.0, 
              fourcc='mp4v'
):

    # 检查目录是否存在
    if not osp.exists(vis_dir):
        print(f"Error: Visualization directory does not exist: {vis_dir}")
        return
    
    # 获取目录中的文件列表
    try:
        fnames = os.listdir(vis_dir)
    except OSError as e:
        print(f"Error: Cannot access directory {vis_dir}, {e}")
        return
    
    # 过滤出图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    fnames = [fname for fname in fnames if fname.lower().endswith(image_extensions)]
    
    if not fnames:
        print(f"Error: Cannot read any image from {vis_dir}")
        return
    
    fnames.sort()
    
    # 找到第一个可以读取的图像
    first_image = None
    first_image_path = None
    for fname in fnames:
        img_path = osp.join(vis_dir, fname)
        img = cv2.imread(img_path)
        if img is not None:
            first_image = img
            first_image_path = img_path
            break
    
    if first_image is None:
        print(f"Error: Cannot read any image from {vis_dir}")
        return
    
    h, w, _ = first_image.shape
    im_size = (int(w*resize), int(h*resize))
    fourcc  = cv2.VideoWriter_fourcc(*fourcc)
    out     = cv2.VideoWriter(video_path, fourcc, fps, im_size)

    for fname in tqdm(fnames):
        im_path = osp.join(vis_dir, fname)
        im      = cv2.imread(im_path)
        if im is not None:
            im = cv2.resize(im, None, fx=resize, fy=resize)
            out.write(im)