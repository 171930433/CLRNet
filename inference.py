import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
import time  # 1. 导入time模块
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path

# 运行脚本的命令示例:
# python ./inference.py configs/clrnet/clr_resnet18_culane.py --img ../my_slam_ws/img/ --load_from ./culane_r18.pth --savedir ./vis/

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        if torch.cuda.is_available():
            self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        # 如果图片读取失败，提前返回None
        if ori_img is None:
            return None
        ori_img = cv2.resize(ori_img, (self.cfg.ori_img_w, self.cfg.ori_img_h))
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path': img_path, 'ori_img': ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            if torch.cuda.is_available():
                for key, val in data.items():
                    if torch.is_tensor(val):
                        data[key] = val.cuda()
            
            output = self.net(data)
            lanes = self.net.module.heads.get_lanes(output)
        return lanes

def get_img_paths(path):
    """
    改进版函数，仅获取有效的图片文件路径。
    """
    p = str(Path(path).absolute())
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))
    elif os.path.isdir(p):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        paths = []
        for ext in image_extensions:
            paths.extend(sorted(glob.glob(os.path.join(p, ext))))
    elif os.path.isfile(p):
        paths = [p]
    else:
        raise Exception(f'错误: {p} 不存在')

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    paths = [x for x in paths if x.lower().endswith(supported_formats)]
    
    return paths

def process(args):
    """
    交互式运行模式:
    - 'd': 下一张图片
    - 'a': 上一张图片
    - 's': 保存当前图片
    - 'q': 退出
    """
    cfg = Config.fromfile(args.config)
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    
    paths = get_img_paths(args.img)
    if not paths:
        print("在指定路径下未找到任何图片。")
        return

    current_idx = 0
    window_name = "CLRNet Lane Detection (d: next, a: prev, s: save, q: quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        img_path = paths[current_idx]
        print(f"处理图片 {current_idx + 1}/{len(paths)}: {osp.basename(img_path)}")

        data = detect.preprocess(img_path)
        if data is None:
            print(f"警告: 无法读取图片 {img_path}，已跳过。")
            # 自动跳到下一张或决定如何处理
            if len(paths) == 1: break
            current_idx = min(current_idx + 1, len(paths) - 1)
            continue

        # --- 2. 计算推理时间 ---
        torch.cuda.synchronize() # 确保GPU操作完成
        start_time = time.time()
        
        data['lanes'] = detect.inference(data)[0]
        
        torch.cuda.synchronize() # 确保GPU操作完成
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒

        # --- 可视化 ---
        lanes = [lane.to_array(cfg) for lane in data['lanes']]
        vis_img = data['ori_img'].copy()
        imshow_lanes(vis_img, lanes, show=False, width=1)

        # --- 3. 在图片上绘制推理时间 ---
        time_text = f"Inference: {inference_time:.2f} ms"
        cv2.putText(vis_img, time_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        vis_img = cv2.resize(vis_img, (1920, 1080))
        cv2.imshow(window_name, vis_img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            current_idx = min(current_idx + 1, len(paths) - 1)
        elif key == ord('a'):
            current_idx = max(current_idx - 1, 0)
        # --- 4. 按's'键保存图片 ---
        elif key == ord('s'):
            if cfg.savedir:
                if not osp.exists(cfg.savedir):
                    os.makedirs(cfg.savedir)
                out_file = osp.join(cfg.savedir, osp.basename(data['img_path']))
                cv2.imwrite(out_file, vis_img)
                print(f"图片已保存至: {out_file}")
            else:
                print("未指定保存目录。请使用 --savedir 参数来设置保存路径。")
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img', help='The path of the img (img file or img_folder)')
    parser.add_argument('--show', action='store_true', help='(此模式下忽略)')
    parser.add_argument('--savedir', type=str, default=None, help='保存图片的目录')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)