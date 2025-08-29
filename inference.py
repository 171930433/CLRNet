import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path
# from tqdm import tqdm # tqdm is no longer needed for the interactive loop

# The command to run the script remains the same, for example:
# python your_script_name.py ../configs/clrnet/clr_resnet18_llamas.py --img ../test_data --load_from ../models/llamas_r18.pth --savedir ../vis

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        # The original code used device_ids=range(1), which assumes at least one GPU.
        # Making it more robust to run on CPU or a single GPU setup.
        if torch.cuda.is_available():
            self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path': img_path, 'ori_img': ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            # Move data to GPU if available
            if torch.cuda.is_available():
                for key, val in data.items():
                    if torch.is_tensor(val):
                        data[key] = val.cuda()
            
            output = self.net(data)
            lanes = self.net.module.heads.get_lanes(output)
        return lanes

    # This show method is no longer used in the new interactive process function,
    # but we'll keep it for potential future use.
    def show(self, data):
        out_file = self.cfg.savedir
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data


def get_img_paths(path):
    """
    Improved function to get paths of only image files.
    """
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        # Define common image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        paths = []
        for ext in image_extensions:
            paths.extend(sorted(glob.glob(os.path.join(p, ext))))
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')

    # Filter out any non-image files that might have been caught by glob
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    paths = [x for x in paths if x.lower().endswith(supported_formats)]
    
    return paths


# ======================================================================================
# START OF MODIFIED SECTION
# ======================================================================================
def process(args):
    """
    This function is modified to run in an interactive mode.
    - 'd': Next image
    - 'a': Previous image
    - 'q': Quit
    """
    cfg = Config.fromfile(args.config)
    # The --show argument is now ignored, as the window is always shown in interactive mode
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    
    paths = get_img_paths(args.img)
    if not paths:
        print("No images found in the specified path.")
        return

    current_idx = 0
    window_name = "CLRNet Lane Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        # 1. Get current image path
        img_path = paths[current_idx]
        print(f"Processing image {current_idx + 1}/{len(paths)}: {osp.basename(img_path)}")

        # 2. Preprocess and Inference
        data = detect.preprocess(img_path)
        data['lanes'] = detect.inference(data)[0]

        # 3. Visualization
        # We manually call the visualization function to get the image array
        lanes = [lane.to_array(cfg) for lane in data['lanes']]
        
        # We need a copy of the original image to draw on, otherwise the drawing persists
        # across different key presses on the same image.
        vis_img = data['ori_img'].copy()
        imshow_lanes(vis_img, lanes, show=False) # show=False to prevent it from creating its own window

        # 4. Display the image
        cv2.imshow(window_name, vis_img)

        # 5. Save the image if a save directory is provided
        if cfg.savedir:
            if not osp.exists(cfg.savedir):
                os.makedirs(cfg.savedir)
            out_file = osp.join(cfg.savedir, osp.basename(data['img_path']))
            cv2.imwrite(out_file, vis_img)

        # 6. Wait for user input to control flow
        key = cv2.waitKey(0) & 0xFF  # wait indefinitely for a key press

        if key == ord('q'):  # 'q' to quit
            break
        elif key == ord('d'):  # 'd' to go to the next image
            current_idx = min(current_idx + 1, len(paths) - 1) # Move to next, but don't go past the end
        elif key == ord('a'):  # 'a' to go to the previous image
            current_idx = max(current_idx - 1, 0) # Move to previous, but don't go below zero
            
    # Clean up
    cv2.destroyAllWindows()

# ======================================================================================
# END OF MODIFIED SECTION
# ======================================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img', help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true',
                        help='Whether to show the image (ignored in interactive mode)')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)