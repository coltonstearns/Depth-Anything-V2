import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2

# from ext depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)  # was 518!
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--dataset', type=str, default='vkitti', choices=['vkitti', 'hypersim'])

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{args.dataset}_{args.encoder}.pth', map_location='cpu'))

    depth_anything = depth_anything.to(DEVICE).eval()

    # get rgb filenames
    filenames = os.listdir(args.img_path)
    filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
    filenames.sort()
    os.makedirs(args.outdir, exist_ok=True)

    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)
        h, w = raw_image.shape[:2]
        depth = depth_anything.infer_image(raw_image, args.input_size)

        # convert to our depth map
        FAR_PLANE = 1e2
        # depth = depth.astype(np.float32) / 65535.0 * 1000.0
        # depth[depth == 0] = 1/FAR_PLANE
        # depth = 1 / depth
        depth = depth.astype(np.float32)
        print(depth.shape)
        # print(depth.min())
        # print(depth.max())
        # print(depth.mean())
        filename = os.path.basename(filename)
        outpath = os.path.join(args.outdir, filename.replace('png', 'npy'))
        np.save(outpath, depth)
