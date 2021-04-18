import sys
import argparse
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

import utils as vis
sys.path.append('raft_core')
from raft import RAFT
from raft_utils.utils import InputPadder


@torch.no_grad()
def main(args):
    os.makedirs(args.output_path, exist_ok = True)

    checkpoint = {k.replace('module.', '') : v for k, v in torch.load(args.checkpoint, map_location = 'cpu').items()}

    model = RAFT(args)
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    images = sorted(os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if any(map(f.endswith, ['.png', '.jpg'])))

    load_image = lambda img_path: torch.as_tensor(cv2.imread(img_path)).flip(-1).movedim(-1, 0).unsqueeze(0).div(255.0).to(args.device)

    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        print(imfile1, imfile2)
        output_path = os.path.join(args.output_path, os.path.basename(imfile1) + '_' + os.path.basename(imfile2))

        image1, image2 = map(load_image, [imfile1, imfile2])

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters = 20, test_mode = True)
        
        img, flo = image1[0].movedim(0, -1).cpu(), flow_up[0].movedim(0, -1).cpu()
        
        flo = torch.as_tensor(vis.flow_to_image(flo.numpy()))

        cv2.imwrite(output_path, torch.cat([img * 255, flo]).flip(-1).numpy())
        print(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help="restore checkpoint")
    parser.add_argument('--input-path', '-i', help="dataset for evaluation")
    parser.add_argument('--output-path', '-o', default = 'data/raft_demo_frames')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--device', default = 'cpu')
    args = parser.parse_args()

    main(args)
