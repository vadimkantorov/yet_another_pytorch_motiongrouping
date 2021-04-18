import sys
import argparse
import os
import cv2
import torch

import train
import utils

sys.path.append('raft_core')
import raft

def build_optical_flow_model(args):
    checkpoint = {k.replace('module.', '') : v for k, v in torch.load(args.checkpoint, map_location = 'cpu').items()}
    model = raft.RAFT(args).eval()
    model.load_state_dict(checkpoint)
    return model.to(args.device)

@torch.no_grad()
def main(args):
    model = build_optical_flow_model(args)
    dataset, collate_fn, batch_frontend = train.build_dataset(args)

    print(args.dataset_root_dir_flow)
    for idx, (frames_paths, frames, flow_paths, flow) in enumerate(dataset):
        assert frames.shape[-1] % 8 == 0 and frames.shape[-1] % 8 == 0

        frames = frames.to(args.device)
        img_src, img_dst, flow_dst = frames[:1], frames[1:], flow_paths[1:]
        
        flow_lo, flow_hi = model(img_src.expand(len(img_dst), -1, -1, -1), img_dst, iters = args.num_iter, test_mode = True)

        os.makedirs(os.path.dirname(flow_dst[0]), exist_ok = True)
        for flo, flow_path in zip(flow_hi.cpu(), flow_dst):
            flo = torch.as_tensor(utils.flow_to_image(flo.movedim(0, -1).numpy()))
            cv2.imwrite(flow_path, flo.flip(-1).numpy())

        print(idx, '/', len(dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed-precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate-corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--device', default = 'cpu')
    parser.add_argument('--checkpoint', help="restore checkpoint", default = '../selfsupslots/data/common/raft/models/raft-things.pth')
    parser.add_argument('--small', action = 'store_true')
    parser.add_argument('--num-iter', type = int, default = 20)
    
    parser.add_argument('--dataset', default = 'DAVIS', choices = ['DAVIS'])
    parser.add_argument('--dataset-root-dir', default = '../selfsupslots/data/common/DAVIS')
    parser.add_argument('--dataset-root-dir-flow', default = '../selfsupslots/data/common/DAVISflow')
    parser.add_argument('--dataset-year', type = int, default = 2017)
    parser.add_argument('--dataset-resolution', default = '480p')
    parser.add_argument('--split-name', default = 'train', choices = ['train', 'val'])
    parser.add_argument('--dt', type = int, action = 'append', default = [0, -2, -1, 1, 2])

    main(parser.parse_args())
