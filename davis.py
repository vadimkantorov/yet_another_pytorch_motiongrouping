import os
import cv2
import torch
import torchvision

class DAVIS(torchvision.datasets.VisionDataset):
    def __init__(self, root, split_name, year = 2017, resolution = '480p', root_flow = None, transform = torchvision.transforms.ToTensor(), dt = [0, -2, -1, 1, 2], fmt = '{:05d}.jpg', filter = None):
        super().__init__(root, transform = transform)
        self.resolution = resolution
        self.classes = list(map(str.strip, open(os.path.join(root, 'ImageSets', str(year), split_name + '.txt'))))
        frame_count = { class_name : 1 + int(os.path.splitext(sorted(os.listdir(os.path.join(root, 'JPEGImages', resolution, class_name)))[-1])[0] ) for class_name in self.classes }
        
        frame_path = lambda class_name, t: os.path.join(root, 'JPEGImages', resolution, class_name, fmt.format(t))
        flow_path = lambda class_name, t, t_n: os.path.join(root_flow, 'JPEGImages', resolution, class_name, (fmt + '_' + fmt).format(t, t_n))
        
        # assert all(os.path.exists(frame_path(class_name, t)) for class_name, cnt in frame_count.items() for t in range(cnt))
        
        self.frames = [ (frame_path(class_name, min(max(0, t + n), cnt - 1)) for n in dt) for class_name, cnt in frame_count.items() for t in range(cnt) ]
        self.frames_flow = [ (flow_path(class_name, t, min(max(0, t + n), cnt - 1)) for n in dt) for class_name, cnt in frame_count.items() for t in range(cnt) ] if root_flow else ([()] * len(self.frames))

    def __getitem__(self, idx):
        read_stack = lambda paths: torch.stack([ torch.as_tensor(cv2.imread(frame_path)).flip(-1) for frame_path in paths ])

        frames = read_stack(self.frames[idx])
        frames_flow = read_stack(self.frames_flow[idx]) if self.frames_flow[idx] else torch.empty((len(frames),) + ((0,) * (frames.ndim - 1)), dtype = frames.dtype)

        return frames, frames_flow, self.frames[idx], self.frames_flow[idx]

    def __len__(self):
        return len(self.frames)


if __name__ == '__main__':
    dataset = DAVIS('../selfsupslots/data/common/DAVIS', split_name = 'train', year = 2017, resolution = '480p')
    frames, frames_flow = dataset[0]
