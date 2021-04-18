import os
import cv2
import torch
import torchvision

class DAVIS(torchvision.datasets.VisionDataset):
    def __init__(self, root, split_name, year = 2016, resolution = '480p', root_flow = None, transform = torchvision.transforms.ToTensor(), dt = [0, -2, -1, 1, 2], fmt = '{:05d}.jpg', filter = None, pad = lambda tensor, multiple = 8: tensor[..., : (tensor.shape[-2] // multiple) * multiple, : (tensor.shape[-1] // multiple) * multiple] ):
        super().__init__(root, transform = transform)
        self.resolution = resolution
        self.pad = pad
        self.classes = list(map(str.strip, open(os.path.join(root, 'ImageSets', str(year), split_name + '.txt'))))
        frame_count = { class_name : 1 + int(os.path.splitext(sorted(os.listdir(os.path.join(root, 'JPEGImages', resolution, class_name)))[-1])[0] ) for class_name in self.classes }

        frame_path = lambda class_name, t: os.path.join(root, 'JPEGImages', resolution, class_name, fmt.format(t))
        flow_path = lambda class_name, t, t_n: os.path.join(root_flow, 'JPEGImages', resolution, class_name, (fmt + '_' + fmt).format(t, t_n))

        self.frames = [ list(frame_path(class_name, min(max(0, t + n), cnt - 1)) for n in dt) for class_name, cnt in frame_count.items() for t in range(cnt) ]
        self.frames_flow = [ list(flow_path(class_name, t, min(max(0, t + n), cnt - 1)) for n in dt) for class_name, cnt in frame_count.items() for t in range(cnt) ] if root_flow else ([()] * len(self.frames))

        # metadata DAVIS/Annotations/480p/lucia/00000.png

    def __getitem__(self, idx):
        read_stack = lambda paths, **kwargs: self.pad(torch.stack([ torch.as_tensor(cv2.imread(frame_path)) for frame_path in paths ]).movedim(-1, 1)).flip(1).div(255.0) if paths and all(map(os.path.exists, paths)) else torch.empty(**kwargs)
        
        frames = read_stack(self.frames[idx])
        frames_flow = read_stack(self.frames_flow[idx], size = frames.shape, dtype = frames.dtype)

        return self.frames[idx], frames, self.frames_flow[idx], frames_flow

    def __len__(self):
        return len(self.frames)
