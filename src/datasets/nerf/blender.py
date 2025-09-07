import torch.utils.data as data
import torch
import numpy as np
import os
import imageio
import cv2
import json
import ipdb
from src.config import cfg


class Dataset(data.Dataset):
    def get_rays(self, H, W, focal, c2w):
        """Get ray origins, directions from a pinhole camera."""
        i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                              torch.arange(H, dtype=torch.float32),
                              indexing='xy')
        i, j = i.to(c2w.device), j.to(c2w.device)
        
        dirs = torch.stack([(i - W * 0.5) / focal,
                            -(j - H * 0.5) / focal,
                            -torch.ones_like(i)], -1)
        
        # Rotate ray directions from camera frame to the world frame
        # (c2w[:3, :3] @ dirs[..., None]).squeeze(-1) 是正确的矩阵向量乘法
        rays_d = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
        
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        
        return rays_o, rays_d
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
        Output:
            None
        """
        super(Dataset, self).__init__()
        """
        Write your codes here.
        """
        self.data_root = kwargs.get('data_root')
        self.split = kwargs.get('split', 'train')
        self.input_ratio = kwargs.get('input_ratio', 1.0)
        self.cams = kwargs.get('cams', None)
        self.H_orig = kwargs.get('H') # Store original H
        self.W_orig = kwargs.get('W') # Store original W
        self.batch_size = cfg.train.batch_size if self.split == 'train' else cfg.test.batch_size

        path = os.path.join(self.data_root, cfg.scene, f'transforms_{self.split}.json')
        with open(path, 'r') as f:
            meta = json.load(f)

        frames = meta['frames']
        if self.cams is not None:
            start, stop, step = self.cams
            if stop == -1:
                stop = len(frames) 
            frames = frames[start:stop:step]  

        # Calculate scaled dimensions
        self.H = int(self.H_orig * self.input_ratio)
        self.W = int(self.W_orig * self.input_ratio)
        
        camera_angle_x = float(meta['camera_angle_x'])
        # Focal length calculation should use original W
        focal_orig = .5 * self.W_orig / np.tan(.5 * camera_angle_x)
        self.focal = focal_orig * self.input_ratio

        all_rays_o = []
        all_rays_d = []
        self.rgbs = []

        for frame in frames:
            path = os.path.join(self.data_root, cfg.scene, frame['file_path'] + '.png')
            img = imageio.imread(path)
            
            if self.input_ratio != 1.0:
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            
            img = (np.array(img) / 255.).astype(np.float32)
            if img.shape[-1] == 4: # Handle RGBA
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])

            self.rgbs.append(torch.from_numpy(img))

            pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

            #pose[:3, 1:3] *= -1 # Equivalent to [x, -y, -z] for columns y and z

            rays_o, rays_d = self.get_rays(self.H, self.W, self.focal, pose)
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

        self.rays_o = torch.cat([r.view(-1, 3) for r in all_rays_o], 0)
        self.rays_d = torch.cat([r.view(-1, 3) for r in all_rays_d], 0)
        self.rays = torch.cat([self.rays_o, self.rays_d], 1) # Shape [N_total_rays, 6]
        self.rgbs = torch.cat([r.view(-1, 3) for r in self.rgbs], 0)
    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典
        """
        """
        Write your codes here.
        """
        if self.split == 'train':
            num_rays = self.rays.shape[0]
            rand_indices = torch.randint(0, num_rays, (self.batch_size,))
            
            batch_rays = self.rays[rand_indices]
            batch_rgbs = self.rgbs[rand_indices]
            
            ret = {'rays': batch_rays, 'rgbs': batch_rgbs}
        else:
            pixels_per_image = self.H * self.W
            start = index * pixels_per_image
            end = start + pixels_per_image
            batch_rays = self.rays[start:end]
            batch_rgbs = self.rgbs[start:end]

            ret = {'rays': batch_rays, 'rgbs': batch_rgbs, 'H': self.H, 'W': self.W, 'focal': self.focal}
        
        ret['near'] = np.float32(cfg.task_arg.near)
        ret['far'] = np.float32(cfg.task_arg.far)
        ret['i'] = index
        #ipdb.set_trace()
        return ret

    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        """
        Write your codes here.
        """
        if self.split == 'train':
            return 1000000
        else:
            return self.rays.shape[0] // (self.H * self.W)