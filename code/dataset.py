import os
import json
import yaml
import scipy
import pickle
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from utils import read_nifti



class Geometry(object):
    def __init__(self, config):
        self.v_res = config['nVoxel'][0]    # ct scan
        self.p_res = config['nDetector'][0] # projections
        self.v_spacing = np.array(config['dVoxel'])[0]    # mm
        self.p_spacing = np.array(config['dDetector'])[0] # mm

        self.DSO = config['DSO'] # mm
        self.DSD = config['DSD'] # mm

    def project(self, points, angle):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T

        d1 = self.DSO
        d2 = self.DSD
        
        coeff = (d2) / (d1 - points[:, 0]) # N,
        d_points = points[:, [2, 1]] * coeff[:, None] # [N, 2] float
        d_points /= (self.p_res * self.p_spacing)
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]

        return d_points


class Mixed_CBCT_dataset(Dataset):
    def __init__(self, dst_list, **kwargs) -> None:
        super().__init__()
        print('mixed_dataset:', dst_list)
        self.name_list = dst_list
        self.datasets = []
        for dst_name in self.name_list:
            self.datasets.append(CBCT_dataset(dst_name, **kwargs))
    
    def __len__(self):
        dst_len = [len(d) for d in self.datasets]
        return np.sum(dst_len)
    
    @property
    def num_dst(self):
        return len(self.datasets)
    
    def find_dst(self, index):
        for i, dst in enumerate(self.datasets):
            if index >= len(dst):
                index -= len(dst)
            else:
                return i, index

    def __getitem__(self, index):
        dst_idx, index = self.find_dst(index)

        dst_vec = np.zeros(self.num_dst)
        dst_vec[dst_idx] = 1.

        data_dict = self.datasets[dst_idx][index]
        data_dict['dst_vec'] = dst_vec
        data_dict['dst_name'] = self.name_list[dst_idx]
        return data_dict


class CBCT_dataset(Dataset):
    def __init__(
            self,
            dst_name,
            split='train',
            num_views=10,
            npoint=5000,
            out_res=256,
            random_views=False,
            view_offset=0
        ):
        super().__init__()
        dst_root = './data'
        
        # load dataset info
        if dst_name in ['knee_cbct']:
            data_root = os.path.join(dst_root, dst_name)
            with open(os.path.join(data_root, 'info.json'), 'r') as f:
                cfg = json.load(f)
                name_list = sorted(cfg[split])
                print('CBCT_dataset, name: {}, split: {}, len: {}.'.format(dst_name, split, len(name_list)))
        else:
            raise ValueError(dst_name)

        # load projection config
        with open(os.path.join(data_root, cfg['projection_config']), 'r') as f:
            proj_cfg = yaml.safe_load(f)
            self.geo = Geometry(proj_cfg)

        # prepare points
        if split == 'train':
            # load blocks' coordinates [train only]
            self.blocks = np.load(os.path.join(data_root, cfg['blocks']))
        else:
            # prepare sampling points
            points = np.mgrid[:out_res, :out_res, :out_res]
            points = points.astype(float) / (out_res - 1)
            points = points.reshape(3, -1)
            self.points = points.transpose(1, 0) # N, 3
        
        # other parameters
        self.out_res = out_res
        self.data_root = data_root
        self.cfg = cfg

        self.name_list = name_list
        self.npoint = npoint
        self.is_train = (split == 'train')
        self.num_views = num_views
        self.random_views = random_views
        self.view_offset = view_offset

    def __len__(self):
        return len(self.name_list)
    
    def sample_projections(self, name):
        # -- load projections
        with open(os.path.join(self.data_root, self.cfg['projections'].format(name)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projections'] # K, 1, res^2
            angles = data['angles']     # K,

        # -- sample projections
        views = np.linspace(0, len(projs), self.num_views, endpoint=False).astype(int)
        offset = np.random.randint(len(projs) - views[-1]) if self.random_views else self.view_offset
        views += offset
        projs = projs[views].astype(float) / 255.
        projs = projs[:, None, ...]
        angles = angles[views]

        # -- de-normalization [required for mixed dataset]
        # projs = projs * self.cfg['projection_norm'] / 0.2
        
        return projs, angles
    
    def load_ct(self, name):
        image = read_nifti(os.path.join(self.data_root, self.cfg['image'].format(name)))
        image = image.astype(np.float32) / 255.
        if self.out_res == 128:
            image = scipy.ndimage.zoom(image, 0.5, order=3, prefilter=False)
        elif self.out_res != 256:
            raise ValueError
        return image
    
    def load_block(self, name, b_idx):
        path = os.path.join(self.data_root, self.cfg['image_block'].format(name, b_idx))
        return np.load(path)

    def sample_points(self, points, values=None):
        choice = np.random.choice(len(points), size=self.npoint, replace=False)
        points = points[choice]
        if values is not None:
            values = values[choice]
            values = values.astype(float) / 255.
            return points, values
        else: return points

    def __getitem__(self, index):
        name = self.name_list[index]

        # -- load projections
        projs, angles = self.sample_projections(name)

        # -- load sampling points
        if not self.is_train:
            points = self.points
            image = self.load_ct(name)
            p_gt = np.zeros(len(points))
        else:
            b_idx = np.random.randint(len(self.blocks))
            block_values = self.load_block(name, b_idx)
            block_coords = self.blocks[b_idx] # N, 3
            points, p_gt = self.sample_points(block_coords, block_values)

        # -- project points and view direction
        proj_points = []
        for a in angles:
            p = self.geo.project(points, a)
            proj_points.append(p)
        proj_points = np.stack(proj_points, axis=0) # M, N, 2
        
        # -- normalize points
        points = deepcopy(points) # ~[0, 1]
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= 2 # => [-1, 1]

        # -- normalize viewing angles
        angles = np.array(angles, dtype=float) # ~[0, +pi]
        angles = angles / np.pi * 2 - 1 # => [-1, 1]

        # -- collect data
        ret_dict = {
            'name': name,
            'points': points,           # 3D points
            'angles': angles[:, None],  # angles
            'proj_points': proj_points, # projected points
            'projs': projs,             # 2D projections
            'p_gt': p_gt[None, :]       # labels
        }
        if not self.is_train:
            ret_dict['image'] = image

        return ret_dict


if __name__ == '__main__':
    dst = CBCT_dataset(dst_name='knee_zhao', random_views=True, num_views=10)
    item = dst[0]
    import pdb; pdb.set_trace()