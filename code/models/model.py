import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.unet import UNet
from models.point_classifier import SurfaceClassifier



def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]


class MLP(nn.Module):
    def __init__(self, mlp_list, use_bn=False):
        super().__init__()

        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn:
                layers += [nn.BatchNorm2d(mlp_list[i + 1])]
            layers += [nn.LeakyReLU(inplace=True),]
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DIF_Net(nn.Module):
    def __init__(self, num_views, combine, mid_ch=128):
        super().__init__()
        self.combine = combine

        self.image_encoder = UNet(1, mid_ch)

        if self.combine == 'mlp':
            self.view_mixer = MLP([num_views, num_views // 2, 1])
        
        self.point_classifier = SurfaceClassifier(
            [mid_ch, 256, 64, 16, 1],
            no_residual=False
        )
        print(f'DIF_Net, mid_ch: {mid_ch}, combine: {self.combine}')

    def forward(self, data, is_eval=False, eval_npoint=100000):
        # projection encoding
        projs = data['projs'] # B, M, C, W, H
        b, m, c, w, h = projs.shape
        projs = projs.reshape(b * m, c, w, h) # B', C, W, H
        proj_feats = self.image_encoder(projs)
        proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_) # B, M, C, W, H

        # point-wise forward
        if not is_eval:
            p_pred = self.forward_points(proj_feats, data)
            p_gt = data['p_gt']
            return p_pred, p_gt
        
        else:
            total_npoint = data['proj_points'].shape[2]
            n_batch = int(np.ceil(total_npoint / eval_npoint))

            pred_list = []
            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)
                p_pred = self.forward_points(
                    proj_feats, {
                        'proj_points': data['proj_points'][..., left:right, :],
                        'points': data['points'][..., left:right, :],
                    }
                ) # B, C, N
                pred_list.append(p_pred)

            pred = torch.cat(pred_list, dim=2)
            return pred
    
    def forward_points(self, proj_feats, data):
        n_view = proj_feats[0].shape[1]

        # 1. query view-specific features
        p_list = []
        for i in range(n_view):
            f_list = []
            for proj_f in proj_feats:
                feat = proj_f[:, i, ...] # B, C, W, H
                p = data['proj_points'][:, i, ...] # B, N, 2
                p_feats = index_2d(feat, p) # B, C, N
                f_list.append(p_feats)
            p_feats = torch.cat(f_list, dim=1)
            p_list.append(p_feats)
        p_feats = torch.stack(p_list, dim=-1) # B, C, N, M

        # 2. cross-view fusion
        if self.combine == 'max':
            p_feats = F.max_pool2d(p_feats, (1, n_view))
            p_feats = p_feats.squeeze(-1) # B, C, N
        elif self.combine == 'mlp':
            p_feats = p_feats.permute(0, 3, 1, 2)
            p_feats = self.view_mixer(p_feats)
            p_feats = p_feats.squeeze(1)
        else:
            raise NotImplementedError

        # 3. point-wise classification
        p_pred = self.point_classifier(p_feats)
        return p_pred
