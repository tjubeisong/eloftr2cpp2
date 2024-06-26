import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger

from typing import Dict, Union, Tuple, List

from ..utils.full_config import full_default_cfg



# # @torch.no_grad()
# @torch.jit.script
# def get_fine_ds_match(conf_matrix, data:Dict[str, torch.Tensor], W, WW, C, scale, fp16):
#     # W, WW, C, scale = self.W, self.WW, self.C, self.scale
#     m, _, _ = conf_matrix.shape

#     conf_matrix = conf_matrix.reshape(m, -1)[:len(data['mconf']),...]
#     val, idx = torch.max(conf_matrix, dim = -1)
#     idx = idx[:,None]
#     idx_l, idx_r = idx // WW, idx % WW

#     data.update({'idx_l': idx_l, 'idx_r': idx_r})
    

#     # if self.fp16:
#     # if fp16:
#     #     grid = create_meshgrid(W, W, False, conf_matrix.device, dtype=torch.float16) - W // 2 + 0.5 # kornia >= 0.5.1
#     # else:
#     #     grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
#     grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
#     grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
#     delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
#     delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

#     scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
#     scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale

#     # if torch.is_tensor(scale0) and scale0.numel() > 1: # scale0 is a tensor
#     #     mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
#     #     mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
#     # else: # scale0 is a float
#     #     mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0)).reshape(-1, 2)
#     #     mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1)).reshape(-1, 2)

#     mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0)).reshape(-1, 2)
#     mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1)).reshape(-1, 2)

    
#     data.update({
#         "mkpts0_c": mkpts0_f,
#         "mkpts1_c": mkpts1_f
#     })
#     return data
    
class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()
        config = full_default_cfg
        self.config = config
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.fp16 = config['half']
        self.M = 0
        self.W = 0
        self.WW = 0
        self.C = 0
        self.scale = 0.0

    def forward(self, feat_0, feat_1, hw0_i:List[int], hw0_f:List[int], mkpts0_c, mkpts1_c, mconf, hw0_c:List[int], bs:int):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW))
        scale = hw0_i[0] / hw0_f[0]
        # self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale
        self.M = M
        self.W = W
        self.WW = WW
        self.C = C
        # print("===================================================", scale)
        self.scale = scale

        # corner case: if no coarse matches found
        # print("M = ==== === === ", M)
        fine_res = {}
        if M == 0:
            # assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            # data.update({
            #     'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
            #     'mkpts0_f': data['mkpts0_c'],
            #     'mkpts1_f': data['mkpts1_c'],
            # })
            # return

            fine_res.update({
                'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                'mkpts0_f': mkpts0_c,
                'mkpts1_f': mkpts1_c
            })
            return fine_res

            # return torch.empty(0, WW, WW, device=feat_0.device), mkpts0_c, mkpts1_c

        # compute pixel-level confidence matrix
        # with torch.autocast(enabled=True, device_type='cuda'):
        #     feat_f0, feat_f1 = feat_0[...,:-self.local_regress_slicedim], feat_1[...,:-self.local_regress_slicedim]
        #     feat_ff0, feat_ff1 = feat_0[...,-self.local_regress_slicedim:], feat_1[...,-self.local_regress_slicedim:]
        #     feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5
        #     conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
        #     conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim)**.5)

        feat_f0, feat_f1 = feat_0[...,:-self.local_regress_slicedim], feat_1[...,:-self.local_regress_slicedim]
        feat_ff0, feat_ff1 = feat_0[...,-self.local_regress_slicedim:], feat_1[...,-self.local_regress_slicedim:]
        feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5
        conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
        conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim)**.5)

        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W+2, self.W+2)
        softmax_matrix_f = softmax_matrix_f[...,1:-1,1:-1].reshape(M, self.WW, self.WW)

        # for fine-level supervision
        # if self.training:
        #     data.update({'sim_matrix_ff': conf_matrix_ff})
        #     data.update({'conf_matrix_f': softmax_matrix_f})


        # # ======================================================================================
        # compute pixel-level absolute kpt coords
        
        # tm = get_fine_ds_match(softmax_matrix_f, __data, self.W, self.WW, self.C, self.scale, self.fp16)
        # data.update(**tm)
        
        # W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = softmax_matrix_f.shape

        conf_matrix = softmax_matrix_f.reshape(m, -1)[:len(mconf),...]
        val, idx = torch.max(conf_matrix, dim = -1)
        idx = idx[:,None]
        idx_l, idx_r = idx // self.WW, idx % self.WW

        # data.update({'idx_l': idx_l, 'idx_r': idx_r})
        fine_res.update({'idx_l': idx_l, 'idx_r': idx_r})
        
        grid = create_meshgrid(self.W, self.W, False, conf_matrix.device) - self.W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        scale = hw0_i[0]/hw0_c[0]
        # scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        # scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale

        scale0, scale1 = scale, scale
        mkpts0_f = (mkpts0_c[:,None,:] + (delta_l * scale0)).reshape(-1, 2)
        mkpts1_f = (mkpts1_c[:,None,:] + (delta_r * scale1)).reshape(-1, 2)

        
        # data.update({
        #     "mkpts0_c": mkpts0_f,
        #     "mkpts1_c": mkpts1_f
        # })
 
        fine_res['mkpts0_c'] = mkpts0_f
        fine_res['mkpts1_c'] = mkpts1_f

        # #=======================================================================================

        # generate seconde-stage 3x3 grid
        idx_l, idx_r = fine_res['idx_l'], fine_res['idx_r']
        # print('idx_l.numel = ', idx_l.numel())
        

        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
        m_ids = m_ids[:len(mconf)]
        idx_r_iids, idx_r_jids = idx_r // W, idx_r % W

        m_ids, idx_l, idx_r_iids, idx_r_jids = m_ids.reshape(-1), idx_l.reshape(-1), idx_r_iids.reshape(-1), idx_r_jids.reshape(-1)
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long) # [1, 3, 3, 2]

        
        m_ids = m_ids[...,None,None].expand(-1, 3, 3)
        idx_l = idx_l[...,None,None].expand(-1, 3, 3) # [m, k, 3, 3]

        idx_r_iids = idx_r_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]


        
        
        
        if idx_l.numel() == 0:
            # data.update({
            #     'mkpts0_f': data['mkpts0_c'],
            #     'mkpts1_f': data['mkpts1_c'],
            # })
            # return
            
            fine_res.update({
                'mkpts0_f': mkpts0_c,
                'mkpts1_f': mkpts1_c,
            })
            return fine_res

        # compute second-stage heatmap
        conf_matrix_ff = conf_matrix_ff.reshape(M, self.WW, self.W+2, self.W+2)
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l, idx_r_iids, idx_r_jids]
        conf_matrix_ff = conf_matrix_ff.reshape(-1, 9)
        conf_matrix_ff = F.softmax(conf_matrix_ff / self.local_regress_temperature, -1)
        heatmap = conf_matrix_ff.reshape(-1, 3, 3)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]

        # print("data[bs] == 1", bs)
        # if data['bs'] == 1:
        #     scale1 = scale * data['scale1'] if 'scale0' in data else scale
        # else:
        #     scale1 = scale * data['scale1'][data['b_ids']][:len(data['mconf']), ...][:,None,:].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale

        
        # scale1 = scale * data['scale1'] if 'scale0' in data else scale
        
        # compute subpixel-level absolute kpt coords
        # self.get_fine_match_local(coords_normalized, data, scale1)

        
        # mkpts0_f and mkpts1_f
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normalized * (3 // 2) * scale1)

        fine_res.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })

        return fine_res

    # def get_fine_match_local(self, coords_normed, data, scale1):
    #     W, WW, C, scale = self.W, self.WW, self.C, self.scale

    #     mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']

    #     # mkpts0_f and mkpts1_f
    #     mkpts0_f = mkpts0_c
    #     mkpts1_f = mkpts1_c + (coords_normed * (3 // 2) * scale1)

    #     data.update({
    #         "mkpts0_f": mkpts0_f,
    #         "mkpts1_f": mkpts1_f
    #     })
