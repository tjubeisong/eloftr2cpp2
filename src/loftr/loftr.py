import torch
import torch.nn as nn
from typing import Dict, Union, List, Tuple
from copy import deepcopy
# from einops.einops import rearrange

from .utils.full_config import full_default_cfg

from .backbone import build_backbone
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
# from src.utils.misc import detect_NaN

from loguru import logger

def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    return matcher




def mysqrt(feat):
    return feat / feat.shape[-1]**.5


class LoFTR(nn.Module):
    # def __init__(self, config, profiler=None):
    def __init__(self, profiler=None):
        super().__init__()
        # Misc
        # self.config = config
        config = deepcopy(full_default_cfg)
        # print(config)
        self.config = config
        self.profiler = profiler        

        # Modules
        self.backbone = build_backbone(config)            
        self.loftr_coarse = LocalFeatureTransformer()
        # self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.coarse_matching = CoarseMatching()
        self.fine_preprocess = FinePreprocess()

        # self.fine_matching = FineMatching(config)

        self.mul = self.config['resolution'][0] // self.config['resolution'][1]
        print("============================inited finished======================")
        
        

    def forward(self, data:Dict[str, torch.Tensor]):
        """         print("----------------------------6-------------------------")
        # print(config)
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        print("----------------------------7-------------------------")

        # 1. Local Feature CNN
        # data.update({
        #     'bs': data['image0'].size(0),
        #     'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        # })

        bs = data['image0'].size(0)
        hw0_i = data['image0'].shape[2:]
        hw1_i = data['image1'].shape[2:]

        # if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
        #     ret_dict = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
        #     feats_c = ret_dict['feats_c']
        #     data.update({
        #         'feats_x2': ret_dict['feats_x2'],
        #         'feats_x1': ret_dict['feats_x1'],
        #     })
        #     (feat_c0, feat_c1) = feats_c.split(data['bs'])
        # else:  # handle different input shapes
        #     ret_dict0, ret_dict1 = self.backbone(data['image0']), self.backbone(data['image1'])
        #     feat_c0 = ret_dict0['feats_c']
        #     feat_c1 = ret_dict1['feats_c']
        #     data.update({
        #         'feats_x2_0': ret_dict0['feats_x2'],
        #         'feats_x1_0': ret_dict0['feats_x1'],
        #         'feats_x2_1': ret_dict1['feats_x2'],
        #         'feats_x1_1': ret_dict1['feats_x1'],
        #     })

        ret_dict0, ret_dict1 = self.backbone(data['image0']), self.backbone(data['image1'])
        feat_c0 = ret_dict0['feats_c']
        feat_c1 = ret_dict1['feats_c']
        
        # data.update({
        #     'feats_x2_0': ret_dict0['feats_x2'],
        #     'feats_x1_0': ret_dict0['feats_x1'],
        #     'feats_x2_1': ret_dict1['feats_x2'],
        #     'feats_x1_1': ret_dict1['feats_x1'],
        # })
        
        feats = [ret_dict0['feats_x2'], ret_dict0['feats_x1'], ret_dict1['feats_x2'], ret_dict1['feats_x1']]
        # mul = self.config['resolution'][0] // self.config['resolution'][1]
        # data.update({
        #     'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
        #     'hw0_f': [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul] ,
        #     'hw1_f': [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul]
        # })

        mul = self.mul
        hw0_c, hw1_c = [feat_c0.shape[2:], feat_c1.shape[2:]]
        hw0_f = [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul]
        hw1_f = [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul]
        
        # 2. coarse-level loftr module
        # mask_c0 = mask_c1 = None  # mask is useful in training
        # if 'mask0' in data:
        #     mask_c0, mask_c1 = data['mask0'], data['mask1']

        # print("-------------------------------------------------------1-------------------------------------------------")   
        
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1)
        
        
        # feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
        
        feat_c0 = feat_c0.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        feat_c1 = feat_c1.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        
        # detect NaN during mixed precif self.config['replace_nan'] and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))):
        #  ision training
        # if self.config['replace_nan'] and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
        #     detect_NaN(feat_c0, feat_c1)
        
        # 3. match coarse-level
        # self.coarse_matching(feat_c0, feat_c1, data, 
        #                         mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0, 
        #                         mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1
        #                         )
        coarse_res = self.coarse_matching(feat_c0, feat_c1, hw0_c, hw1_c, hw0_f, hw1_f, hw0_i, hw1_i)
        print('coarse_res = ', coarse_res)
        print("-=-=-= ", feat_c0.device)
        
        # prevent fp16 overflow during mixed precision training
        
        # can't be scripted
        # feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
        #                [feat_c0, feat_c1])

        # feat_c0, feat_c1 = map(mysqrt, [feat_c0, feat_c1])
        feat_c0, feat_c1 = feat_c0 / feat_c0.shape[-1]**.5, feat_c1 / feat_c1.shape[-1]**.5
        print("-------------------------------------------------------2-------------------------------------------------")  
        
        # 4. fine-level refinement
        print('data = ===================================', data.keys())
      
        feat_f0_unfold, feat_f1_unfold, _W = self.fine_preprocess(feat_c0, feat_c1, data['hw0_f'], data['hw0_c'], data['b_ids'],
                        data['hw0_i'], data['hw1_i'], data['feats_x2_0'], data['feats_x2_1'], data['feats_x1_0'], 
                        data['feats_x1_1'], data['i_ids'], data['j_ids'], data['hw1_c'])

      
        
        '''
        
        del data['feats_x2_0'], data['feats_x1_0'], data['feats_x2_1'], data['feats_x1_1']
        data.update({'W': _W})
        
        print("-------------------------------------------------------3-------------------------------------------------")         
        # detect NaN during mixed precision training
        # if self.config['replace_nan'] and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))):
        #     detect_NaN(feat_f0_unfold, feat_f1_unfold)
        
        del feat_c0, feat_c1, mask_c0, mask_c1
        print("-------------------------------------------------------4-------------------------------------------------")
        
        # 5. match fine-level            

        
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        '''
        return data
        
    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

