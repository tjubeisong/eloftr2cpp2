import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.einops import rearrange, repeat

from loguru import logger

from ..utils.full_config import full_default_cfg

from typing import Dict, Union, Tuple, List

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FinePreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        print("============================================11111111111111111=====================")
        config = full_default_cfg
        self.config = config
        block_dims = config['backbone']['block_dims']
        self.W = int(self.config['fine_window_size'])
        self.fine_d_model = int(block_dims[0])

        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        self._reset_parameters()
        
        print("============================================2222222222222222222=====================", self.fine_d_model)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def inter_fpn(self, feat_c, x2, x1, stride):
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2., mode='bilinear', align_corners=False)

        x2 = self.layer2_outconv(x2)
        x2 = self.layer2_outconv2(x2+feat_c)
        x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)

        x1 = self.layer1_outconv(x1)
        x1 = self.layer1_outconv2(x1+x2)
        x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)
        return x1
    
    # def forward(self, feat_c0, feat_c1, hw0_f, hw0_c, b_ids, hw0_i, hw1_i, feats_x2_0, feats_x2_1,
    #     feats_x1_0, feats_x1_1, i_ids, j_ids, hw1_c):
        print("============================================3333333333333333333333=====================")
        #  data:Dict[str, Union[int, bool, str, torch.Tensor, Tuple[int], Dict[str, float], List[int]]]):
    
        # res = Dict[str, Tuple]

        
        W = self.W

        # stride = data['hw0_f'][0] // data['hw0_c'][0]
        stride = hw0_f[0] // hw0_c[0]
        
        # res['W'] = W
        # data.update({'W': W})
        
        # if data['b_ids'].shape[0] == 0:
        if b_ids.shape[0] == 0:
        #     # print("***********ahahahhahaahhaahahhhahh****************")
            feat0 = torch.empty(0, int(self.W**2), self.fine_d_model, device=feat_c0.device)
            feat1 = torch.empty(0, int(self.W**2), self.fine_d_model, device=feat_c0.device)
            # feat0 = torch.empty(0, int(8**2), 64, device=feat_c0.device)
            # feat1 = torch.empty(0, int(8**2), 64, device=feat_c0.device)
            
            return feat0, feat1
        
        print("-------------------------------------------------------3.1-------------------------------------------------") 
        # print(data['hw0_i'], data['hw1_i'])
        # if data['hw0_i'] == data['hw1_i']:
        print(hw0_i, hw1_i)
        # if hw0_i == hw1_i:
        #     print("")
        #     feat_c = rearrange(torch.cat([feat_c0, feat_c1], 0), 'b (h w) c -> b c h w', h=hw0_c[0]) # 1/8 feat
        #     x2 = feats_x2 # 1/4 feat
        #     x1 = feats_x1 # 1/2 feat

        #     # del data['feats_x2'], data['feats_x1']
        #     res['feats_x2'] = None
        #     res['feats_x1'] = None
            

        #     # 1. fine feature extraction
        #     x1 = self.inter_fpn(feat_c, x2, x1, stride)                    
        #     feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

        #     # 2. unfold(crop) all local windows
        #     feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
        #     feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
        #     feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
        #     feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)

        #     # 3. select only the predicted matches
        #     feat_f0 = feat_f0[b_ids, data['i_ids']]  # [n, ww, cf]
        #     feat_f1 = feat_f1[b_ids, data['j_ids']]

        #     return feat_f0, feat_f1  
        # else:  # handle different input shapes
        # feat_c0, feat_c1 = rearrange(feat_c0, 'b (h w) c -> b c h w', h=hw0_c[0]), rearrange(feat_c1, 'b (h w) c -> b c h w', h=hw1_c[0]) # 1/8 feat
        
        feat_c0 = feat_c0.permute(0, 2, 1)
        feat_c0 = torch.stack(feat_c0.chunk(chunks=hw0_c[0], dim=2), dim=2)
        feat_c1 = feat_c1.permute(0, 2, 1)
        feat_c1 = torch.stack(feat_c1.chunk(chunks=hw1_c[0], dim=2), dim=2)
        
        # x2_0, x2_1 = data['feats_x2_0'], data['feats_x2_1'] # 1/4 feat
        # x1_0, x1_1 = data['feats_x1_0'], data['feats_x1_1'] # 1/2 feat
        
        x2_0, x2_1 = feats_x2_0, feats_x2_1 # 1/4 feat
        x1_0, x1_1 = feats_x1_0, feats_x1_1 # 1/2 feat
        

        # del data['feats_x2_0'], data['feats_x1_0'], data['feats_x2_1'], data['feats_x1_1']
        # res['feats_x2_0'] = None
        # res['feats_x1_0'] = None
        # res['feats_x2_1'] = None
        # res['feats_x1_1'] = None
        

        # 1. fine feature extraction
        feat_f0, feat_f1 = self.inter_fpn(feat_c0, x2_0, x1_0, stride), self.inter_fpn(feat_c1, x2_1, x1_1, stride)
        print('feat_f0 = ', feat_f0.size(), 'stride = ', stride, 'W = ', W)
        # 2. unfold(crop) all local windows
        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=(stride, stride), padding=(0, 0))
        # feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
        print("-------------------------------------------------------3.2-------------------------------------------------") 
        feat_f0 = torch.stack(feat_f0.split(W**2, dim=1)).permute(0, 3, 2, 1)
        
        feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=(stride, stride), padding=(1, 1))
        # feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)
        feat_f1 = torch.stack(feat_f1.split((W+2)**2, dim=1)).permute(0, 3, 2, 1)

        # 3. select only the predicted matches
        # feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        # feat_f1 = feat_f1[data['b_ids'], data['j_ids']]
        
        feat_f0 = feat_f0[b_ids, i_ids]  # [n, ww, cf]
        feat_f1 = feat_f1[b_ids, j_ids]

        return feat_f0, feat_f1, W
        