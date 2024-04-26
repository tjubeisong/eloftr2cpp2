import os
import sys
pathonpath = os.path.abspath(os.path.dirname((os.path.abspath('.'))))
print(pathonpath)
sys.path.insert(0, pathonpath)
os.chdir("..")
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

# You can choose model type in ['full', 'opt']
# model_type = 'full' # 'full' for best quality, 'opt' for best efficiency

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
# precision = 'fp32' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

# You can also change the deepcopy
# if model_type == 'full':
#     _default_cfg = deepcopy(full_default_cfg)
# elif model_type == 'opt':
    # _default_cfg = deepcopy(opt_default_cfg)
    
_default_cfg = deepcopy(full_default_cfg)

# if precision == 'mp':
#     _default_cfg['mp'] = True
# elif precision == 'fp16':
#     _default_cfg['half'] = True

# print(_default_cfg)
# matcher = LoFTR(config=_default_cfg)

matcher = LoFTR()

        
# matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
# matcher = reparameter(matcher) # no reparameterization will lead to low performance
 
# if precision == 'fp16':
#     matcher = matcher.half()


matcher = matcher.eval().cuda()

# Load example images
img0_pth = "assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
img1_pth = "assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))  # input size shuold be divisible by 32
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

# if precision == 'fp16':
#     img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
#     img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
# else:
#     img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
#     img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

print(img0.shape, img1.shape)

# print(_default_cfg)


# class MyModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x**2

# mymodel = MyModel()

# msm = torch.jit.trace(mymodel, torch.tensor([2]))
# msm.save("tmed_model.pt")

# script start
sm = torch.jit.script(matcher)
print("res = ", sm(batch))
# print(type(img0), type(img1))
# sm = matcher(batch)
# sm = torch.jit.trace(matcher, batch)
sm.save("traced_eloftr_model.zip")
# # script end

# Inference with EfficientLoFTR and get prediction
# with torch.no_grad():
#     if precision == 'mp':
#         with torch.autocast(enabled=True, device_type='cuda'):
#             matcher(batch)
#     else:
#         matcher(batch)
#         print("-------------------------------------------------------2-------------------------------------------------", precision)
        
#     mkpts0 = batch['mkpts0_f'].cpu().numpy()
#     mkpts1 = batch['mkpts1_f'].cpu().numpy()
#     mconf = batch['mconf'].cpu().numpy()

# # Draw
# if model_type == 'opt':
#     # print(mconf.max())
#     mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

# import numpy as np
# print("mkpts0", mkpts0.shape)
# print("mkpts1", mkpts1.shape)
# print("mconf", mconf.shape)

# color = cm.jet(mconf)
# text = [
#     'LoFTR',
#     'Matches: {}'.format(len(mkpts0)),
#     ]
# fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)

# fig.savefig("res--0422-v5.jpg", dpi=300)

