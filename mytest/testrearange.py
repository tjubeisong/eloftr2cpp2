import numpy as np
import torch
from einops.einops import rearrange

n, h, w, d = 3, 300, 600, 10

a = torch.arange(n*h*w*d).reshape(n, h, w, d)



# b = rearrange(a, 'n h w (nd d) -> n nd (h w) d', nd=2)
# bb = rearrange(a, 'n h w (nd d) -> n (h w) nd d', nd=2)

'''print(b.shape)
print('-- 3, 2, 180000, 5')

p
c = torch.chunk(torch.tensor(a), 2, 3)
print(c[0].shape, c[1].shape)

d = torch.stack(c, dim=1)
print(d.shape)

e = torch.flatten(d, start_dim=2, end_dim=3)

print(e.shape)

h = e-b

print(h.max())


def my_rearrange_type1(x, nd):
    x = torch.chunk(x, chunks=2, dim=3)
    x = torch.stack(x, dim=1)
    x = torch.flatten(x, start_dim=2, end_dim=3)
    return x

def my_rearrange_type2(x, nd):
    x = torch.chunk(x, chunks=2, dim=3)
    x = torch.stack(x, dim=3)
    x = torch.flatten(x, start_dim=1, end_dim=2)
    return x

res1 =  my_rearrange_type1(a, 2)
res2 =  my_rearrange_type2(a, 2)
print(b.shape, res1.shape)
print(torch.equal(b, res1))

print(bb.shape, res2.shape)
print(torch.equal(bb, res2))
print(torch.max(bb-res2))

'''

h0c = 10
w0c = 30
h1c = 5
w1c = 120
ori = torch.arange(n*h*w).reshape(n, h, w)
mask1 = rearrange(ori, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                h0c = h0c, 
                w0c = w0c,
                h1c = h1c, 
                w1c = w1c
) 

mask2 = torch.stack(torch.chunk(ori, chunks = h1c, dim = 2), dim=2)
mask2 = torch.stack(torch.chunk(mask2, chunks = h0c, dim = 1), dim=1)

print(mask1.shape)
print(mask2.shape)


print(torch.max(mask1-mask2))

mask3 = torch.flatten(mask2, start_dim=3, end_dim=4)
mask3 = torch.flatten(mask3, start_dim=1, end_dim=2)

print('mask3.shape', mask3.shape)
print(torch.max(ori-mask3))


