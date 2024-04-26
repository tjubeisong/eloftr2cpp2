"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module
import torch.nn.functional as F
# from einops.einops import rearrange

if hasattr(F, 'scaled_dot_product_attention'):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False

def crop_feature(query, key, value, x_mask, source_mask):
    mask_h0, mask_w0, mask_h1, mask_w1 = x_mask[0].sum(-2)[0], x_mask[0].sum(-1)[0], source_mask[0].sum(-2)[0], source_mask[0].sum(-1)[0]
    query = query[:, :mask_h0, :mask_w0, :]
    key = key[:, :mask_h1, :mask_w1, :]
    value = value[:, :mask_h1, :mask_w1, :]
    return query, key, value, mask_h0, mask_w0

def pad_feature(m, mask_h0, mask_w0, x_mask):
    bs, L, H, D = m.size()
    m = m.view(bs, mask_h0, mask_w0, H, D)
    if mask_h0 != x_mask.size(-2):
        m = torch.cat([m, torch.zeros(m.size(0), x_mask.size(-2)-mask_h0, x_mask.size(-1), H, D, device=m.device, dtype=m.dtype)], dim=1)
    elif mask_w0 != x_mask.size(-1):
        m = torch.cat([m, torch.zeros(m.size(0), x_mask.size(-2), x_mask.size(-1)-mask_w0, H, D, device=m.device, dtype=m.dtype)], dim=2)
    return m
    

def my_rearrange_type1(x):
    x = torch.chunk(x, chunks=8, dim=3)
    x = torch.stack(x, dim=1)
    x = torch.flatten(x, start_dim=2, end_dim=3)
    return x

def my_rearrange_type2(x):
    x = torch.chunk(x, chunks=8, dim=3)
    x = torch.stack(x, dim=3)
    x = torch.flatten(x, start_dim=1, end_dim=2)
    return x



def _forward(query, key, value, q_mask=None, kv_mask=None, flash=False, fp32=False):
    if q_mask is not None:
        query, key, value, mask_h0, mask_w0 = crop_feature(query, key, value, q_mask, kv_mask)

    # if self.flash:
    #     query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim), [query, key, value])
    # else:
    #     query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim), [query, key, value])
    
    
    
    if flash:
        query = my_rearrange_type1(query)
        key = my_rearrange_type1(key)
        value = my_rearrange_type1(value)
    else:
        query = my_rearrange_type2(query)
        key = my_rearrange_type2(key)
        value = my_rearrange_type2(value)
     
    """
    print("flash, ", flash)
    print("query", query.shape)jit sci
    if self.flash:
        query = rearrange(query, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
        key = rearrange(key, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
        value = rearrange(value, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
    else:
        query = rearrange(query, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
        key = rearrange(key, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
        value = rearrange(value, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
    """

    # m = self.attention(query, key, value, q_mask=None, kv_mask=None)

    if flash and not fp32:
        args = [x.contiguous() for x in [query, key, value]]
        with sdp_kernel(enable_math= False, enable_flash= True, enable_mem_efficient= False):
            out = F.scaled_dot_product_attention(*args)
    elif flash:
        args = [x.contiguous() for x in [query, key, value]]
        out = F.scaled_dot_product_attention(*args)
    else:
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        # Compute the attention and the weighted average
        softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        out = torch.einsum("nlsh,nshd->nlhd", A, value)
    m = out

    if flash:
        m = rearrange(m, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

    if q_mask is not None:
        m = pad_feature(m, mask_h0, mask_w0, q_mask)
    
    return m        

class Attention(Module):
    def __init__(self, no_flash=False, nhead=8, dim=256, fp32=False):
        super().__init__()
        self.flash = FLASH_AVAILABLE and not no_flash
        self.nhead = nhead
        self.dim = dim
        self.fp32 = fp32

    
   
    # def attention(self, query, key, value, q_mask=None, kv_mask=None):
    def attention(self, query, key, value):
        # assert q_mask is None and kv_mask is None, "Not support generalized attention mask yet."
        if self.flash and not self.fp32:
            args = [x.contiguous() for x in [query, key, value]]
            with sdp_kernel(enable_math= False, enable_flash= True, enable_mem_efficient= False):
                out = F.scaled_dot_product_attention(*args)
        elif self.flash:
            args = [x.contiguous() for x in [query, key, value]]
            out = F.scaled_dot_product_attention(*args)
        else:
            QK = torch.einsum("nlhd,nshd->nlsh", query, key)
    
            # Compute the attention and the weighted average
            softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)

            out = torch.einsum("nlsh,nshd->nlhd", A, value)
        return out
   
   
    
    # def forward(self, query, key, value, q_mask=None, kv_mask=None):
    def forward(self, query, key, value):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            if FLASH_AVAILABLE: # pytorch scaled_dot_product_attention
                queries: [N, H, L, D]
                keys: [N, H, S, D]
                values: [N, H, S, D]
            else:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
    
        bs = query.size(0)
        # if bs == 1 or q_mask is None: 
            # m = _forward(query, key, value, q_mask=q_mask, kv_mask=kv_mask)
            # if q_mask is not None:
            #     query, key, value, mask_h0, mask_w0 = crop_feature(query, key, value, q_mask, kv_mask)           
        query = my_rearrange_type2(query)
        key = my_rearrange_type2(key)
        value = my_rearrange_type2(value)
        
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        out = torch.einsum("nlsh,nshd->nlhd", A, value)
        m = out
        # if q_mask is not None:
        #     m = pad_feature(m, mask_h0, mask_w0, q_mask)
            
        # else: # for faster trainning with padding mask while batch size > 1
        #     m_list = []
        #     for i in range(bs):
        #         # m_list.append(_forward(query[i:i+1], key[i:i+1], value[i:i+1], q_mask=q_mask[i:i+1], kv_mask=kv_mask[i:i+1]))
                
        #         # _query, _key, _value, _q_mask, _kv_mask = query[i:i+1], key[i:i+1], value[i:i+1], q_mask[i:i+1], kv_mask[i:i+1]
        #         _query, _key, _value = query[i:i+1], key[i:i+1], value[i:i+1]
                
        #         # if _q_mask is not None:
        #         #     _query, _key, _value, _mask_h0, _mask_w0 = crop_feature(_query, _key, _value, _q_mask, _kv_mask)           
        #         _query = my_rearrange_type2(_query)
        #         _key = my_rearrange_type2(_key)
        #         _value = my_rearrange_type2(_value)
           
        #         _QK = torch.einsum("nlhd,nshd->nlsh", _query, _key)

        #         _softmax_temp = 1. / _query.size(3)**.5  # sqrt(D)
        #         _A = torch.softmax(_softmax_temp * _QK, dim=2)
        #         _out = torch.einsum("nlsh,nshd->nlhd", _A, _value)
        #         _m = _out
        #         # if _q_mask is not None:
        #         #     _m = pad_feature(_m, _mask_h0, _mask_w0, _q_mask)
        #         m_list.append(_m)
        #     m = torch.cat(m_list, dim=0)
        
        return m

