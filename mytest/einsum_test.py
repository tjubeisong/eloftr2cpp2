import torch

temperature = 0.3

feat_c0 = torch.rand(10*20*30).view(10, 20, 30)

feat_c1 = torch.rand(10*20*30).view(10, 20, 30)

sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                    feat_c1) / temperature


feat_c1_t = feat_c1.permute(0, 2, 1)
y_res = torch.bmm(feat_c0, feat_c1_t)/temperature


print (torch.max(sim_matrix-y_res))


"""
n, l, c = feat_c0.shape
n, s, c = feat_c1.shape


feat_c0_r = feat_c0.view(n*l, c)
feat_c1_t = feat_c1.transpose(1, 2)

feat_c1_expanded = feat_c1_t.unsqueeze(0).expand(n*l, -1, -1)

result = torch.bmm(feat_c0_re.unsqueeze(1), feat_c1_expanded)

result = result.squeeze(1).view(n, l, s)
result = result / temperature

print(result)

print(result.shape, sim_matrix.shape)
print(result-sim_matrix)
"""
