import torch
import numpy as np
from tqdm import tqdm

# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
from metrics.StructuralLosses.match_cost import match_cost
from metrics.StructuralLosses.nn_distance import nn_distance


def compute_cd(x, y, reduce_func=torch.mean):
    if x.is_cuda:
        d1, d2 = nn_distance(x, y)
    else:
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind = torch.arange(0, num_points).to(sample).long()
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        p = rx.transpose(2, 1) + ry - 2 * zz
        d1, d2 = p.min(1)[0], p.min(2)[0]
    return reduce_func(d1, dim=1) + reduce_func(d2, dim=1)


def compute_emd(x, y):
    return match_cost(x, y) / float(x.size(1))


def compute_pairwise_cd_emd(x, y, batch_size=32):
    cd, emd = [], []
    y = y.contiguous()
    for xi in tqdm(x):
        cdi, emdi = [], []
        for yb in torch.split(y, batch_size, dim=0):
            xb = xi.view(1, -1, 3).expand_as(yb).contiguous()
            cdi.append(compute_cd(xb, yb).view(1, -1))
            emdi.append(compute_emd(xb, yb).view(1, -1))
        cd.append(torch.cat(cdi, dim=1))
        emd.append(torch.cat(emdi, dim=1))
    cd, emd = torch.cat(cd, dim=0), torch.cat(emd, dim=0)
    return cd, emd


def compute_mmd_cov(d):
    NS, NR = d.shape
    min_val_fromsmp, min_idx = d.min(dim=1)
    min_val, _ = d.min(dim=0)
    mmd = min_val.mean()
    cov = torch.tensor(min_idx.unique().view(-1).size(0) / float(NR)).to(d)
    return mmd, cov


def compute_knn(dxx, dxy, dyy, k):
    X, Y = dxx.size(0), dyy.size(0)
    label = torch.cat((torch.ones(X), torch.zeros(Y))).to(dxx)
    mx = torch.cat((dxx, dxy), dim=1)
    my = torch.cat((dxy.t(), dyy), dim=1)
    m = torch.cat((mx, my), dim=0)
    m = m + torch.diag(torch.ones(X + Y) * float("inf")).to(dxx)
    _, idx = m.topk(k, dim=0, largest=False)
    count = torch.zeros(X + Y).to(dxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, torch.ones(X + Y).to(dxx) * (float(k) / 2)).float()
    acc = torch.eq(label, pred).float().mean()
    return acc
