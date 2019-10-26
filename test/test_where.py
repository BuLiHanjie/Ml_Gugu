import torch
import os
import numpy as np
from pytorch_model.components.embedding import ContinuousEmbedding

if __name__ == '__main__':
    # a = torch.randn(5)
    # q = [0.2, 0.5, 1]
    # q1 = torch.tensor([-np.inf] + q)
    # q2 = torch.tensor(q + [np.inf])
    # a = a.unsqueeze(-1).repeat(1, q1.shape[0])
    # q1 = q1.unsqueeze(0).repeat(a.shape[0], 1)
    # q2 = q2.unsqueeze(0).repeat(a.shape[0], 1)
    # print(a, a.shape)
    # print(q1)
    # print(q2)
    # s = (a > q1) * (a <= q2)
    # print(s)
    # res = s.argmax(-1)
    # print(res)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')
    model = ContinuousEmbedding(12, [0, 0.2, 0.4, 0.6, 0.8, 1]).to(device)
    x = torch.randn(3, 4).to(device)
    print(x)
    print(model(x))