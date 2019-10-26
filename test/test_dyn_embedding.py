import torch
import torch.nn as nn
import numpy as np
from pytorch_model.components.embedding import DynEmbedding
from pytorch_model.optimizer.sgd import SGD


if __name__ == '__main__':
    opt = SGD([nn.Parameter(torch.rand(3))], 0.1)
    emb = DynEmbedding(16, opt, 2)
    for i in range(10):
        s = emb(np.array(['a', 'b', 'c']))
        target = torch.zeros(3, 16)
        # print(s, s.shape)

        opt.zero_grad()
        loss = nn.MSELoss()(target, s)
        print(loss)
        loss.backward()
        opt.step()
    emb.eval()
    s = emb(np.array(['a', 'b', 'c', 'd']))
    print(s)
