import torch
import torch.nn as nn
import numpy as np
from pytorch_model.components.embedding import DynEmbedding
from pytorch_model.optimizer.sgd import SGD


class SubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.opt = SGD([nn.Parameter(torch.rand(3))], 0.1)
        self.emb = DynEmbedding(3, self.opt, 2)
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.emb(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.opt = SGD([nn.Parameter(torch.rand(3))], 0.1)
        self.emb = DynEmbedding(3, self.opt, 2)
        self.emb1 = SubModel()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.emb1(x)

class Model_Lieanr(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(3, 5)


if __name__ == '__main__':
    # a = Model_Lieanr()
    # print(a._parameters.items())
    # s = a.state_dict()
    # print(s)
    # print(s.get('p.weight'))
    # torch.save({
    #     'model_state_dict': a.state_dict()
    # }, '../log/linear_save.pth')
    # checkpoint = torch.load('../log/linear_save.pth')
    # # print(checkpoint)
    # print('load....')
    # a = Model_Lieanr()
    # a.load_state_dict(checkpoint['model_state_dict'])
    # exit()
    model = Model()
    s = model.forward(np.array(['a', 'b', 'c', 'd']))
    print(model._parameters.items())
    print(s)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.state_dict(),
    }, '../log/save.pth')
    checkpoint = torch.load('../log/save.pth')
    # print(checkpoint)
    print('load....')
    model = Model()
    model.load_state_dict(checkpoint['model_state_dict'])
    print('meta data:', checkpoint['model_state_dict']._metadata)
    model.eval()
    s2 = model.forward(np.array(['a', 'd', 'b', 'c', 'd', 'e']))
    print(s2)

    # print(s - s2)