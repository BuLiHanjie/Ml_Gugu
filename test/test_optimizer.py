import torch
import torch.nn as nn
from pytorch_model.optimizer.sgd import SGD


if __name__ == '__main__':
    a = nn.Parameter(torch.ones(2, 3).cuda()).reshape(-1)
    print(a)
    for v in a:
        print(int(v))
    exit()
    target = torch.zeros(2).cuda()
    a = nn.Parameter(torch.ones(2).cuda())
    print(a, a[0], a[[0, 1]])
    exit()
    op = SGD([a], lr=0.1)
    b = nn.Parameter(torch.randn(2).cuda())
    op.add_param(b)
    print('init :', b)
    func = nn.MSELoss()
    loss = func(target, a + b)
    print('loss:', loss)
    loss.backward()
    op.step()
    print('a:', a)
    print('end :', b)