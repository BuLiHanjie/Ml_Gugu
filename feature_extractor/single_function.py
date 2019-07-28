import numpy as np


class SingleFunction:
    class Len:
        def __init__(self):
            self.res = dict()

        def push(self, cond, cols):
            self.res[cols] = len(cond)

        def get(self, cols):
            return self.res.get(cols, -1)

    class OneMap:
        def __init__(self):
            self.res = dict()

        def push(self, cond, cols):
            self.res[cols] = cond

        def get(self, cols):
            return self.res.get(cols, 0)

    class Hash:
        def __init__(self, scale, mod):
            self.res = dict()
            self.scale = scale
            self.mod = mod

        def push(self, cond, cols):
            s = 0
            if isinstance(cond, (list, np.ndarray)):
                for v in cond:
                    s = (s * self.scale + v) % self.mod
            else:
                s = cond * self.scale % self.mod
            self.res[cols] = s

        def get(self, cols):
            return self.res.get(cols, 0)
