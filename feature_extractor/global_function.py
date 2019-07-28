import numpy as np


class GlobalFunction:
    class Rank:
        def __init__(self):
            self.res = dict()
            pass

        def run(self, values):
            vs = np.unique(values)
            for i, v in enumerate(vs):
                self.res[v] = i

        def get(self, values):
            res = [self.res.get(v, -1) for v in values]
            return res

    class ArrayCount:

        def __init__(self):
            self.count = dict()

        def run(self, values):
            for cond in values:
                if isinstance(cond, (list, np.ndarray)):
                    for v in cond:
                        self.count[v] = self.count.get(v, 0) + 1
                else:
                    self.count[cond] = self.count.get(cond, 0) + 1

        def get(self, values):
            res = list()
            for cond in values:
                s = 0
                if isinstance(cond, (list, np.ndarray)):
                    for v in cond:
                        s += self.count.get(v, 0)
                else:
                     s = self.count.get(cond, 0)
                res.append(s)
            return res
