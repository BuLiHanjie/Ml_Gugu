import numpy as np


class GroupBase:
    def __init__(self):
        self.res = dict()

    def push(self, *args):
        pass

    def get(self, *args):
        pass

    def rescale(self, res):
        return res


class GroupFunction:

    class Count(GroupBase):

        def push(self, *args):
            self.res[args[0]] = self.res.get(args[0], 0) + 1

        def get(self, *args):
            return self.res.get(args[0], 0)

    class Min(GroupBase):

        def push(self, *args):
            self.res[args[0]] = min(self.res.get(args[0], 1e10), args[1])

        def get(self, *args):
            return self.res.get(args[0], -1)

    class Max(GroupBase):

        def push(self, *args):
            self.res[args[0]] = max(self.res.get(args[0], -1e10), args[1])

        def get(self, *args):
            return self.res.get(args[0], -1)

    class Sum(GroupBase):

        def push(self, *args):
            self.res[args[0]] = self.res.get(args[0], 0) + args[1]

        def get(self, *args):
            return self.res.get(args[0], 0)

    class Mean(GroupBase):
        def __init__(self):
            super().__init__()
            self.res_sum = dict()
            self.res_cnt = dict()

        def push(self, *args):
            self.res_sum[args[0]] = self.res_sum.get(args[0], 0) + args[1]
            self.res_cnt[args[0]] = self.res_cnt.get(args[0], 0) + 1

        def get(self, *args):
            if args[0] not in self.res_sum:
                return 0.
            return self.res_sum.get(args[0]) / self.res_cnt.get(args[0])

    class Ratio(GroupBase):
        def __init__(self):
            super().__init__()
            self.res_cnt1 = dict()
            self.res_cnt2 = dict()

        def push(self, *args):
            self.res_cnt1[args[0] + (args[1],)] = self.res_cnt1.get(args[0] + (args[1],), 0) + 1
            self.res_cnt2[args[0]] = self.res_cnt2.get(args[0], 0) + 1

        def get(self, *args):
            if args[0] not in self.res_cnt2:
                return 0.
            return self.res_cnt1.get(args[0] + (args[1],), 0) / self.res_cnt2.get(args[0])

    class Nunique(GroupBase):

        def push(self, *args):
            if args[0] not in self.res:
                self.res[args[0]] = set()
            s = self.res.get(args[0])
            s.add(args[1])

        def get(self, *args):
            if args[0] not in self.res:
                return 0.
            return len(self.res.get(args[0]))

    class Std(GroupBase):
        def __init__(self):
            super().__init__()
            self.res_list = dict()
            self.res = dict()

        def push(self, *args):
            if args[0] not in self.res_list:
                self.res_list[args[0]] = list()
            s = self.res_list.get(args[0])
            s.append(args[1])

        def get(self, *args):
            if args[0] not in self.res:
                if args[0] not in self.res_list:
                    self.res[args[0]] = 0.
                else:
                    self.res[args[0]] = np.std(self.res_list.get(args[0]))
            return self.res.get(args[0])

    class Rank(GroupBase):
        def __init__(self):
            super().__init__()
            self.res_list = dict()
            self.res = dict()

        def push(self, *args):
            if args[0] not in self.res_list:
                self.res_list[args[0]] = list()
            s = self.res_list.get(args[0])
            s.append(args[1])

        def get(self, *args):
            if args[0] not in self.res and (args[0] in self.res_list):
                s = np.unique(self.res_list.get(args[0]))
                ss = dict(zip(s, np.arange(len(s) + 1)))
                self.res[args[0]] = ss
            if args[0] not in self.res:
                return -1
            return self.res.get(args[0]).get(args[1])

    class ArrayFunction:
        class CategoryCount(GroupBase):
            def __init__(self, categories):
                super().__init__()
                self.cats = categories

            def push(self, cols, cond):
                if cols not in self.res:
                    d = dict([(v, 0) for v in self.cats])
                    self.res[cols] = d
                d = self.res[cols]
                if cond in d:
                    d[cond] = d.get(cond) + 1

            def get(self, cols, cond):
                d = self.res.get(cols, None)
                if d is None:
                    return np.zeros(len(self.cats), dtype=np.int32)
                s = np.array([d[v] for v in self.cats])
                return s

        class CategoryRatio(CategoryCount):
            def __init__(self, categories):
                super().__init__(categories)
                self.cols_count = dict()

            def push(self, cols, cond):
                super().push(cols, cond)
                self.cols_count[cols] = self.cols_count.get(cols, 0) + 1

            def get(self, cols, cond):
                s = super().get(cols, cond)
                if cols not in self.cols_count:
                    return s
                s /= self.cols_count.get(cols)
                return s

    class ArrayUnique(GroupBase):
        def __init__(self, na):
            super().__init__()
            self.res = dict()
            self.res_sorted = dict()
            self.na = na
            pass

        def push(self, cols, cond):
            if cols not in self.res:
                self.res[cols] = set()
            d = self.res[cols]
            if isinstance(cond, np.ndarray) or isinstance(cond, list):
                for v in cond:
                    d.add(v)
            else:
                d.add(cond)

        def get(self, cols):
            # print('get', cols)
            if cols not in self.res:
                return self.na
            if cols not in self.res_sorted:
                self.res_sorted[cols] = np.sort(list(self.res[cols])).astype(np.int32)
            return self.res_sorted[cols]

    class CompareCount(GroupBase):
        def __init__(self, compares):
            super().__init__()
            func_map = {
                'eq': lambda a, b: int(a == b),
                'less': lambda a, b: int(a < b),
                'greater': lambda a, b: int(a > b),
                'le': lambda a, b: int(a <= b),
                'ge': lambda a, b: int(a >= b),
            }
            self.funcs = [func_map[v] for v in compares]
            self.contain = dict()

        def push(self, cols, cond):
            if cols not in self.contain:
                self.contain[cols] = list()
            self.contain[cols].append(cond)

        def get(self, cols, cond):
            res = list()
            for f in self.funcs:
                s = 0
                ls = self.contain.get(cols, list())
                for v in ls:
                    s += f(v, cond)
                res.append(s)
            return res

        def rescale(self, res):
            res = np.array(res).T.tolist()
            return res

    class CompareRatio(GroupBase):
        def __init__(self, compares):
            super().__init__()
            func_map = {
                'eq': lambda a, b: int(a == b),
                'less': lambda a, b: int(a < b),
                'greater': lambda a, b: int(a > b),
                'le': lambda a, b: int(a <= b),
                'ge': lambda a, b: int(a >= b),
            }
            self.funcs = [func_map[v] for v in compares]
            self.contain = dict()

        def push(self, cols, cond):
            if cols not in self.contain:
                self.contain[cols] = list()
            self.contain[cols].append(cond)

        def get(self, cols, cond):
            res = list()
            for f in self.funcs:
                s = 0
                ls = self.contain.get(cols, list())
                for v in ls:
                    s += f(v, cond)
                res.append(s / len(ls) if len(ls) > 0 else 0.)
            return res

        def rescale(self, res):
            res = np.array(res).T.tolist()
            return res


