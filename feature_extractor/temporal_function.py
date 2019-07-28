import pandas as pd
import numpy as np
from feature_extractor.tree import Treap


class TemporalBase:
    def __init__(self):
        self.time_gap = None
        self.keep_now = None

    def set_time_gap(self, v):
        self.time_gap = v

    def set_keep_now(self, v):
        self.keep_now = v

    def init_params(self, *args):
        pass

    def push(self, *args):
        pass

    def pop(self, *args):
        pass

    def rescale(self, res):
        return res


class TemporalFunction:

    class Sum(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.s = 0

        def push(self, *args):
            self.s += args[0]

        def pop(self, *args):
            self.s -= args[0]

        def attach(self, *args):
            self.res[args[1] + (args[2],)] = self.s

        def get(self, *args):
            return self.res.get(args[1] + (args[2],))

    class Count(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.s = 0

        def push(self, *args):
            self.s += 1

        def pop(self, *args):
            self.s -= 1

        def attach(self, *args):
            self.res[args[1] + (args[2], )] = self.s

        def get(self, *args):
            return self.res.get(args[1] + (args[2], ))

    class Mean(TemporalBase):

        def __init__(self):
            self.res = dict()
            self._sum = 0
            self._cnt = 0

        def push(self, *args):
            self._sum += args[0]
            self._cnt += 1

        def pop(self, *args):
            self._sum -= args[0]
            self._cnt -= 1

        def attach(self, *args):
            self.res[args[1] + (args[2],)] = self._sum / self._cnt if self._cnt > 0 else 0.

        def get(self, *args):
            return self.res[args[1] + (args[2],)]

    class Std(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.queue = list()
            self.start = 0

        def push(self, cond, cols, date):
            self.queue.append(cond)

        def pop(self, cond, cols, date):
            self.start += 1

        def attach(self, cond, cols, date):
            self.res[cols + (date,)] = np.std(self.queue[self.start:]) if self.start < len(self.queue) else 0.

        def get(self, *args):
            return self.res[args[1] + (args[2],)]

    class Max(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.queue = list()
            self.start = 0

        def push(self, *args):
            self.queue.append(args[0])

        def pop(self, *args):
            self.start += 1

        def attach(self, *args):
            self.res[args[1] + (args[2],)] = max(self.queue[self.start:]) if self.start < len(self.queue) else 0.

        def get(self, *args):
            return self.res[args[1] + (args[2],)]

    class Min(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.queue = list()
            self.start = 0

        def push(self, *args):
            self.queue.append(args[0])

        def pop(self, *args):
            self.start += 1

        def attach(self, *args):
            self.res[args[1] + (args[2],)] = min(self.queue[self.start:]) if self.start < len(self.queue) else 0.

        def get(self, *args):
            return self.res[args[1] + (args[2],)]

    class Median(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.queue = list()
            self.start = 0

        def push(self, *args):
            self.queue.append(args[0])

        def pop(self, *args):
            self.start += 1

        def attach(self, *args):
            self.res[args[1] + (args[2],)] = np.median(self.queue[self.start:]) if self.start < len(self.queue) else 0.

        def get(self, *args):
            return self.res[args[1] + (args[2],)]

    class Diff(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.queue = list()
            self.start = 0

        def push(self, *args):
            self.queue.append(args[0])

        def pop(self, *args):
            self.start += 1

        def attach(self, *args):
            self.res[args[1] + (args[2],)] = self.queue[-1] - self.queue[self.start] if self.start < len(self.queue) else 0.

        def get(self, *args):
            return self.res[args[1] + (args[2],)]

    class EqualRatio(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.count_dict = dict()
            self.cnt = 0

        def push(self, *args):
            self.count_dict[args[0]] = self.count_dict.get(args[0], 0) + 1
            self.cnt += 1

        def pop(self, *args):
            self.count_dict[args[0]] = self.count_dict[args[0]] - 1
            self.cnt -= 1

        def attach(self, *args):
            self.res[(args[0],) + args[1] + (args[2],)] =\
                self.count_dict[args[0]] / self.cnt if args[0] in self.count_dict and self.cnt > 0 else 0.

        def get(self, *args):
            return self.res.get((args[0],) + args[1] + (args[2],), 0)

    class EqualCount(TemporalBase):

        def __init__(self):
            self.res = dict()
            self.count_dict = dict()

        def push(self, *args):
            self.count_dict[args[0]] = self.count_dict.get(args[0], 0) + 1

        def pop(self, *args):
            self.count_dict[args[0]] = self.count_dict[args[0]] - 1

        def attach(self, *args):
            self.res[(args[0],) + args[1] + (args[2],)] = self.count_dict.get(args[0], 0)

        def get(self, *args):
            return self.res.get((args[0],) + args[1] + (args[2],), 0)

    class GreaterRatio(TemporalBase):
        def __init__(self):
            self.res = dict()
            self.count_dict = dict()
            self.cnt = 0

        def push(self, *args):
            self.count_dict[args[0]] = self.count_dict.get(args[0], 0) + 1
            self.cnt += 1

        def pop(self, *args):
            self.count_dict[args[0]] = self.count_dict[args[0]] - 1
            self.cnt -= 1

        def attach(self, *args):
            self.res[(args[0],) + args[1] + (args[2],)] = \
                sum([v for k, v in self.count_dict.items() if k > args[0]]) / self.cnt if self.cnt > 0 else 0.

        def get(self, *args):
            return self.res.get((args[0],) + args[1] + (args[2],), 0)

    class GreaterCount(TemporalBase):
        def __init__(self):
            self.res = dict()
            self.count_dict = dict()

        def push(self, *args):
            self.count_dict[args[0]] = self.count_dict.get(args[0], 0) + 1

        def pop(self, *args):
            self.count_dict[args[0]] = self.count_dict[args[0]] - 1

        def attach(self, *args):
            self.res[(args[0],) + args[1] + (args[2],)] = sum([v for k, v in self.count_dict.items() if k > args[0]])

        def get(self, *args):
            return self.res.get((args[0],) + args[1] + (args[2],), 0)

    class LessRatio(TemporalBase):
        def __init__(self):
            self.res = dict()
            self.count_dict = dict()
            self.cnt = 0

        def push(self, *args):
            self.count_dict[args[0]] = self.count_dict.get(args[0], 0) + 1
            self.cnt += 1

        def pop(self, *args):
            self.count_dict[args[0]] = self.count_dict[args[0]] - 1
            self.cnt -= 1

        def attach(self, *args):
            self.res[(args[0],) + args[1] + (args[2],)] = \
                sum([v for k, v in self.count_dict.items() if k < args[0]]) / self.cnt if self.cnt > 0 else 0.

        def get(self, *args):
            return self.res.get((args[0],) + args[1] + (args[2],), 0)

    class LessCount(TemporalBase):
        def __init__(self):
            self.res =dict()
            self.count_dict = dict()

        def push(self, *args):
            self.count_dict[args[0]] = self.count_dict.get(args[0], 0) + 1

        def pop(self, *args):
            self.count_dict[args[0]] = self.count_dict[args[0]] - 1

        def attach(self, *args):
            self.res[(args[0],) + args[1] + (args[2],)] = sum([v for k, v in self.count_dict.items() if k < args[0]])

        def get(self, *args):
            return self.res.get((args[0],) + args[1] + (args[2],), 0)

    class LessEqCount_Tree(TemporalBase):
        def __init__(self):
            self.res =dict()
            self.tree = Treap()

        def push(self, cond, cols, date):
            self.tree.add(cond)

        def pop(self, cond, cols, date):
            self.tree.delete(cond)

        def attach(self, cond, cols, date):
            self.res[(cond,) + cols + (date,)] = self.tree.less_equal_count(cond)
            # print('attach:', (cond,) + cols + (date,), self.res[(cond,) + cols + (date,)])

        def get(self, cond, cols, date):
            # print('get:', (cond,) + cols + (date,), self.res.get((cond,) + cols + (date,), 0))
            return self.res.get((cond,) + cols + (date,), 0)

    class LessEqRatio_Tree(TemporalBase):
        def __init__(self):
            self.res =dict()
            self.tree = Treap()
            self.count = 0

        def push(self, cond, cols, date):
            self.tree.add(cond)
            self.count += 1

        def pop(self, cond, cols, date):
            self.tree.delete(cond)
            self.count -= 1

        def attach(self, cond, cols, date):
            self.res[(cond,) + cols + (date,)] = self.tree.less_equal_count(cond) / self.count if self.count > 0 else 0.

        def get(self, cond, cols, date):
            return self.res.get((cond,) + cols + (date,), 0)

    class DiffMean(TemporalBase):
        def __init__(self):
            self.res = dict()
            self.count = None
            self.sum = None
            self.gap = None
            self.t = None

        def init_params(self, dl):
            self.count = [0.] * dl
            self.sum = [0.] * dl
            self.gap = self.time_gap - 1
            self.t = 0 if self.keep_now else -1

        def push(self, *args):
            self.sum[args[2]] += args[0]
            self.count[args[2]] += 1

        def pop(self, *args):
            self.sum[args[2]] -= args[0]
            self.count[args[2]] -= 1

        def attach(self, *args):
            r = args[2] + self.t
            l = args[2] - self.gap + self.t
            r = max(r, 0)
            l = max(l, 0)
            sr = self.sum[r] / self.count[r] if self.count[r] > 0 else 0
            sl = self.sum[l] / self.count[l] if self.count[l] > 0 else 0
            self.res[args[1] + (args[2],)] = sr - sl

        def get(self, *args):
            return self.res[args[1] + (args[2],)]

    # class DiffLatest(TemporalBase):
    #     def __init__(self):
    #         self.res = dict()
    #         self.last_cond = None
    #         self.last_cols = None
    #         self.last_date = None
    #
    #     def init_params(self):
    #         self.t = 0 if self.keep_now else 1
    #
    #     def push(self, cond, cols, date):
    #         # print('push', cols, cond, date)
    #         self.last_cond = cond
    #         self.last_cols = cols
    #         self.last_date = date
    #
    #     def pop(self, cond, cols, date):
    #         pass
    #
    #     def attach(self, cond, cols, date):
    #         # print('attach', cols, cond, self.last_cols, self.last_cond, date, self.last_date)
    #         # if self.last_date is not None:
    #         #     print(date - self.last_date, self.time_gap + self.t)
    #         if np.isnan(cond):
    #             cond = -1
    #         if cols == self.last_cols and date - self.last_date < self.time_gap + self.t and cond != -1:
    #             self.res[cols + (date, cond)] = cond - self.last_cond
    #         else:
    #             self.res[cols + (date, cond)] = 0.
    #
    #     def get(self, cond, cols, date):
    #         if np.isnan(cond):
    #             cond = -1
    #         return self.res[cols + (date, cond)]

    class DiffLatest(TemporalBase):
        def __init__(self):
            self.res = dict()
            self.last_cond = None
            self.last_cols = None
            self.last_date = None

        def init_params(self):
            self.t = 0 if self.keep_now else 1

        def push(self, cond, cols, date):
            # print('push', cols, cond, date)
            self.last_cond = cond
            self.last_cols = cols
            self.last_date = date

        def pop(self, cond, cols, date):
            pass

        def attach(self, cond, cols, date):
            if pd.isnull(cond):
                cond = -1
            if cols == self.last_cols and date - self.last_date < self.time_gap + self.t and cond != -1:
                self.res[cols + (date, cond)] = cond - self.last_cond
            else:
                self.res[cols + (date, cond)] = -1

        def get(self, cond, cols, date):
            if pd.isnull(cond):
                cond = -1
            return self.res[cols + (date, cond)]

    class DiffRatioLatest(TemporalBase):
        def __init__(self):
            self.res = dict()
            self.last_cond = None
            self.last_cols = None
            self.last_date = None

        def init_params(self):
            self.t = 0 if self.keep_now else 1

        def push(self, cond, cols, date):
            # print('push', cols, cond, date)
            self.last_cond = cond
            self.last_cols = cols
            self.last_date = date

        def pop(self, cond, cols, date):
            pass

        def attach(self, cond, cols, date):
            if pd.isnull(cond):
                cond = -1
            if cols == self.last_cols and date - self.last_date < self.time_gap + self.t and cond != -1:
                self.res[cols + (date, cond)] = cond / self.last_cond if self.last_cond > 0 else 0.
            else:
                self.res[cols + (date, cond)] = 0.

        def get(self, cond, cols, date):
            if pd.isnull(cond):
                cond = -1
            return self.res[cols + (date, cond)]

    class GapMean(TemporalBase):
        def __init__(self):
            self.res = dict()
            self.last_cols = None
            self.last_date = None
            self.cnt = 0

        def push(self, cond, cols, date):
            if cols != self.last_cols:
                self.last_cols = cols
                self.last_date = date
                self.cnt = 1.
            else:
                self.cnt += 1

        def pop(self, cond, cols, date):
            pass

        def attach(self, cond, cols, date):
            self.res[cols + (date,)] = (date - self.last_date) / self.cnt if cols == self.last_cols else 0.

        def get(self, cond, cols, date):
            return self.res.get(cols + (date,), 0)

    class NeqCount(TemporalBase):

        def __init__(self):
            self.res = dict()
            self._cnt = 0
            self.count = dict()

        def push(self, cond, cols, date):
            self.count[cond] = self.count.get(cond, 0) + 1
            self._cnt += 1

        def pop(self, cond, cols, date):
            self.count[cond] = self.count[cond] - 1
            self._cnt -= 1

        def attach(self, cond, cols, date):
            self.res[cols + (cond, date)] = self._cnt - self.count.get(cond, 0)

        def get(self, cond, cols, date):
            return self.res[cols + (cond, date)]

    class NeqMean(TemporalBase):

        def __init__(self):
            self.res = dict()
            self._cnt = 0
            self._unique = 0
            self.count = dict()

        def push(self, cond, cols, date):
            self.count[cond] = self.count.get(cond, 0) + 1
            if self.count[cond] == 1:
                self._unique += 1
            self._cnt += 1

        def pop(self, cond, cols, date):
            self.count[cond] = self.count[cond] - 1
            if self.count[cond] == 0:
                self._unique -= 1
            self._cnt -= 1

        def attach(self, cond, cols, date):
            a = self._cnt - self.count.get(cond, 0)
            b = self._unique - (1 if self.count.get(cond, 0) > 0 else 0)
            self.res[cols + (cond, date)] = a / b if a > 0 else 0

        def get(self, cond, cols, date):
            return self.res[cols + (cond, date)]

    class UniqueNum(TemporalBase):

        def __init__(self):
            self.res = dict()
            self._unique = 0
            self.count = dict()

        def push(self, cond, cols, date):
            self.count[cond] = self.count.get(cond, 0) + 1
            if self.count[cond] == 1:
                self._unique += 1

        def pop(self, cond, cols, date):
            self.count[cond] = self.count[cond] - 1
            if self.count[cond] == 0:
                self._unique -= 1

        def attach(self, cond, cols, date):
            self.res[cols + (date, )] = self._unique

        def get(self, cond, cols, date):
            return self.res.get(cols + (date, ), 0)

    class ArrayFunction:
        class CategoriesCount(TemporalBase):
            def __init__(self):
                super().__init__()

                self.res = dict()
                self.count = dict()

            def init_params(self, categories):
                self.cats = categories
                for c in self.cats:
                    self.count[c] = 0

            def push(self, cond, cols, date):
                if isinstance(cond, (int, float)):
                    cond = [cond]
                for v in cond:
                    if v in self.count:
                        self.count[v] += 1

            def pop(self, cond, cols, date):
                if isinstance(cond, (int, float)):
                    cond = [cond]
                for v in cond:
                    if v in self.count:
                        self.count[v] -= 1

            def attach(self, cond, cols, date):
                s = np.array([self.count[c] for c in self.cats])
                self.res[cols + (date, )] = s

            def get(self, cond, cols, date):
                return self.res.get(cols + (date, ), np.zeros(len(self.cats)))

            def rescale(self, res):
                res = np.array(res).T.tolist()
                return res

        class CategoriesRatio(CategoriesCount):
            def __init__(self):
                super().__init__()
                self.n = 0

            def push(self, cond, cols, date):
                super().push(cond, cols, date)
                self.n += 1

            def pop(self, cond, cols, date):
                super().pop(cond, cols, date)
                self.n -= 1

            def attach(self, cond, cols, date):
                s = np.array([self.count[c] for c in self.cats])
                if self.n != 0:
                    s = s / self.n
                self.res[cols + (date,)] = s

            def get(self, cond, cols, date):
                return self.res.get(cols + (date, ), np.zeros(len(self.cats)))

            def rescale(self, res):
                return super().rescale(res)

        class CategoriesOnehot(CategoriesCount):
            def __init__(self):
                super().__init__()

            def attach(self, cond, cols, date):
                s = np.array([int(self.count[c] > 0) for c in self.cats])
                self.res[cols + (date, )] = s

            def get(self, cond, cols, date):
                return self.res.get(cols + (date, ), np.zeros(len(self.cats)))

    class Latest:
        class ArrayLen(TemporalBase):
            def __init__(self):
                super().__init__()
                self.res = dict()
                self.last_cols = None
                self.last_cond = None
                self.last_date = None

            def push(self, cond, cols, date):
                self.last_cols = cols
                self.last_cond = cond
                self.last_date = date

            def attach(self, cond, cols, date):
                self.res[cols + (date, )] = len(self.last_cond) \
                    if self.last_cols is not None and self.last_cols == cols else 0

            def get(self, cond, cols, date):
                return self.res.get(cols + (date, ), 0)

        class DateDiff(ArrayLen):
            def attach(self, cond, cols, date):
                self.res[cols + (date, )] = date - self.last_date \
                    if self.last_cols is not None and self.last_cols == cols else 0

            def get(self, cond, cols, date):
                return self.res.get(cols + (date, ), 0)
