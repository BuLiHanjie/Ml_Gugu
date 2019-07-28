import numpy as np
from feature_extractor.single_function import SingleFunction
from feature_extractor.group_function import GroupFunction
from feature_extractor.temporal_function import TemporalFunction
from feature_extractor.global_function import GlobalFunction


class FeatureExtractor:

    @staticmethod
    def single_apply(df_key, df_log, cols, cond_col, apply_func, *args):
        log_cols = list(map(tuple, df_log[cols].values.tolist()))
        log_cond = list(df_log[cond_col]) if cond_col is not None else [None] * len(df_log)
        queue = eval('SingleFunction.{}'.format(apply_func))(*args) if isinstance(apply_func, str) else apply_func(*args)
        for _cols, _cond in zip(log_cols, log_cond):
            queue.push(_cond, _cols)
        res = [queue.get(_c) for _c in list(map(tuple, df_key[cols].values.tolist()))]
        return res

    @staticmethod
    def group_apply(df_key, df_log, cols, cond_col, apply_func, *args):
        log_cols = list(map(tuple, df_log[cols].values.tolist()))
        log_cond = list(df_log[cond_col]) if cond_col is not None else [None] * len(df_log)
        queue = eval('GroupFunction.{}'.format(apply_func))(*args) if isinstance(apply_func, str) else apply_func(*args)
        for _cols, _cond in zip(log_cols, log_cond):
            queue.push(_cols, _cond)
        if cond_col is None or cond_col not in df_key.columns:
            res = [queue.get(v) for v in list(map(tuple, df_key[cols].values.tolist()))]
        else:
            res = [queue.get(v1, v2) for v1, v2 in zip(list(map(tuple, df_key[cols].values.tolist())), df_key[cond_col])]
        # print('res:', res)
        res = queue.rescale(res)
        # print('res:', res)
        return res

    @staticmethod
    def temporal_apply(df_key, df_log, date_col, date_gap, cols, cond_col, apply_func, *args, sorted=False, equal=True):
        df_key_sort = df_key
        if not sorted:
            df_log = df_log.sort_values(cols + [date_col])
            df_key_sort = df_key.sort_values(cols + [date_col])
        log_cols = list(map(tuple, df_log[cols].values.tolist()))
        log_cond = df_log[cond_col].values.tolist() if cond_col is not None else [None] * len(df_log)
        log_date = list(df_log[date_col])
        key_cols = list(map(tuple, df_key_sort[cols].values.tolist()))
        key_date = list(df_key_sort[date_col])
        key_cond = df_key_sort[cond_col].values.tolist() if cond_col is not None and cond_col in df_key_sort.columns \
            else [None] * len(df_key_sort)
        queue = eval('TemporalFunction.{}'.format(apply_func))() if isinstance(apply_func, str) else apply_func()
        queue.set_time_gap(date_gap)
        queue.set_keep_now(equal)
        queue.init_params(*args)
        l, r, n = 0, 0, len(log_cols)
        func = lambda a, b: a <= b if equal else a < b
        for _c, _d, _cond in zip(key_cols, key_date, key_cond):
            while r < n and (log_cols[r] < _c or (log_cols[r] == _c and func(log_date[r], _d))):
                queue.push(log_cond[r], log_cols[r], log_date[r])
                r += 1
            while l < r and (log_cols[l] < _c or (log_cols[l] == _c and func(log_date[l], _d - date_gap))):
                queue.pop(log_cond[l], log_cols[l], log_date[l])
                l += 1
            queue.attach(_cond, _c, _d)
        res = [queue.get(_cond, _c, _d) for _c, _d, _cond in zip(
            list(map(tuple, df_key[cols].values.tolist())),
            df_key[date_col],
            df_key[cond_col] if cond_col is not None and cond_col in df_key.columns else [None] * len(df_key)
        )]
        res = queue.rescale(res)
        return res

    @staticmethod
    def global_apply(df_key, df_log, cond_col, apply_func):
        queue = eval('GlobalFunction.{}'.format(apply_func))() if isinstance(apply_func, str) else apply_func()
        queue.run(list(df_log[cond_col]))
        res = queue.get(list(df_key[cond_col]))
        return res

    @staticmethod
    def equal(df_key, cond, value):
        return list((df_key[cond] == value).astype(int))

    @staticmethod
    def to_category(df_key, cond):
        return df_key[cond].asdtype('category')
