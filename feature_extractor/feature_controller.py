import pandas as pd
import numpy as np
import time
import datetime
from multiprocessing import Pool
from multiprocessing import Manager
from feature_extractor.function_manager import FuncType


class FeatureController:

    def __init__(self, df_key, df_log, extractor, custom_functions, K=5, group_name=None, nthreads=1):
        '''
        a controller of feature generation, including ...
        :param df_key: pd.DataFrame for key columns(str type means the local storage of df_key)
        :param df_log: pd.DataFrame for data collection(str type means the local storage of df_key)
        :param extractor: a container with extractor functions
        :param custom_functions: custom_functions
        :param K: the number of flods in cv generation
        :param group_name: sampples in one group are splited into one flod in cv generation
        '''
        self.df_key = df_key
        self.df_log = df_log
        self.extractor = extractor
        self.custom_functions = custom_functions
        # self.fm = fm
        self.K = K
        self.group_name = group_name
        if self.group_name is None:
            self.group_name = '_g'
            self.df_key[self.group_name] = np.arange(len(self.df_key))
        self.fm_functions = list(custom_functions.functions)
        self.fm_func_type = list(custom_functions.func_type)
        self.feature_names = list()
        self.lock_append = Manager().Lock()
        if nthreads is None or nthreads < 1:
            nthreads = 1
        self.nthreads = nthreads
        pass

    # def __getstate__(self):
    #     return self.df_key, self.df_log, self.functions, self.K, self.group_name
    #
    # def __setstate__(self, state):
    #     self.df_key, self.df_log, self.functions, self.K, self.group_name = state

    def append_feature(self, data):
        # self.lock_append.acquire()
        names, features = data
        # print(names, features)
        self.feature_names.extend(names)
        for _n, _f in zip(names, features):
            self.df_key[_n] = _f
        print('features {} add'.format(names))
        # self.lock_append.release()

    def error_log(self, e):
        print('error message...')
        print(e)

    def run_single_core(self):
        for func, _type in zip(self.fm_functions, self.fm_func_type):
            print('func[{}] start at {}'.format(func, datetime.datetime.now()))
            names, features = None, None
            if _type == FuncType.Single:
                names, features = self.run_single(
                    self.custom_functions, func, self.extractor, self.df_key, self.df_log)
            if _type == FuncType.Cv_Single:
                names, features = self.run_cv_single(
                    self.custom_functions, func, self.extractor, self.df_key, self.K, self.group_name)
            if _type == FuncType.Cv_Both:
                names, features = self.run_cv_both(
                    self.custom_functions, func, self.extractor, self.df_key, self.df_log, self.K, self.group_name)
            self.append_feature((names, features))
            # print('func[{}] cost {:.1f} mins'.format(func.__name__, (time.time() - start_time) / 60))
            # print('features :', names)

    def run(self):
        pool = Pool(self.nthreads)
        results = list()
        print('pool initializing...')
        for _func, _func_type in zip(self.fm_functions, self.fm_func_type):
            # print('eval...', _func)
            # _func = eval('self.custom_functions.' + _func)
            # print(_func)
            if _func_type == FuncType.Single:
                results.append(
                    pool.apply_async(
                        self.run_single,
                        args=(self.custom_functions, _func, self.extractor, self.df_key, self.df_log),
                        callback=self.append_feature,
                        error_callback=self.error_log
                    )
                )
            if _func_type == FuncType.Cv_Single:
                results.append(
                    pool.apply_async(
                        self.run_cv_single,
                        args=(self.custom_functions, _func, self.extractor, self.df_key, self.K, self.group_name),
                        callback=self.append_feature,
                        error_callback=self.error_log
                    )
                )
            if _func_type == FuncType.Cv_Both:
                results.append(
                    pool.apply_async(
                        self.run_cv_both,
                        args=(self.custom_functions, _func, self.extractor, self.df_key, self.df_log, self.K, self.group_name),
                        callback=self.append_feature,
                        error_callback=self.error_log
                    )
                )
        print('pool start...')
        pool.close()
        pool.join()
        # for res in results:
        #     print(res.get())

    @staticmethod
    def run_single(custom, func_name, extractor, df_key, df_log):
        start_time = time.time()
        func = custom.get_func(func_name)
        if not isinstance(df_log, pd.DataFrame):
            df_log = pd.read_parquet(df_log)
        if not isinstance(df_key, pd.DataFrame):
            df_key = pd.read_parquet(df_key)
        # print('run_single func:', func)
        func_result = func(extractor, df_key, df_log)
        name, feature = func_result if type(func_result) is tuple else (func_name, func_result)
        if type(name) is not list:
            name = [name]
            feature = [feature]
        print('func[{}] cost {:.1f} mins'.format(func_name, (time.time() - start_time) / 60))
        return name, feature

    @staticmethod
    def run_cv_single(custom, func_name, extractor, df_key, K, group_name):
        start_time = time.time()
        func = custom.get_func(func_name)
        if not isinstance(df_key, pd.DataFrame):
            df_key = pd.read_parquet(df_key)
        history_k = dict()
        group = df_key[group_name].tolist()
        ks = np.random.randint(K, size=len(df_key))
        for i, (_g, _k) in enumerate(zip(group, ks)):
            if _g in history_k:
                ks[i] = history_k[_g]
            else:
                history_k[_g] = _k
        df_key['_k'] = ks
        df_key['_i'] = np.arange(len(df_key))
        # self.df_log['_k'] = np.random.randint(self.K, size=len(self.df_log))
        names, features, indexes = list(), list(), list()
        for k in range(K):
            _df_key = df_key[df_key['_k'] == k]
            _df_log = df_key[df_key['_k'] != k]
            func_result = func(extractor, _df_key, _df_log)
            name, feature = func_result if type(func_result) is tuple else (func_name, func_result)
            if type(name) is not list:
                name = [name]
                feature = [feature]
            if len(names) == 0:
                names = name
                features = feature
                indexes = _df_key['_i'].tolist()
            else:
                for a, b in zip(features, feature):
                    a.extend(b)
                indexes.extend(_df_key['_i'].tolist())
            print('function {} flod {} finish at {} ...'.format(func_name, k, datetime.datetime.now()))
        # print(features)
        features = np.array(features)
        # print(features.shape)
        # print(max(indexes), len(df_key))
        index_reverse = [0] * features.shape[1]
        if len(indexes) > features.shape[1]:
            print(len(indexes), features.shape, features)
        for i, _index in enumerate(indexes):
            index_reverse[_index] = i
        features = features[:, index_reverse]
        print('func[{}] cost {:.1f} mins'.format(func_name, (time.time() - start_time) / 60))
        return names, features

    @staticmethod
    def run_cv_both(custom, func_name, extractor, df_key, df_log, K, group_name):
        start_time = time.time()
        func = custom.get_func(func_name)
        if not isinstance(df_log, pd.DataFrame):
            df_log = pd.read_parquet(df_log)
        if not isinstance(df_key, pd.DataFrame):
            df_key = pd.read_parquet(df_key)
        history_k = dict()
        group = df_key[group_name].tolist()
        ks = np.random.randint(K, size=len(df_key))
        for i, (_g, _k) in enumerate(zip(group, ks)):
            if _g in history_k:
                ks[i] = history_k[_g]
            else:
                history_k[_g] = _k
        df_key['_k'] = ks
        df_key['_i'] = np.arange(len(df_key))
        df_log['_k'] = np.random.randint(K, size=len(df_log))
        names, features, indexes = list(), list(), list()
        for k in range(K):
            _df_key = df_key[df_key['_k'] == k]
            _df_log = df_log[df_log['_k'] != k]
            func_result = func(extractor, _df_key, _df_log)
            name, feature = func_result if type(func_result) is tuple else (func_name, func_result)
            if type(name) is not list:
                name = [name]
                feature = [feature]
            if len(names) == 0:
                names = name
                features = feature
                indexes = _df_key['_i'].tolist()
            else:
                for a, b in zip(features, feature):
                    a.extend(b)
                indexes.extend(_df_key['_i'].tolist())
            print('function {} flod {} finish at {} ...'.format(func_name, k, datetime.datetime.now()))
        # print(features)
        features = np.array(features)
        index_reverse = [0] * features.shape[1]
        if len(indexes) > features.shape[1]:
            print(len(indexes), features.shape, features)
        for i, _index in enumerate(indexes):
            index_reverse[_index] = i
        features = features[:, index_reverse]
        print('func[{}] cost {:.1f} mins'.format(func_name, (time.time() - start_time) / 60))
        return names, features

    def get_key_columns(self):
        return list(self.df_key.columns)

    def to_csv(self, path, columns=None, float_format='%.4f'):
        if columns is None:
            columns = self.df_key.columns
        self.df_key[columns].to_csv(path, index=None, float_format=float_format, date_format='%Y%m%d%H%M%S')

    def to_feather(self, path, columns=None):
        if columns is None:
            columns = self.df_key.columns
        self.df_key[columns].to_feather(path)

    def to_parquet(self, path, columns=None):
        if columns is None:
            columns = self.df_key.columns
        self.df_key[columns].to_parquet(path)
