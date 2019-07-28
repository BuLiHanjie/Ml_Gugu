import numpy as np
from sklearn.model_selection import KFold


class CvModel:
    def __init__(self, x, y, model_class, init_score=None, names=None, K=5, weight=None, seed=None):
        self.x, self.y = x, y
        self.names = names
        self.model_class = model_class
        self.init_score = init_score
        self.weight = weight
        self.K = K
        self.seed = seed
        self.models = list()
        self.train_pred = None

    def train(self, **kwargs):
        res = list()
        indexs = list()
        kf = KFold(n_splits=self.K, shuffle=True, random_state=self.seed)
        for index, (train_index, test_index) in enumerate(kf.split(self.x)):
            train_x, train_y, train_w = self.x[train_index], self.y[train_index], self.weight[train_index]
            test_x, test_y, test_w = self.x[test_index], self.y[test_index], self.weight[test_index]
            train_init_score = None
            test_init_score = None
            if self.init_score is not None:
                train_init_score = self.init_score[train_index]
                test_init_score = self.init_score[test_index]
            model = self.model_class(train_x, train_y, test_x, test_y, names=self.names, weight=train_w,
                                     init_train_score=train_init_score,
                                     init_valid_score=test_init_score
                                     )
            print('model {} training...'.format(index))
            model.train(**kwargs)
            print('model {} finish...'.format(index))
            self.models.append(model)
            res.append(model.predict(test_x, self.names, init_score=test_init_score))
            indexs.append(test_index)
        res = np.concatenate(res)
        indexs = np.concatenate(indexs)
        index = np.zeros(indexs.shape, dtype=np.int32)
        for i in range(len(indexs)):
            index[indexs[i]] = i
        res = res[index]
        self.train_pred = res
        return res

    def predict_train(self):
        res = self.train_pred if self.train_pred is None or self.init_score is None else \
            self.train_pred + self.init_score
        return res

    def predict(self, x, names=None, init_score=None):
        res = list()
        for model in self.models:
            if init_score is None:
                res.append(model.predict(x, names=names))
            else:
                res.append(model.predict(x, names=names, init_score=init_score))
        return np.mean(res, axis=0)

    def get_score(self, importance_type='split'):
        """
        get the feature importance
        :param importance_type: {split, weight, gain}
        :return: dict
        """
        score = [model.get_score(importance_type=importance_type) for model in self.models]
        keys = list(score[0].keys())
        res = dict()
        for k in keys:
            res[k] = np.mean(list(map(lambda x: x[k], score)))
        return res
