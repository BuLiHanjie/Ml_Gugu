import xgboost as xgb
from ml_model.ops import data_copy


class XgbModel:
    def __init__(self, train_x, train_y, valid_x, valid_y, names=None, weight=None, init_train_score=None,
                 init_valid_score=None):
        if type(weight) == int:
            train_x, train_y = data_copy(train_x, train_y, weight)
            weight = None
        self.dtrain = xgb.DMatrix(train_x, train_y, feature_names=names, weight=weight)
        self.dvalid = xgb.DMatrix(valid_x, valid_y, feature_names=names)
        self.model = None
        self.names = names

    def train(self, **kwargs):
        evallist = [(self.dvalid, 'eval'), (self.dtrain, 'train')]
        fobj = None
        if 'fobj' in kwargs:
            fobj = kwargs['fobj']
            del kwargs['fobj']
        self.model = xgb.train(kwargs, self.dtrain, kwargs['num_round'], evallist, obj=fobj,
                               feval=kwargs.get('feval', None),
                               maximize=False,
                               early_stopping_rounds=50)

    def predict(self, x, names=None, init_score=None):
        data = xgb.DMatrix(x, feature_names=names)
        result = self.model.predict(data)
        return result

    def get_score(self, importance_type='split'):
        '''weight, gain, cover'''
        importance = self.model.get_score(importance_type=importance_type)
        for _n in self.names:
            if _n not in importance:
                importance[_n] = 0.
        return importance
