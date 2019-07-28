import lightgbm as lgb
from ml_model.ops import data_copy


class LgbModel:

    def __init__(self, train_x, train_y, valid_x, valid_y, names=None, weight=None, init_train_score=None,
                 init_valid_score=None):
        if type(weight) == int:
            train_x, train_y = data_copy(train_x, train_y, weight)
            weight = None
        self.dtrain = lgb.Dataset(train_x, train_y, feature_name=names, weight=weight, init_score=init_train_score)
        self.dvalid = lgb.Dataset(valid_x, valid_y, feature_name=names, init_score=init_valid_score)
        self.feature_names = names
        self.model = None

    def train(self, **kwargs):
        # evallist = [(self.dvalid, 'eval'), (self.dtrain, 'train')]
        feval = None
        valid_sets = [self.dtrain]
        if self.dvalid is not None:
            valid_sets.append(self.dvalid)
        if not isinstance(kwargs['metric'], str):
            feval = kwargs['metric']
            kwargs['metric'] = 'None'
        fobj = None
        if 'fobj' in kwargs:
            fobj = kwargs['fobj']
            del kwargs['fobj']
        self.model = lgb.train(kwargs, self.dtrain, kwargs['num_round'], valid_sets=valid_sets,
                               feval=feval, fobj=fobj)

    def predict(self, x, names=None, init_score=None):
        # data = lgb.Dataset(x, feature_name=names)
        result = self.model.predict(x)
        if init_score is not None:
            result += init_score
        return result

    def get_score(self, importance_type='split'):
        '''gain, split'''
        importance = self.model.feature_importance(importance_type=importance_type)
        importance_map = dict(zip(self.feature_names, importance))
        return importance_map
