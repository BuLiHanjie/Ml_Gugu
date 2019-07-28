from sklearn.linear_model import LogisticRegression
from ml_model.ops import data_copy

class LrModel:
    def __init__(self, train_x, train_y, valid_x, valid_y, names=None, weight=None):
        print('weight :', weight)
        if type(weight) == int:
            print('copy datas')
            train_x, train_y = data_copy(train_x, train_y, weight)
        self.dtrain = (train_x, train_y)
        self.dvalid = (valid_x, valid_y)
        self.feature_names = names
        self.model = None

    def train(self, **kwargs):
        # evallist = [(self.dvalid, 'eval'), (self.dtrain, 'train')]
        self.model = LogisticRegression(**kwargs)
        self.model.fit(self.dtrain[0], self.dtrain[1])
        pred = self.predict(self.dtrain[0])
        from sklearn.metrics import roc_auc_score
        print('train auc:', roc_auc_score(self.dtrain[1], pred))

    def predict(self, x, names=None):
        # data = lgb.Dataset(x, feature_name=names)
        result = self.model.predict_proba(x)[:, 1]
        return result

    def get_score(self):
        return None
