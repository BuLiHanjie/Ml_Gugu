from ml_model.cv_model import CvModel


class SequenceCvModel:
    def __init__(self, model_classes, params, train_x, train_y):
        self.model_classes = model_classes
        self.params = params
        self.dtrain = (train_x, train_y)
        self.models = None
        pass

    def train(self, train_params, metric):
        self.models = list()
        init_score = None
        for index, (m_class, param, train_param) in enumerate(
                zip(self.model_classes, self.params, train_params)):
            p = {
                'init_score': init_score,
            }
            for k, v in param.items():
                p[k] = v
            model = CvModel(self.dtrain[0], self.dtrain[1], m_class, **p)
            train_pred = model.train(**train_param)
            init_score = train_pred
            print('step {} train evaluate {}'.format(index, metric(self.dtrain[1], train_pred)))
            self.models.append(model)
        return init_score

    def predict(self, x, names=None, init_score=None):
        for model in self.models:
            init_score = model.predict(x, names=names, init_score=init_score)
        return init_score

    def get_score(self, importance_type='split'):
        '''gain, split'''
        res = dict()
        for model in self.models:
            score = model.get_score(importance_type)
            for k, v in score.items():
                res[k] = res.get(k, 0) + v
        return res


