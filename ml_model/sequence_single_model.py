class SequenceSingleModel:
    def __init__(self, model_classes, params, train_x, train_y, valid_x, valid_y):
        self.model_classes = model_classes
        self.params = params
        self.dtrain = (train_x, train_y)
        self.dvalid = (valid_x, valid_y) 
        self.models = None
        pass

    def train(self, train_params, metric):
        self.models = list()
        init_train_score = None
        init_valid_score = None
        for index, (m_class, param, train_param) in enumerate(
                zip(self.model_classes, self.params, train_params)):
            p = {
                'init_train_score': init_train_score,
                'init_valid_score': init_valid_score
            }
            for k, v in param.items():
                p[k] = v
            model = m_class(self.dtrain[0], self.dtrain[1], self.dvalid[0], self.dvalid[1], **p)
            model.train(**train_param)
            init_train_score = model.predict(self.dtrain[0])
            init_valid_score = model.predict(self.dvalid[0])
            print('step {} train evaluate {}'.format(index, metric(self.dtrain[1], init_train_score)))
            print('step {} valid evaluate {}'.format(index, metric(self.dvalid[1], init_valid_score)))
            self.models.append(model)

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


