from torch_model.str_to_function import str_top_optimizer, str_to_metric


def evaluate(metrics):
    def wrapper(func):
        def operation(*args, metrics=metrics, **kwargs):
            res = func(*args, **kwargs)
            if res is None:
                return res
            y_pred, y_true = res[0], res[1]
            if not isinstance(metrics, list):
                metrics = [metrics]
            for _m in metrics:
                if isinstance(_m, str):
                    _m = str_to_metric(_m)
                print(_m.__name__, ':', _m(y_true, y_pred))
            return res
        return operation
    return wrapper


def get_optimizer(name, param, kwargs):
    # res = None
    if isinstance(name, list):
        res = list()
        for _n, _p, _k in zip(name, param, kwargs):
            res.append(str_top_optimizer(_n, _p, **_k))
    else:
        res = str_top_optimizer(name, param, **kwargs)
    return res


@evaluate('mse')
def f():
    return [1, 1, 0], [1, 0.9, 0.2]


if __name__ == '__main__':
    a = f()
    print(a)
