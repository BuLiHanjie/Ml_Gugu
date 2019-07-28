from enum import IntEnum

class FuncType(IntEnum):
    Single = 0
    Cv_Single = 1
    Cv_Both = 2


class FunctionManager:
    functions = list()
    func_type = list()

    def __init__(self):
        pass

    @staticmethod
    def register(func):
        FunctionManager.functions.append(func.__name__)
        FunctionManager.func_type.append(FuncType.Single)

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def register_cv( func):
        FunctionManager.functions.append(func.__name__)
        FunctionManager.func_type.append(FuncType.Cv_Single)

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def register_cv_both(func):
        FunctionManager.functions.append(func.__name__)
        FunctionManager.func_type.append(FuncType.Cv_Both)

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def get_func(self, name):
        return eval('self.' + name)
