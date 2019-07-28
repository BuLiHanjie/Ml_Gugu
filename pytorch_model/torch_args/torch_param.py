from abc import abstractmethod
from torch_model.str_to_function import str_to_activation


class ParamBase:

    def __init__(self, in_shape):
        self._in_shape = in_shape
        if in_shape is not None and isinstance(in_shape, int):
            self._in_shape = (in_shape,)
        self.father = None
        self.param_map = dict()

    def get_in_shape(self):
        return self._in_shape if self._in_shape is not None else self.father.get_out_shape()

    @property
    def in_shape(self):
        return self._in_shape

    @abstractmethod
    def get_out_shape(self):
        pass


class LinearParam:

    def __init__(self, in_features, out_features, activation, bias=True):
        self._in_features = in_features
        self._out_features = out_features
        self._activation = activation
        self._bias = bias
        self._activations = str_to_activation(self._activations) if isinstance(self._activations, str) \
            else self._activations

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features

    @property
    def activation(self):
        return self._activation

    @property
    def bias(self):
        return self._bias


class MlpParam:

    def __init__(self, in_features, hidden_layers, activations, bias=True):
        self._in_features = in_features
        self._hidden_layers = hidden_layers
        self._activations = [activations] * len(hidden_layers) if not isinstance(activations, list) else activations
        self._bias = [bias] * len(hidden_layers) if not isinstance(bias, list) else bias
        self._activations = [str_to_activation(v) if isinstance(v, str) else v for v in self._activations]

    @property
    def in_features(self):
        return self._in_features

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @property
    def activations(self):
        return self._activations

    @property
    def bias(self):
        return self._bias


class EmbeddingParam:

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = _weight


class DeepFmParam:

    def __init__(self, field_size, embedding_dim, deep_dim,
                 mlp_hidden_layers, mlp_activations,
                 linear_out_features, linear_activaions):
        self.field_size = field_size
        self.emedding_dim = embedding_dim
        self.deep_dim = deep_dim
        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_activations = mlp_activations
        self.linear_out_features = linear_out_features
        self.linear_activaions = linear_activaions
