"""
Basic feature encoders
"""
import torch
from . import common
from . import initializer


class BaseEncoder(torch.nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def in_dimension(self):
        raise NotImplementedError

    def out_dimension(self):
        raise NotImplementedError


class DenseEncoder(BaseEncoder):
    def __init__(self, in_dimension, out_dimension, activation=torch.nn.Tanh):
        super(DenseEncoder, self).__init__()
        self._in_dimension = in_dimension
        self._out_dimension = out_dimension
        self._fully_connect = torch.nn.Sequential(
            torch.nn.Linear(self._in_dimension, self._out_dimension),
            activation()
        )

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self._fully_connect(x)

    def out_dimension(self):
        return self._out_dimension

    def in_dimension(self):
        return self._in_dimension


class IDEncoder(BaseEncoder):
    def __init__(self, vocab_size, out_dimension, add_plugin=False):
        super(IDEncoder, self).__init__()
        self._embedding_matrix = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=out_dimension
        )
        initializer.default_weight_init(self._embedding_matrix.weight)
        self.add_plugin = add_plugin
        if self.add_plugin:
            self._plugin_downsampling = common.Linear(out_dimension, out_dimension // 2, bias=False)
            self._plugin_upsampling = common.Linear(out_dimension // 2, out_dimension, bias=False)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self._plugin_upsampling(self._plugin_downsampling(self._embedding_matrix(x)))

    def out_dimension(self):
        return self._embedding_matrix.embedding_dim
