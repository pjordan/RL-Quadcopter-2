
"""Layers that act as normalizers.
"""

import numpy as np
from keras.engine.base_layer import Layer

class MinMaxNormalization(Layer):
    """Min-Max normalization layer.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        min_value: array of Floats, the minimum output value.
        max_value: array of Floats, the maximum output value.
    """

    def __init__(self, max_value, min_value, **kwargs):
        super(MinMaxNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.min_value = np.array(min_value)
        self.max_value = np.array(max_value)

    def call(self, inputs):
        return (inputs - self.min_value) / (self.max_value - self.min_value)

    def get_config(self):
        config = {'max_value': self.max_value, 'min_value': self.min_value}
        base_config = super(MinMaxNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class MinMaxDenormalization(Layer):
    """Min-Max denormalization layer.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        min_value: array of Floats, the minimum output value.
        max_value: array of Floats, the maximum output value.
    """

    def __init__(self, min_value, max_value, **kwargs):
        super(MinMaxDenormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.min_value = np.array(min_value)
        self.max_value = np.array(max_value)

    def call(self, inputs):
        return inputs * self.max_value - (inputs + 1.0) * self.min_value

    def get_config(self):
        config = {'max_value': self.max_value, 'min_value': self.min_value}
        base_config = super(MinMaxDenormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape