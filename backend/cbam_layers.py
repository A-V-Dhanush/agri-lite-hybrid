"""
CBAM Attention Layers for AgriLite-Hybrid Model Loading.

These layer definitions must be registered so TensorFlow can
deserialize the saved .keras model that uses CBAM attention.
"""

import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc1 = layers.Dense(
            max(self.channels // self.reduction_ratio, 8),
            activation='relu', kernel_initializer='he_normal',
            use_bias=True, name=f'{self.name}_fc1'
        )
        self.fc2 = layers.Dense(
            self.channels, kernel_initializer='he_normal',
            use_bias=True, name=f'{self.name}_fc2'
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.avg_pool(inputs)))
        max_out = self.fc2(self.fc1(self.max_pool(inputs)))
        attention = tf.nn.sigmoid(avg_out + max_out)
        attention = tf.reshape(attention, [-1, 1, 1, self.channels])
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'reduction_ratio': self.reduction_ratio})
        return config


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1, kernel_size=self.kernel_size, strides=1,
            padding='same', activation='sigmoid',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{self.name}_conv'
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        return inputs * self.conv(concat)

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class CBAM(layers.Layer):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channel_attention = ChannelAttention(
            self.channels, self.reduction_ratio, name=f'{self.name}_channel_att'
        )
        self.spatial_attention = SpatialAttention(
            self.kernel_size, name=f'{self.name}_spatial_att'
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.spatial_attention(self.channel_attention(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size
        })
        return config
