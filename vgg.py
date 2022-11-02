import tensorflow as tf


class Conv2DLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        normalization=None,
        activation='relu',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.normalization = normalization
        self.activation = activation

        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            use_bias=(normalization is None),
        )

        if normalization is None:
            self.norm = None
        elif normalization == 'batch':
            self.norm = tf.keras.layers.BatchNormalization()
        else:
            raise ValueError("Unknown type of normalization")
        
        self.act = tf.keras.layers.Activation(activation)

    def call(self, x, training=False):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x, training=training)
        return self.act(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'normalization': self.normalization,
            'activation': self.activation,
        })
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)


model_cfg = {
    # Configuration A
    'vgg11': [
        [ {'filters': 64} ],
        [ {'filters': 128} ],
        [ {'filters': 256}, {'filters': 256} ],
        [ {'filters': 512}, {'filters': 512} ],
        [ {'filters': 512}, {'filters': 512} ],
    ],
    # Configuration B
    'vgg13': [
        [ {'filters': 64}, {'filters': 64} ],
        [ {'filters': 128}, {'filters': 128} ],
        [ {'filters': 256}, {'filters': 256} ],
        [ {'filters': 512}, {'filters': 512} ],
        [ {'filters': 512}, {'filters': 512} ],
    ],
    # Configuration C
    'vgg16-1': [
        [ {'filters': 64}, {'filters': 64} ],
        [ {'filters': 128}, {'filters': 128} ],
        [ {'filters': 256}, {'filters': 256},
          {'filters': 256, 'kernel_size': (1,1)} ],
        [ {'filters': 512}, {'filters': 512},
          {'filters': 512, 'kernel_size': (1,1)} ],
        [ {'filters': 512}, {'filters': 512},
          {'filters': 512, 'kernel_size': (1,1)} ],
    ],
    # Configuration D
    'vgg16': [
        [ {'filters': 64}, {'filters': 64} ],
        [ {'filters': 128}, {'filters': 128} ],
        [ {'filters': 256}, {'filters': 256}, {'filters': 256} ],
        [ {'filters': 512}, {'filters': 512}, {'filters': 512} ],
        [ {'filters': 512}, {'filters': 512}, {'filters': 512} ],
    ],
    # Configuration E
    'vgg19': [
        [ {'filters': 64}, {'filters': 64} ],
        [ {'filters': 128}, {'filters': 128} ],
        [ {'filters': 256}, {'filters': 256},
          {'filters': 256}, {'filters': 256} ],
        [ {'filters': 512}, {'filters': 512},
          {'filters': 512}, {'filters': 512} ],
        [ {'filters': 512}, {'filters': 512},
          {'filters': 512}, {'filters': 512} ],
    ],
}


def get_model(
    cfg,
    input_shape=[224,224,3],
    normalization=None,
    pooling='max',
    include_top=True,
    classes=1000,
    classifier_activation='softmax',
    dropout_rate=0.5,
    weight_decay=5e-4,
):
    if isinstance(cfg, str):
        cfg = model_cfg[cfg]

    if pooling == 'max':
        PoolingLayer = tf.keras.layers.MaxPool2D
    elif pooling == 'avg':
        PoolingLayer = tf.keras.layers.AveragePooling2D
    else:
        raise ValueError("Unknown pooling type")

    input = tf.keras.layers.Input(shape=input_shape)

    x = input
    for n, block in enumerate(cfg):
        for i, layer_cfg in enumerate(block):
            x = Conv2DLayer(
                normalization=normalization,
                name=f'block{n+1}_conv{i+1}',
                **layer_cfg,
            )(x)
        x = PoolingLayer(
            pool_size=(2,2),
            strides=(2,2),
            padding='same',
            name=f'block{n+1}_pool',
        )(x)

    if include_top is True:
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            activity_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
            name='fc1',
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name='dropout1')(x)
        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            activity_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
            name='fc2',
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name='dropout2')(x)
        x = tf.keras.layers.Dense(
            units=classes,
            activation=classifier_activation,
            name='predictions',
        )(x)

    return tf.keras.Model(inputs=input, outputs=x)
