#!/usr/bin/env python

import argparse

from os import environ as env
env['TF_CPP_MIN_LOG_LEVEL'] = '2'               # hide info & warnings
env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'       # grow GPU memory as needed

import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import tensorflow as tf
import tensorflow_datasets as tfds
import vgg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train VGG ConvNet on ILSVRC-2012 aka ImageNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument('--model', required=True,
                        choices=vgg.model_cfg.keys(),
                        help='VGG model configuration')
    parser.add_argument('--normalization',
                        choices=['batch'],
                        help='use normalization')
    parser.add_argument('--pooling', default='max',
                        choices=['max', 'avg'],
                        help='type of pooling')

    # Training configuration
    parser.add_argument('--dataset', default='imagenet2012',
                        choices=['imagenet2012', 'imagenette'],
                        help='image dataset to use in training')
    parser.add_argument('--data_dir', default='/tmp',
                        help='data directory')
    parser.add_argument('--optimizer', default='sgd',
                        choices=['adam', 'sgd'],
                        help='optimizer')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum used with SGD optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='dropout rate in fc1, fc2')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='rate of weight decay in fc1, fc2')

    # Loading / saving
    parser.add_argument('--load_model',
                        help='load a model to resume training')
    parser.add_argument('--save_model', default='saved/model.h5',
                        help='where to save the trained model')

    return parser.parse_args()


def image_preprocess(image):
    # Convert to float32 values in 0.0..1.0.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Isotropic resize & random crop
    H, W, C = tf.unstack(tf.shape(image))
    S = tf.random.uniform([], 256, 512)
    h = S * tf.maximum(1.0, tf.cast(H/W, dtype=tf.float32))
    w = S * tf.maximum(1.0, tf.cast(W/H, dtype=tf.float32))
    size = tf.cast([h,w], dtype=tf.int32)
    image = tf.image.resize(image, size)
    image = tf.image.random_crop(image, [224,224,C])

    # Normalize w.r.t. ImageNet mean / std
    return (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]


if __name__ == '__main__':

    args = parse_args()

    (train, val), info = tfds.load(
        args.dataset, split=['train', 'validation'],
        data_dir=args.data_dir,
        as_supervised=True,
        with_info=True,
    )

    num_classes = info.features['label'].num_classes

    def img_lbl_preprocess(image, label):
        return image_preprocess(image), tf.one_hot(label, num_classes)

    train = train.map(img_lbl_preprocess)
    train = train.batch(args.batch_size, drop_remainder=True)
    train = train.prefetch(tf.data.AUTOTUNE)

    val = val.map(img_lbl_preprocess)
    val = val.batch(args.batch_size, drop_remainder=True)
    val = val.prefetch(tf.data.AUTOTUNE)

    if args.load_model:
        model = tf.keras.models.load_model(args.load_model)
    else:
        model = vgg.get_model(
            cfg=args.model,
            normalization=args.normalization,
            pooling=args.pooling,
            dropout_rate=args.dropout_rate,
            weight_decay=args.weight_decay,
            classes=num_classes,
        )

    if args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
        )
    elif args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=args.momentum,
        )
    else:
        raise ValueError("Unknown optimizer")

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'],
    )
    model.fit(train, validation_data=val, epochs=args.epochs)

    model.save(args.save_model, save_traces=False)
