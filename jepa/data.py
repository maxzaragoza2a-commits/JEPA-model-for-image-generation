"""CIFAR-10 input pipeline.

Two dataset builders, because the project has two distinct phases:

* `build_pretrain_dataset` -- JEPA pretraining is self-supervised, so it yields
  images only, with no labels at all.
* `build_probe_dataset` -- linear-probe evaluation freezes the encoder and
  fits a classifier on top, so it yields (image, label) pairs.
"""

import numpy as np
import tensorflow as tf
import keras

from jepa.config import JEPAConfig

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

AUTOTUNE = tf.data.AUTOTUNE


def load_cifar10():
    """Download (once, cached in ~/.keras/datasets) and return raw uint8 arrays."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Keras hands back labels shaped (N, 1); flatten to (N,).
    return (x_train, y_train.squeeze(-1)), (x_test, y_test.squeeze(-1))


def standardise(images, config: JEPAConfig):
    """uint8 [0, 255] -> float32, zero mean and unit variance per channel."""
    images = tf.cast(images, tf.float32) / 255.0
    mean = tf.constant(config.mean, dtype=tf.float32)
    std = tf.constant(config.std, dtype=tf.float32)
    return (images - mean) / std


def augment(image, config: JEPAConfig):
    """Random crop (with 4px reflect padding) plus a random horizontal flip.

    Kept deliberately cheap: the masking in JEPA already supplies most of the
    learning signal, and heavy augmentation is expensive on CPU.
    """
    pad = 4
    image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    image = tf.image.random_crop(
        image, size=[config.image_size, config.image_size, config.num_channels]
    )
    image = tf.image.random_flip_left_right(image)
    return image


def build_pretrain_dataset(config: JEPAConfig, split: str = "train"):
    """Self-supervised stream of augmented images. No labels."""
    (x_train, _), (x_test, _) = load_cifar10()
    images = x_train if split == "train" else x_test

    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.shuffle(10_000, seed=config.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda img: augment(img, config), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img: standardise(img, config), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def build_probe_dataset(config: JEPAConfig, split: str = "train", shuffle: bool = None):
    """(image, label) pairs for linear-probe evaluation of a frozen encoder."""
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    images, labels = (x_train, y_train) if split == "train" else (x_test, y_test)
    if shuffle is None:
        shuffle = split == "train"

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(10_000, seed=config.seed, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda img, lab: (augment(img, config), lab), num_parallel_calls=AUTOTUNE
        )
    ds = ds.map(
        lambda img, lab: (standardise(img, config), lab), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(config.batch_size).prefetch(AUTOTUNE)
    return ds
