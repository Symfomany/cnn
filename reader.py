from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle

import numpy as np
import tensorflow as tf

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
NUM_HEIGHT = 32
NUM_WIDTH = 32
NUM_DEPTH = 3

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24
height = IMAGE_SIZE
width = IMAGE_SIZE

def _read_images(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def _file_to_images(data_path, train=True):
    if train:
        filenames = [os.path.join(data_path, 'data_batch_%d' % i) for i in range(1, 6)]
    else:
        filenames = [os.path.join(data_path, 'test_batch')]
    labels = None
    images = None
    for file in filenames:
        data = _read_images(file)
        labels = np.concatenate([labels, np.array(data[b'labels'])], axis=0) if not labels is None else np.array(data[b'labels'])
        images = np.concatenate([images, data[b'data']], axis=0) if not images is None else data[b'data']
    return (labels, images)

def _tranpose_data(label, image):
    label = tf.cast(label, tf.int32)

    # reshape image to array
    image = tf.reshape(image, [NUM_DEPTH, NUM_HEIGHT, NUM_WIDTH])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(image, [1, 2, 0])
    image = tf.cast(image, tf.float32)

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(image, [height, width, 3], seed=1)

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image, seed=1)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.

    # なぜかAll 0 になるのでひとまず保留
    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63, seed=1)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8, seed=1)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    return (label, float_image)


def cifar10_raw_data(data_path="cifar-10-batches-py"):
    train_data = _file_to_images(data_path, train=True)
    test_data = _file_to_images(data_path, train=False)
    return train_data, test_data

def cifar10_train_iterator(raw_data, batch_size):
    # エポック毎に順番をシャッフルしてデータを取得する
    label, image = tf.train.slice_input_producer([raw_data[0], raw_data[1]], shuffle=True, seed=1)
    label, image = _tranpose_data(label, image)
    
    # データをバッチ化してエンキューする
    labels, images = tf.train.batch([label, image], batch_size=batch_size)
    return (labels, images)

def cifar10_eval_iterator(raw_data, batch_size):
    label, image = tf.train.slice_input_producer([raw_data[0], raw_data[1]], shuffle=False)
    
    label = tf.cast(label, tf.int32)

    # reshape image to array
    image = tf.reshape(image, [NUM_DEPTH, NUM_HEIGHT, NUM_WIDTH])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(image, [1, 2, 0])
    reshaped_image = tf.cast(image, tf.float32)

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # データをバッチ化してエンキューする
    labels, images = tf.train.batch([label, float_image], batch_size=batch_size)
    return (labels, images)