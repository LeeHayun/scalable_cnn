from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow.contrib.eager as tfe

import os
import sys

sys.path.append(os.path.abspath('../models/research/slim'))

from vgg16 import Vgg16
from preprocessing import vgg_preprocessing
import itertools

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('data_dir', 'data', 'data directory')
#tf.app.flags.DEFINE_string('log_dir', '/tmp/hayun', 'logging directory')
tf.app.flags.DEFINE_integer('batch_size', 2, 'batch size')
tf.app.flags.DEFINE_integer('shuffle_buffer_size', 1024, 'Shuffle buffer size')
tf.app.flags.DEFINE_integer('num_map_threads', 4, 'Number of map threads')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'Number of classes')

#file_pattern = os.path.join(FLAGS.data_dir, 'train-*-of-*')
#dataset_fn = tf.data.TFRecordDataset
#parse_fn = lamdba x: parse_fn(x, is_train=True)

tf.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

def parse_fn(data):
    # file_name, image, label
    image = vgg_preprocessing.preprocess_for_train(data['image'], 224, 224,
                                                   resize_side_min=256, resize_side_max=512)
    one_hot = tf.one_hot(data['label'], FLAGS.num_classes)
    return image, one_hot


# Load a given dataset by name, along with the DatasetInfo
# Imagenet2012: train=1,281,167 / validation=50,000
dataset = tfds.load(name="imagenet2012", split=tfds.Split.TRAIN)


dataset = dataset.map(parse_fn, FLAGS.num_map_threads)
dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)
dataset = dataset.batch(FLAGS.batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

#print(dataset)
#for features in dataset.take(1):
#    print(features)

#vgg = Vgg16(vgg16_npy_path="vgg16.npy")
print('!!!!!!!!!!!!')

for images, labels in tfe.Iterator(dataset):
    print(images.shape, labels.shape)

#for features in dataset.take(2):
#    print(features['image'])
    #vgg.build(features['image'])

    #print(vgg.prob)
