from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

import os
import sys

sys.path.append(os.path.abspath('../models/research/slim'))

import vgg16
from preprocessing import vgg_preprocessing
import itertools

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/hayun', 'logging directory')
tf.app.flags.DEFINE_integer('batch_size', 2, 'batch size')
tf.app.flags.DEFINE_integer('shuffle_buffer_size', 1024, 'Shuffle buffer size')
tf.app.flags.DEFINE_integer('num_map_threads', 4, 'Number of map threads')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'Number of classes')

def main(_):
    # Horovod: initialize Horovod.
    # hvd.init()

    # Horovod: pin GPU to be sued to process local rank (one GPU per process)
    #config = tf.ConfigProto()
    #config.gpu_options.visible_device_list = str(hvd.local_rank())

    def parse_fn(data):
        # file_name, image, label
        image = vgg_preprocessing.preprocess_for_train(data['image'], 224, 224,
                                                       resize_side_min=256, resize_side_max=512)
        return image, data['label']

    # Horovod: adjust learning rate based on number of GPUs
    #opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

    # Load a given dataset by name, along with the DatasetInfo
    # Imagenet2012: train=1,281,167 / validation=50,000
    dataset = tfds.load(name="imagenet2012", split=tfds.Split.TRAIN)


    dataset = dataset.map(parse_fn, FLAGS.num_map_threads)
    dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    checkpoint_dir = FLAGS.checkpoint_dir
    step_counter = tf.train.get_or_create_global_step()

    hooks = [tf.train.StopAtStepHook(last_step=1)]
    config = tf.ConfigProto(log_device_placement=False)

    #vgg = Vgg16(vgg16_npy_path="vgg16.npy")
    ph_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    ph_output = tf.placeholder(tf.float32, shape=(None))

    vgg = vgg16.vgg16(FLAGS)
    logits = vgg.inference(ph_input)
    loss = vgg.loss(logits, ph_output)
    train_op = vgg.train(loss, step_counter) 

    """
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        mon_sess.run(iterator.initializer)
        #while not mon_sess.should_stop():
            # Run a training step synchronously.
            #mon_sess.run(vgg.train_op)
    """

if __name__ == '__main__':
    tf.app.run()
