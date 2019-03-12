from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Global constants describing the imagenet2012 data set.
IMAGE_SIZE = 224
NUM_CLASSES = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1281167
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 50000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

WEIGHT_DECAY = 0.0005

DEBUG_MODE = True

def _variable_on_device(name, shape, initializer, trainable=True):
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    var = _variable_on_device(name, shape, initializer, trainable)
    if wd is not None and trainable:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


class vgg16(object):
    def __init__(self, flags):
        self.npy_weights = np.load('vgg16.npy', encoding='latin1').item()
        self.model_params = []
        self.batch_size = flags.batch_size


    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


    def inference(self, images):
        conv1_1 = self._conv2d('conv1_1', images, 64, 3, 1)
        conv1_2 = self._conv2d('conv1_2', conv1_1, 64, 3, 1)
        pool1 = self._max_pool('pool1', conv1_2, 2, 2)

        conv2_1 = self._conv2d('conv2_1', pool1, 128, 3, 1)
        conv2_2 = self._conv2d('conv2_2', conv2_1, 128, 3, 1)
        pool2 = self._max_pool('pool2', conv2_2, 2, 2)

        conv3_1 = self._conv2d('conv3_1', pool2, 256, 3, 1)
        conv3_2 = self._conv2d('conv3_2', conv3_1, 256, 3, 1)
        conv3_3 = self._conv2d('conv3_3', conv3_2, 256, 3, 1)
        pool3 = self._max_pool('pool3', conv3_3, 2, 2)

        conv4_1 = self._conv2d('conv4_1', pool3, 512, 3, 1)
        conv4_2 = self._conv2d('conv4_2', conv4_1, 512, 3, 1)
        conv4_3 = self._conv2d('conv4_3', conv4_2, 512, 3, 1)
        pool4 = self._max_pool('pool4', conv4_3, 2, 2)

        conv5_1 = self._conv2d('conv5_1', pool4, 512, 3, 1)
        conv5_2 = self._conv2d('conv5_2', conv5_1, 512, 3, 1)
        conv5_3 = self._conv2d('conv5_3', conv5_2, 512, 3, 1)
        pool5 = self._max_pool('pool5', conv5_3, 2, 2)

        fc6 = self._fc('fc6', pool5, 4096, flatten=True)
        fc7 = self._fc('fc7', fc6, 4096)
        logits = self._fc('fc8', fc7, 1000, activation=None)

        return logits

    def loss(self, logits, labels):
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self, total_loss, global_step):
        # Variablers that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = _add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

            # Gradient aksdfjsdlafjaksdjfaje;flkawje

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                print(var)
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        with tf.control_dependencies([apply_gradient_op]):
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

        return variables_averages_op


    def _conv2d(self, layer_name, inputs, filters, size, stride, padding='SAME',
                freeze=False, xavier=False, activation='relu', stddev=0.001):

        use_pretrained_param = False
        if self.npy_weights is not None:
            nw = self.npy_weights
            if layer_name in nw:
                kernel_val = nw[layer_name][0]
                bias_val = nw[layer_name][1]

                # Check the shape
                if (kernel_val.shape == (size, size, inputs.get_shape().as_list()[-1], filters)) \
                        and (bias_val.shape == (filters, )):
                    use_pretrained_param = True
                else:
                    print('Shape of the pretrained parameter of %s does not match, '
                          'use randomly initialized parameter' % layer_name)
            else:
                print('Cannot find %s in the pretrained model. Use randomly initialized '
                      'parameters' % layer_name)

        if DEBUG_MODE:
            print('Input tensor shape to %s: %s' % (layer_name, str(inputs.get_shape())))

        with tf.variable_scope(layer_name) as scope:
            channels = inputs.get_shape()[3]

            if use_pretrained_param:
                if DEBUG_MODE:
                    print('Using pretrained model for %s' % layer_name)
                kernel_init = tf.constant(kernel_val, dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            kernel = _variable_with_weight_decay('kernels', [size, size, int(channels), filters],
                    wd=WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
            biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
            self.model_params += [kernel, biases]

            conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding,
                    name='convolution')
            conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

            if activation == 'relu':
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

            return out
        

    def _fc(self, layer_name, inputs, hiddens, flatten=False, activation='relu',
            xavier=False, stddev=0.001):

        use_pretrained_param = False
        if self.npy_weights is not None:
            nw = self.npy_weights
            if layer_name in nw:
                use_pretrained_param = True
                kernel_val = nw[layer_name][0]
                bias_val = nw[layer_name][1]

        if DEBUG_MODE:
            print('Input tensor shape to %s: %s' % (layer_name, str(inputs.get_shape())))

        with tf.variable_scope(layer_name) as scope:
            input_shape = inputs.get_shape().as_list()
            if flatten:
                dim = input_shape[1] * input_shape[2] * input_shape[3]
                inputs = tf.reshape(inputs, [-1, dim])
            else:
                dim = input_shape[1]

            if use_pretrained_param:
                if DEBUG_MODE:
                    print('Using pretrained model for %s' % layer_name)
                kernel_init = tf.constant(kernel_val, dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                kernel_init = tf.contrib.layers.xavier_initializer()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            weights = _variable_with_weight_decay('weights', shape=[dim, hiddens],
                    wd=WEIGHT_DECAY, initializer=kernel_init)
            biases = _variable_on_device('biases', [hiddens], bias_init)
            self.model_params += [weights, biases]

            outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
            if activation == 'relu':
                out = tf.nn.relu(outputs, 'relu')
            else:
                out = outputs

            return out

    def _max_pool(self, layer_name, inputs, size, stride, padding='SAME'):
        with tf.variable_scope(layer_name) as scope:
            out = tf.nn.max_pool(inputs, ksize=[1, size, size, 1],
                    strides=[1, stride, stride, 1],
                    padding=padding)
            return out


