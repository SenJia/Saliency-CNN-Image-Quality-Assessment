# This is the model definition of the image quality assessment system, 
# "Saliency-based deep convolutional neural network for no-reference image quality assessment".
#
# Author: Sen Jia
#
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages

class QualityNet(object):

    def __init__(self, images, NUM_CHANNELS, SEED):
        """
        Args:
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, 1]
        """
        self._images = images
        self._num_channels = NUM_CHANNELS
        self._parameters = []
        self._initialized = False
        self._SEED = SEED
        self.device_id = ""
  
    def build_graph(self):
      self.global_step = tf.train.get_or_create_global_step()
      self._build_model()
      self._initialized = True

    @property
    def parameters(self):
        return self._parameters

    def _build_model(self):
        self.forward(self._images)

    def forward(self,x,train=False):
        with tf.variable_scope('conv1', reuse=self._initialized):
            x = self._conv('1', x, 3, self._num_channels, 32, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)
            x = self._conv('2', x, 3, 32, 32, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)
            x = self._max_pool(x, ksize=QualityNet._stride_arr(2), stride=self._stride_arr(2))

        with tf.variable_scope('conv2', reuse=self._initialized):
            x = self._conv('1', x, 3, 32, 64, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)
            x = self._conv('2', x, 3, 64, 64, self._stride_arr(1), device_name=self.device_id)            
            x = self._elu(x)
            x = self._max_pool(x, ksize=QualityNet._stride_arr(2), stride=self._stride_arr(2))
    
        with tf.variable_scope('conv3', reuse=self._initialized):
            x = self._conv('1', x, 3, 64, 128, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)
            x = self._conv('2', x, 3, 128, 128, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)
            x = self._max_pool(x, ksize=QualityNet._stride_arr(2), stride=self._stride_arr(2))

        with tf.variable_scope('conv4', reuse=self._initialized):
            x = self._conv('1', x, 3, 128, 256, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)
            x = self._conv('2', x, 3, 256, 256, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)
            x = self._max_pool(x, ksize=QualityNet._stride_arr(2), stride=self._stride_arr(2))
  
        with tf.variable_scope('conv5', reuse=self._initialized):
            x = self._conv('1', x, 3, 256, 512, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)
            x = self._conv('2', x, 3, 512, 512, self._stride_arr(1), device_name=self.device_id)
            x = self._elu(x)

        assert x.get_shape().ndims == 4
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, shape[1]*shape[2]*shape[3]])
        with tf.variable_scope('fc1', reuse=self._initialized):
            x = self._fully_connected(x, 2048, device_name=self.device_id, save=True)

        if train:
            x = tf.nn.dropout(x, 0.5, seed=self._SEED)

        with tf.variable_scope('fc2', reuse=self._initialized):
            x = self._fully_connected(x, 2048, device_name=self.device_id, save=True)

        if train:
            x = tf.nn.dropout(x, 0.5, seed=self._SEED)

        with tf.variable_scope('logit', reuse=self._initialized):
            logits = self._fully_connected(x, 1, device_name=self.device_id, save=True)
    
        return logits

    @staticmethod
    def _stride_arr(stride):
        return [1, stride, stride, 1]

    @staticmethod
    def _dtype():
        return tf.float32

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, device_name, save=True, trainable=True):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters

            kernel = tf.get_variable(
                'w', [filter_size, filter_size, in_filters, out_filters],
                 QualityNet._dtype(),
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=trainable
                 )

            bias = tf.get_variable("bias",shape=[out_filters],
                initializer=tf.constant_initializer(1),
                trainable=trainable
                )

            if save and not self._initialized:
                self._parameters.append(kernel)
                self._parameters.append(bias)
            out = tf.nn.conv2d(x, kernel, strides, padding='SAME')
            out = tf.nn.bias_add(out, bias)
        return out 
  
    def _fully_connected(self, x, out_dim, device_name, save=True, trainable=True):
        with tf.device(device_name):
            w = tf.get_variable(
	        'DW', [x.get_shape()[1], out_dim],
	        initializer=tf.initializers.variance_scaling(distribution="uniform"),
                trainable=trainable
            )
            b = tf.get_variable('biases', [out_dim],
                initializer=tf.constant_initializer(),
                trainable=trainable
            )
        if save and not self._initialized:
            self._parameters.append(w)
            self._parameters.append(b)
        return tf.nn.xw_plus_b(x, w, b)

    @staticmethod
    def _elu(x):
        return tf.nn.elu(x)

    @staticmethod
    def _max_pool(x,ksize,stride):
        return tf.nn.max_pool(x, ksize=ksize, strides=stride, padding='SAME')
