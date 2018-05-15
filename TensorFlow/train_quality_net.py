# This is the training file for the DCNN paper, 
# "Saliency-based deep convolutional neural network for no-reference image quality assessment".
#
# Author: Sen Jia
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import time
import random

import tensorflow as tf
import numpy as np

import QualityNet 

SEED = 66478  # Set to None for random seed.

NUM_EPOCHS = 15 
DECAY_EPOCH = 5

BATCH_SIZE = 64
PATCH_SIZE = (32, 32)

NUM_CHANNELS = 1
MODEL_DIR = "tf_models"

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

def run():
    """
    Train the model using the specified input images and labels.
    The input images and labels will be fed into train_data_node and train_label_node respectively.
    The input images should be in the representation of 4d array (float32), (batch_size, width, height, depth).
    (64, 32, 32, 1) in this experiment.
    The input label is in the shape of (batch_size, 1) (float32). 
    """
    
    # pass your training data here, a list of training batch, [batch1, batch2], batch=(64,32,32,1) float32.
    train_data_lst = None
    train_label_lst = None

    bufsize = 0 

    tf.reset_default_graph()    
    with tf.Session() as sess:
        global_step = tf.train.get_or_create_global_step()
        batch = tf.Variable(0, dtype=tf.int32)
        epoch_counter = tf.Variable(0, dtype=tf.int32)

        train_data_node = tf.placeholder(data_type(),shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], NUM_CHANNELS))
        train_label_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1))

        net = QualityNet.QualityNet(train_data_node, NUM_CHANNELS, SEED=SEED)
        net.build_graph()
        logits = net.forward(train_data_node, train=True)

        loss = tf.divide(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(logits, train_label_node)))), BATCH_SIZE)
        var_list = net.parameters
        epoch_inc_op = tf.assign(epoch_counter, epoch_counter+1)

        learning_rate = tf.train.exponential_decay(
            0.01,
            epoch_counter,
            DECAY_EPOCH,
            0.1,
            staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)

        gvs = optimizer.compute_gradients(loss,var_list)
        capped_gvs = [(tf.clip_by_norm(gv[0], 1), gv[1]) for gv in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        saver = tf.train.Saver(var_list, max_to_keep=1)
        tf.global_variables_initializer().run()

        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0

            for train_data, train_labels in zip(train_data_lst, train_label_lst):

                feed_dict = {train_data_node: train_data,
                             train_label_node: train_labels}
                loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
                epoch_loss += loss_val[-1]

            sess.run(epoch_inc_op)
            print ("Epoch loss:", epoch_loss)
            saver.save(sess, os.path.join(MODEL_DIR, 'model_'+str(epoch+1)))

if __name__ == '__main__':
    run()
