#
# This file is used to load the pretrained model for Image Quality Assessment, based on the work of 
# "Saliency-based deep convolutional neural network for no-reference image quality assessment".
#
# This code requires the pre-processed(local normalized) image as input.
# Any saliency map can be passed into this code along with the input image.
#
# Author: Sen Jia
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np

import QualityNet 

BATCH_SIZE = 64
SEED = 66478  # Set to None for random seed.
PATCH_SIZE = (32, 32)
STEP_SIZE = int(PATCH_SIZE[0]/2)

NUM_CHANNELS = 1
MODEL_DIR = "quality_models"

def data_type():
    return tf.float32

def split_img(img, step=STEP_SIZE, s_map=None, threshold=0):
    patches = []
    if s_map is not None:
        avg_thre = PATCH_SIZE[0] * PATCH_SIZE[1] * threshold
    for y in range(0, img.shape[0], step):
        for x in range(0, img.shape[1], step):
            patch = img[y:y + PATCH_SIZE[1], x:x + PATCH_SIZE[0]]
            if s_map is not None:
                patch_importance = np.sum(s_map[y:y + PATCH_SIZE[1], x:x + PATCH_SIZE[0]])
                if patch_importance < avg_thre: #ignore the image patch if it is not important enough.
                    continue
            if patch.shape[:2] == PATCH_SIZE:
                for angle in range(0,4):
                    patches.append(np.rot90(patch,angle)) # rotate image for data augmentation.
    return patches

def predict():
    img = None # suppose to be a locally normalized 2d array, (h, m).
    s_map = None # optional, this is required when predicting only on salient image patches, 2d array in the value range of [0, 1].

    tf.reset_default_graph()    
    with tf.Session() as sess:
        global_step = tf.train.get_or_create_global_step()
        train_data_node = tf.placeholder(data_type(),shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], NUM_CHANNELS))
        test_data_node = tf.placeholder(data_type(),shape=(None, PATCH_SIZE[0], PATCH_SIZE[1], NUM_CHANNELS))
        net = QualityNet.QualityNet(train_data_node, NUM_CHANNELS, SEED=SEED)
        net.build_graph()
        prediction = net.forward(test_data_node)

        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        saver.restore(sess, os.path.join(MODEL_DIR, 'model')) # load the pretrained model.

        if s_map is None: 
            img_patches = split_img(img)
        else:
            img_patches = split_img(img, s_map=s_map, threshold=0.1)

        pred = sess.run(prediction,feed_dict={test_data_node:img_patches})
        score = np.mean(pred) / 99 # The regressor was trained on the LIVE label, [0, 99].
   
if __name__ == '__main__':
    predict()
