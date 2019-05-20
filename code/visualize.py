# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib import predictor

from dltk.io.augmentation import extract_random_example_array

from reader import read_fn
from skimage.transform import resize


READER_PARAMS = {'extract_examples': False}

def gradcam(layer,logit,pred):

    one_hot = tf.sparse_to_dense(pred, [2], 1.0)
    signal = tf.multiply(logit, one_hot)
    loss = tf.reduce_mean(signal)
    layer = tf.convert_to_tensor(layer)
    grads = tf.gradients(loss, layer)[0]
    # Normalizing the gradients
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    output, grads_val =  tf.Session().run([norm_grads])
    output = layer[0]  # [6,7,6,256]
    grads_val = grads_val[0]  # [6,7,6,256]

    weights = np.mean(grads_val, axis=(0, 1, 2))  # [256]
    cam = np.ones(output.shape[0: 3], dtype=np.float32)  # [6,7,6]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :,:, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = resize(cam, (91,109, 91))




def visulaize(args):

    export_dir = os.path.join(args.model_path, 'best/1556221214/')
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)
    print(my_predictor)

    test_df = pd.read_csv(args.test_csv)

    for output in read_fn(test_df,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=READER_PARAMS):

        img = output['features']['x']
        lbl = output['labels']['y']
        test_id = output['img_id']
        img = np.expand_dims(img,0)
        #y_, predictions, logits = my_predictor.session.run(fetches=[my_predictor._fetch_tensors['y_prob'],my_predictor._fetch_tensors['y_'],my_predictor._fetch_tensors['logits']], feed_dict={my_predictor._feed_tensors['x']: img})
        #op = my_predictor.session.graph.get_operations()
        layer, logits, prediction = my_predictor.session.run(fetches=[my_predictor.graph.get_tensor_by_name('unit_4_1/sub_unit1/conv3d/Conv3D:0'),my_predictor._fetch_tensors['logits'],my_predictor._fetch_tensors['y_']], feed_dict={my_predictor._feed_tensors['x']: img})
        gradcam(layer,logits,prediction)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='UKbb sex classification deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path', '-p',default='../Model/Ukbb_sex_classification/')
    parser.add_argument('--test_csv', default='/data/agelgazzar/PycharmProjects/Ukbb_GenderClassification/test.csv')
    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Call training
    visulaize(args)