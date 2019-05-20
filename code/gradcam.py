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
import nibabel as nib



READER_PARAMS = {'extract_examples': False}






def visulaize(args):

    '''export_dir = os.path.join(args.model_path, 'best/1550815178/')
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)'''

    with tf.Session(graph=tf.Graph()) as sess:
        path_to_model = os.path.join(args.model_path,'best/1551780240')
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],path_to_model)
        #op = sess.graph.get_operations()
        #([print(m.values())for m in op])


        test_df = pd.read_csv(args.test_csv)

        for output in read_fn(test_df,
                              mode=tf.estimator.ModeKeys.EVAL,
                              params=READER_PARAMS):

            img = output['features']['x']
            lbl = output['labels']['y']
            test_id = output['img_id']
            img = np.expand_dims(img,0)
            #x = tf.placeholder(tf.float32,[None,None,None,None,1])

            x = sess.graph.get_tensor_by_name('Placeholder:0')

            #layer = sess.graph.get_tensor_by_name('unit_4_1/sub_unit1/conv3d/Conv3D:0')
            #layer = sess.graph.get_tensor_by_name('unit_4_0/sub_unit_add/add:0')
            layer = sess.graph.get_tensor_by_name('pool/batch_normalization/batchnorm/mul_1:0')

            logits = sess.graph.get_tensor_by_name('last/hidden_units/MatMul:0')
            pred = sess.graph.get_tensor_by_name('pred/ArgMax:0')

            '''if lbl == 0:
                pred = tf.constant(1)
            elif lbl == 1:
                pred = tf.constant(0)'''


            one_hot = tf.sparse_to_dense(pred, [2], 1.0)
            signal = tf.multiply(logits, one_hot)
            loss = tf.reduce_mean(signal)

            grads = tf.gradients(loss, layer)[0]
            # Normalizing the gradients
            norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

            output, grads_val = sess.run([layer, norm_grads], feed_dict={x:img})
            output = output[0]  # [6,7,6,256]
            grads_val = grads_val[0]  # [6,7,6,,256]

            weights = np.mean(grads_val, axis=(0, 1, 2))  # [256]
            cam = np.ones(output.shape[0: 3], dtype=np.float32)  # [7,7]

            # Taking a weighted average
            for i, w in enumerate(weights):
                cam += w * output[:, :, :,i]

            # Passing through ReLU
            cam = np.maximum(cam, 0)
            cam = cam / np.max(cam)
            cam = resize(cam, (91, 109,91))
            img = nib.Nifti1Image(cam, np.eye(4)*2)
            nib.save(img, os.path.join(args.save_dir,'{}_{}.nii.gz'.format(test_id,lbl)))






if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='UKbb sex classification deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='1')
    parser.add_argument('--model_path', '-p',default='../Model/UKbb_catsdogs_classification3/')
    parser.add_argument('--save_dir' ,default= '/data_local/deeplearning/ukbb/DogsCatsVis/VariablePosition2/')
    parser.add_argument('--test_csv', default='test_dogscats.csv')
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