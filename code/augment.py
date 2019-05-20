import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os.path as osp
import os
import nibabel as nib
from nilearn.image import smooth_img
from sklearn.preprocessing import minmax_scale
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import PIL
from PIL import Image
import cv2
import random



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])




def augment(file,aug):
    volume1 = nib.load(file).get_data()
    if aug == 'x':
        volume1 = np.flip(volume1, axis=0)
    if aug == 'y':
        volume1 = np.flip(volume1, axis=1)
    if aug == 'xy':
        volume1 = np.flip(volume1,axis=0)
        volume1 = np.flip(volume1,axis=1)

    vol = nib.Nifti1Image(volume1, np.eye(4) * 2)
    nib.save(vol, file)




if __name__ == '__main__':
    path = '/data_local/deeplearning/UKBiobank/Release_17DEC2018/metadata.csv'
    print('Reading csv file ....')
    df = pd.read_csv(path)
    ids = df['f.eid']
    files = '/data_local/deeplearning/ukbb/T1_MNI_2mm_Augmented/'
    fs = []
    sexs = []
    index = 0
    new_ids = []
    augmetation = []
    augmetation_options = ["none","x","y","xy"]
    print('Preprocessing data ...')
    for i in ids:
        f = osp.join(files, '{}_T1_MNI_2mm.nii.gz'.format(i))
        if osp.isfile(f):
            fs.append(f)
            if df['f.31.0.0'].iloc[index] == 'Male':
                s = 0
            else:
                s = 1
            aug = np.random.choice(augmetation_options)
            augment(f,aug)
            augmetation.append(aug)
            sexs.append(s)
            new_ids.append(i)
            index += 1
            if (index%100==0):
                print(index)

    dic = {'ids': new_ids, 'paths': fs, 'sex': sexs, 'aug': augmetation}
    df = pd.DataFrame(dic)
    df.to_csv('Ukbb_sex_augmented.csv')
    train, validate, test = np.split(df.sample(frac=1), [int(.7 * len(df)), int(.8 * len(df))])
    train.to_csv('train_sex_augmented.csv')
    test.to_csv('test_sex_augmented.csv')
    validate.to_csv('val_sex_augmented.csv')
