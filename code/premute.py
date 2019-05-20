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

def premute_cat(file,mask):
    volume1 = nib.load(file).get_data()
    volume1 = np.interp(volume1, (volume1.min(), volume1.max()), (-1, +1))
    x0 = -1*np.random.randint(17,91)
    y0 = -1*np.random.randint(29,108)
    x1 = x0 + 16
    y1 = y0 + 28
    #volume1[-66:-50,-43:-15,30:50] = mask
    volume1[x0:x1, y0:y1, 30:50] = mask
    vol = nib.Nifti1Image(volume1, np.eye(4) * 2)
    nib.save(vol, file)


def premute_dog(file,mask):
    volume1 = nib.load(file).get_data()
    volume1 = np.interp(volume1, (volume1.min(), volume1.max()), (-1, +1))
    x0 = -1*np.random.randint(27,91)
    y0 = 1*np.random.randint(1,97)
    x1 = x0 + 26
    y1 = y0 + 12
    #volume1[-41:-15,20:32,30:50] = mask
    volume1[x0:x1, y0:y1, 30:50] = mask
    vol = nib.Nifti1Image(volume1, np.eye(4) * 2)
    nib.save(vol, file)

def process_catmask():
        img = mpimg.imread('/data/agelgazzar/Downloads/catmask.jpeg')
        gray = rgb2gray(img)
        gray = add_noise(gray,0.1)
        gray = cv2.resize(gray, None, fx=0.07, fy=0.07)
        a = gray[:, 4:20]
        a = np.transpose(a)
        mask = np.expand_dims(np.interp(a, (a.min(), a.max()), (-1, +1)), -1)
        mask = np.flip(mask)
        mask = np.tile(mask, (1, 1, 20))
        return mask

def process_dogmask():
    img = mpimg.imread('/data/agelgazzar/Downloads/dograw(1).jpg')
    a = rgb2gray(img)
    a = add_noise(a, 0.1)
    a = np.transpose(a)
    mask = np.expand_dims(np.interp(a, (a.min(), a.max()), (-1, +1)), -1)
    mask = np.flip(mask)
    mask = np.tile(mask, (1, 1, 20))
    return mask


def add_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == '__main__':
    path = '/data_local/deeplearning/UKBiobank/Release_17DEC2018/metadata.csv'
    dog_mask = process_dogmask()
    cat_mask = process_catmask()
    df = pd.read_csv(path)
    ids = df['f.eid']
    files = '/data_local/deeplearning/ukbb/T1_Cats_Dogs/'
    lbls = np.random.randint(2, size=4161)
    fs = []
    index = 0
    new_ids = []
    for i in ids:
        f = osp.join(files, '{}_T1_MNI_2mm.nii.gz'.format(i))
        if osp.isfile(f):
            fs.append(f)
            if lbls[index] == 0:
                premute_dog(f,dog_mask)
            else:
                premute_cat(f,cat_mask)
            new_ids.append(i)
            index += 1
            print(index)

    dic = {'ids': new_ids, 'paths': fs, 'sex': lbls}
    df = pd.DataFrame(dic)
    df.to_csv('Ukbb_dogs_cats_v2.csv')