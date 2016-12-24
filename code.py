#!/usr/bin/python2
# -*- coding: utf-8 -*-

import argparse as ap
import cv2
import numpy as np
import os
from scipy.cluster.vq import *
import matplotlib.pyplot as plt


def main():
    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-train", "--training-set", help="rovide a path to the training set", required="True")
    # parser.add_argument("-test", "--test-set", help="Provide a path to the test set", required="False")
    args = vars(parser.parse_args())

    # Get the training classes names and store them in a list
    training_set = args["training_set"]
    # test_path = args["test-set"]

    ##################################################################
    ### 1. Preparing a list of training paths ########################
    ##################################################################

    words = os.listdir(training_set)

    # Discard the ones starting with . (like .DS_Store on Macs)
    words = [x for x in words if not x.startswith('.')]

    imagePaths = []
    for w in words:
        wdir = os.path.join(training_set, w)
        for path, subdirs, files in os.walk(wdir):
            for name in files:
                if not name.startswith('.'):
                    imagePaths.append(os.path.join(path, name))

    ##################################################################
    ### 2. Extract SIFT Descriptors ##################################
    ##################################################################

    # List where all the descriptors are stored
    descriptors = np.array([])
    sift = cv2.xfeatures2d.SIFT_create()
    des_list = [];
    for p in imagePaths:
        img = cv2.imread(p)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray,None)
        descriptors = np.append(descriptors, des) # Creates a list with image path and its descriptors
        des_list.append((p, des));

    # Reshape for K-means to work
    desc = np.reshape(descriptors, (len(descriptors)/128, 128))
    desc = np.float32(desc)

    ##################################################################
    ### 3. Perform k-means clustering ################################
    ##################################################################

    # Run K-means with k
    k = 350;
    codebook, distortion = kmeans(desc, k);

    print codebook.shape;

    ##################################################################
    ### 4. Making the histogram ######################################
    ##################################################################

    features = np.zeros((len(imagePaths), k));
    for i in xrange(len(imagePaths)):
        words, distance = vq(des_list[i][1],codebook)
        for w in words:
            print w;
            features[i][w] += 1


    ##################################################################
    ### Indexing #####################################################
    ##################################################################

    ##################################################################
    ### Retrieval ####################################################
    ##################################################################
main()