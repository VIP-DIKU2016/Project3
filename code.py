#!/usr/bin/python2
# -*- coding: utf-8 -*-

import argparse as ap
import cv2
import numpy as np
import os
from scipy.cluster.vq import *
import matplotlib.pyplot as plt

def extractSIFT(imagePaths):
    # List where all the descriptors are stored
    descriptors = np.array([])
    sift = cv2.xfeatures2d.SIFT_create()
    desList = [];
    for p in imagePaths:
        img = cv2.imread(p)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray,None)
        descriptors = np.append(descriptors, des) # Creates a list with image path and its descriptors
        desList.append((p, des));

    # Reshape for K-means to work
    desc = np.reshape(descriptors, (len(descriptors)/128, 128))
    desc = np.float32(desc)
    return desc, desList;

def main():
    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-train", "--training-set", help="rovide a path to the training set", required="True")
    parser.add_argument("-test", "--test-set", help="Provide a path to the test set", required="False")
    args = vars(parser.parse_args())

    # Get the training classes names and store them in a list
    training_set = args["training_set"]
    test_set = args["test_set"]

    #
    # 1. Preparing a list of training/testing paths 
    #

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

    testWords = os.listdir(test_set)

    testImagePaths = [];
    for w in testWords:
        wdir = os.path.join(test_set, w)
        for path, subdirs, files in os.walk(wdir):
            for name in files:
                if not name.startswith('.'):
                    testImagePaths.append(os.path.join(path, name))   


    #
    # 2. Extract SIFT Descriptors
    #
    desc, desList = extractSIFT(imagePaths);

    #
    # 3. Perform k-means clustering
    #

    k = 350;
    codebook, distortion = kmeans(desc, k);

    print codebook.shape;
    
    #
    # 4. Making the bag of words
    #

    trainBagOfWords = np.zeros((len(imagePaths), k));
    for i in xrange(len(imagePaths)):
        words, distance = vq(desList[i][1],codebook)
        for w in words:
            trainBagOfWords[i][w] += 1


    ##################################################################
    ### Indexing #####################################################
    ##################################################################

    #
    # 1. Extract SIFT
    # 

    testDesc, testDesList = extractSIFT(testImagePaths)

    #
    # 2. Making the bag of words
    # 

    testBagOfWords = np.zeros((len(testImagePaths), k));
    for i in xrange(len(testImagePaths)):
        words, distance = vq(testDesList[i][1],codebook)
        for w in words:
            testBagOfWords[i][w] += 1

    print testBagOfWords;

    table = []

    for i in xrange(len(imagePaths)):
        paths = imagePaths[i].split('/')

        table.append([
            imagePaths[i],
            paths[1], # 'train'
            paths[2], # true class
            trainBagOfWords[i]
        ])
    
    for i in xrange(len(testImagePaths)):
        paths = testImagePaths[i].split('/')

        table.append([
            testImagePaths[i],
            paths[1], # 'test'
            paths[2], # true class
            testBagOfWords[i]
        ])
    
    ##################################################################
    ### Retrieval ####################################################
    ##################################################################
main()