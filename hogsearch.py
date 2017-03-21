import numpy as np
import cv2
from skimage.feature import hog
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import csv


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
   
def get_hog(feature_image, hog_channel = 'ALL', vis = False, 
            feat_vec = True, pix_per_cell = 8, cells_per_block = 2, orientation = 9):
    
    ppc=(pix_per_cell, pix_per_cell)
    cpb=(cells_per_block, cells_per_block)
    orient = orientation
    
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(hog(feature_image[:,:,channel], orientations = orient,
                       pixels_per_cell = ppc,
                       cells_per_block = cpb, 
                       transform_sqrt = False, 
                       visualise = vis, feature_vector = feat_vec))
        if feat_vec:
            hog_features = np.ravel(hog_features)        
        
    else:
        hog_features = hog(feature_image[:,:,channel], orientations = orient,
                       pixels_per_cell = ppc,
                       cells_per_block = cpb, 
                       transform_sqrt = False, 
                       visualise = vis, feature_vector = feat_vec)
    return hog_features

def get_features(name, ppc, cpb, orient, cspace = 'HLS'):
    """
    Get all features for one image file
    do all necessary transformations
    main feature hyperparams are set here
    Returns feature vector
    """
    image = cv2.imread(name)
    if image.shape != (64, 64):
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        
    if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    else: feature_image = np.copy(image)
        
    spatial_features = bin_spatial(feature_image, size=(16, 16))
    hist_features = color_hist(feature_image, nbins = 32)
    hog_features = get_hog(feature_image, pix_per_cell = ppc, 
                            cells_per_block = cpb, orientation = orient )
    return np.concatenate((spatial_features, hist_features, hog_features))

def hog_check(ppc, cpb, orient):
    vehicle_dirs = [('vehicles/GTI_Far', 'image0226.png'),
                ('vehicles/GTI_Left', 'image0220.png'), 
                ('vehicles/GTI_MiddleClose', 'image0125.png'),
                ('vehicles/GTI_Right', 'image0219.png')]
    gti_train = []
    gti_test = []

    """ Extract features from GTI images.
        Split them according to manually selected boundaries to 
        prevent having similar pictures in train and test
    """

    for d, test_limit in vehicle_dirs:
        test = True
        for f in sorted(os.listdir(d + '/')):
            if f.endswith('.png'):
                features = get_features(d + '/' +f, ppc = ppc, cpb = cpb, orient = orient)
                if test:
                    gti_test.append(features)
                else:
                    gti_train.append(features)
                if f == test_limit:
                    test = False

    gti_train_y = np.ones(len(gti_train))
    gti_test_y = np.ones(len(gti_test))
    gti_train = np.array(gti_train)
    gti_test = np.array(gti_test)

    vehicle_features = []
    d = 'vehicles/KITTI_extracted'
    for f in os.listdir(d + '/'):
            if f.endswith('.png'):
                vehicle_features.append(get_features(d + '/' +f, ppc = ppc, cpb = cpb, orient = orient))
    nonvehicle_dirs = ['non-vehicles/GTI', 'non-vehicles/Extras']

    nonvehicle_features = []
    for d in nonvehicle_dirs:
        for f in os.listdir(d + '/'):
            if f.endswith('.png'):
                nonvehicle_features.append(get_features(d + '/' +f, ppc = ppc, cpb = cpb, orient = orient))
    #prepare labels
    y = np.hstack((np.ones(len(vehicle_features)), 
                np.zeros(len(nonvehicle_features))))

    # Create an array stack of feature vectors
    X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    X_trans = X_scaler.transform(X)
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X_trans, y, test_size=0.2, random_state=rand_state)

    # Scale GTI data
    gti_train = X_scaler.transform(gti_train)
    gti_test = X_scaler.transform(gti_test)

    # Add GTI data to the rest
    X_train = np.vstack((X_train, gti_train))
    y_train = np.concatenate((y_train, gti_train_y))

    X_test = np.vstack((X_test, gti_test))
    y_test = np.concatenate((y_test, gti_test_y))

    # Shuffle train and test 
    X_train, y_train = shuffle(X_train, y_train, random_state = rand_state)
    X_test, y_test = shuffle(X_test, y_test, random_state = rand_state)   
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    return((ppc, cpb, orient, accuracy, X.shape[1]))

result =[]
bps = [4, 8, 12, 16]
cpb = [2, 3, 4]
orient = [6, 8, 9, 10, 12]
hog_space = list(product(bps, cpb, orient))
result = [(1, 2, 3, 4), (5, 6, 7, 8)]

for i in tqdm(hog_space):
    result.append(hog_check(i[0], i[1], i[2]))

with open('hog_space.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(result)
