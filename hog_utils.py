import numpy as np
import cv2
from skimage.feature import hog
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
import time
from itertools import product
from tqdm import tqdm
import csv


def bin_spatial(img, size):
    color1 = cv2.resize(img[:,:,0], (size, size)).ravel()
    color2 = cv2.resize(img[:,:,1], (size, size)).ravel()
    color3 = cv2.resize(img[:,:,2], (size, size)).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
   
def get_hog(feature_image, pix_per_cell, cells_per_block, orientation,
            hog_channel = 'ALL', vis = False, 
            feat_vec = True, ):
    
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

def convert_color(image, cspace):
    """
    cConvert image to desired color space
    """
    if cspace != 'BGR':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCbCr':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb )
    else: feature_image = np.copy(image)
    return feature_image

def get_features(name, ppc , cpb, orient, nbins, spat_size, cspace):
    """
    Get all features for one image file
    do all necessary transformations
    main feature hyperparams are set here
    Returns feature vector
    """
    image = cv2.imread(name)
    if image.shape != (64, 64):
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        
    feature_image = convert_color(image, cspace)
    
    spatial_features = bin_spatial(feature_image, size= spat_size)
    hist_features = color_hist(feature_image, nbins = nbins)
    hog_features = get_hog(feature_image, pix_per_cell = ppc, cells_per_block = cpb,
                            orientation = orient)
    return np.concatenate((spatial_features, hist_features, hog_features))

def model_check(ppc, cpb, orient , nbins, spat_size, cspace):
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
        print("{} images".format(d))
        for f in tqdm(sorted(os.listdir(d + '/'))):
            if f.endswith('.png'):
                features = get_features(d + '/' +f, ppc = ppc, cpb = cpb, 
                                        orient = orient, nbins = nbins, 
                                        spat_size = spat_size, cspace = cspace)
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
    print("KITTI images")
    for f in tqdm(os.listdir(d + '/')):
            if f.endswith('.png'):
                vehicle_features.append(get_features(d + '/' +f, ppc = ppc, cpb = cpb, 
                                        orient = orient, nbins = nbins, 
                                        spat_size = spat_size, cspace = cspace))
    nonvehicle_dirs = ['non-vehicles/GTI', 'non-vehicles/Extras']

    nonvehicle_features = []
    print("Non vehicle images")
    for d in tqdm(nonvehicle_dirs):
        for f in tqdm(os.listdir(d + '/')):
            if f.endswith('.png'):
                nonvehicle_features.append(get_features(d + '/' +f, ppc = ppc, cpb = cpb, 
                                        orient = orient, nbins = nbins, 
                                        spat_size = spat_size, cspace = cspace))
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
    result = {'classifier': svc, 'size': X.shape[1], 'scaler': X_scaler , 'accuracy': accuracy}
    return result

class slider:
    """
    Sliding window class
    slide method is a generator of sliding window of size defined by tuple size (xsize, ysize)
    the sliding is done within the limits defined by tuples (min, max) for x and y direction
    Next window is moved by step from previous one (one value vor both x and y)

    """
    def __init__(self, img, xlimits, ylimits, scaler, classifier,  
                size = 64, step = 8, cspace = 'HLS', scale = 1):

        if xlimits[0] is not None:
            self.min_x = xlimits[0]
        else:
            self.min_x = 0
        
        if xlimits[1] is not None:
            self.max_x = xlimits[1]
        else:
            self.max_x = img.shape[1]
        
        if ylimits[0] is not None:
            self.min_y = ylimits[0]
        else:
            self.min_y = 0
        
        if ylimits[1] is not None:
            self.max_y = ylimits[1]
        else:
            self.max_y = img.shape[0]
        
        self.size = size
        self.step = step
               
        self.image = convert_color(img, cspace)
       
        # instances of scaler and clasiffier to deal with slide window data
        self.scaler = scaler
        self.classifier = classifier
        
        
        self.scale = scale
        
        if scale !=1:
            self.min_x = int(self.min_x * scale)
            self.max_x = int(self.max_x * scale)
            
            self.min_y = int(self.min_y * scale)
            self.max_y = int(self.max_y * scale)
            
            w = int(self.image.shape[1] * scale)
            h = int(self.image.shape[0] * scale)
            self.image = cv2.resize(self.image, (w, h))
        
           
    def slide(self):
        """
        Sliding window generator
        returns window of declared size from region of interest defined
        by max_x, min z, max_y, min_y 
        Also returns coordinates of top left corner of the sliding window
        """
        for y in range(self.max_y - self.size, self.min_y, -self.step):
            for x in range(self.min_x, self.max_x - self.size, self.step):
            # yield the current window
                window = self.image[y:y + self.size, x:x + self.size]
                yield (x, y, window)
    
    def features(self):
        """
        Features generator, returns features and top left corner of the sliding window
        """
        for x, y , window in self.slide():
            spatial_features = bin_spatial(window, size=(16, 16))
            hist_features = color_hist(window, nbins = 32)
            #h1 = self.hog_subsample(x, y)
            h2 = get_hog(window)
            hog_features = h2
            out = np.concatenate((spatial_features, hist_features, hog_features))
            out = out.reshape(1, -1)
            out = self.scaler.transform(out)
            yield x, y, out
        
    def predict(self):
        """
        Generate list of predictions for an image, returns tuple of 2 points - topleft and bottom right corners of the window
        """
        predictions = []
        for x, y, feat in self.features():
            res = self.classifier.predict(feat)
            if res == 1:
                point1 = (int(x / self.scale), int(y / self.scale))
                point2 = (int((x + self.size)/ self.scale), int((y + self.size)/ self.scale))
                predictions.append((point1, point2))
        return(predictions)


if __name__ == "__main__":
    result =[]
    bps = [8, 12, 16]
    cpb = [2, 3, 4]
    orient = [6, 8, 9, 10, 12]
    cspace = ['HLS', 'YCbCr' ]
    hog_space = list(product(bps, cpb, orient, cspace))
    results = []

    for i in tqdm(hog_space):
        check = model_check(ppc = i[0], cpb = i[1], orient = i[2], 
        nbins = 32, spat_size = 16, cspace = i[3])
        pars = {'ppc': i[0], 'cpb': i[1], 'orient': i[2]}
        res = {**check, **pars}
        results.append(res)
    
    pickle.dump(results, open('search_results.p', 'wb'))
    with open('hog_space.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            line = [i['ppc'], i['cpb'], i['orient'], i['size'], i['accuracy']]
            writer.writerow(result)
