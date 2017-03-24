import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import glob
import pickle
from collections import deque


class Line:
    def __init__(self, madx3):
        self.detected = None
        # How many xfit values from the past to keep
        self.depth = 12
        self.last_fits = deque([])
        self.mean_fit = None
        self.madx3 = madx3
        self.curvature = None
        self.ym_per_pix = 30/720 
        self.xm_per_pix = 3.7/700
        # number of frames which could be substituted in case of bad checkup
        # i.e MAD tests
        self.nd_limit = 10
        # Non detected counter
        self.nd = 0
    
    def curve(self, fitx, ploty):

        fit = np.polyfit(ploty * self.ym_per_pix, fitx * self.xm_per_pix, 2)
        y = np.max(ploty) * self.ym_per_pix
        a = fit[0]
        b = fit[1]
        curve = np.power(1 + np.square(2*a*y +b), 1.5)/(2*np.abs(a))
        return curve        
    
    def update(self, xfit, ploty):
        self.last_fits.append(xfit)
        if len(self.last_fits) > self.depth:
            self.last_fits.popleft()
        self.mean_fit = np.mean(np.array(self.last_fits), axis = 0)
        self.mean_fit = np.int32(self.mean_fit)
        self.detected = True
        self.curvature = self.curve(xfit, ploty)

    def diff(self, xfit, ploty):
        n_fits = len(self.last_fits)
        if n_fits > 0:
            last_fit = self.last_fits[n_fits -1]
            d = np.sum(last_fit - xfit)
            if np.abs(d) < self.madx3:
                self.update(xfit, ploty)
            elif self.nd == self.nd_limit:
                self.update(xfit, ploty)
            else:
                self.nd += 1
                self.detected = False    
        else:
            # In case of first fit
           self.update(xfit, ploty)



class FrameProcessor:
    def __init__(self, calib_dir = 'camera_cal/', nx = 9, ny = 6):
        self.left = Line(madx3 = 7143.995)
        self.right = Line(madx3 = 11121.39)
        self.calib_dir = calib_dir
        self.nx = nx
        self.ny = ny
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.xsize = None
        self.ysize = None
        self.Minv = None
        self.log = []
        self.ploty = None
        #Processed frames counter
        self.counter = 0
        try:
            f = open('calibration.pickle', 'rb')
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = pickle.load(f)
        except IOError:
            self.calibrate()
                
    def calibrate(self):
        obp = np.zeros((self.nx*self.ny, 3), np.float32)
        obp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        for file in os.listdir(self.calib_dir):
            if file.endswith(".jpg"):
                img = cv2.imread(self.calib_dir + file)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
                if ret:
                    objpoints.append(obp)
                    imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            self.ret = ret
            self.mtx = mtx
            self.dist = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            with open('calibration.pickle', 'wb') as calib_file:
                pickle.dump([ret, mtx, dist, rvecs, tvecs], calib_file)
   
    def beye(self, img):
        src = np.float32([[self.xsize // 2 - 76, self.ysize * .625], 
        [self.xsize // 2 + 76, self.ysize * .625],
        [-100, self.ysize], [self.xsize + 100, self.ysize]])
        dst = np.float32([[100, 0], [self.xsize - 100, 0], [100, self.ysize], 
        [self.xsize - 100, self.ysize]])   
        M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src) 
        warped = cv2.warpPerspective(img, M, (self.xsize, self.ysize))
        return warped

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        binary_output = self.threshold(gradmag, thresh)
        return binary_output

    def abs_sobel_thresh(self, img, thresh, orient='x', sobel_kernel=3):
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = self.threshold(scaled_sobel, thresh)
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  self.threshold(absgraddir, thresh)

        # Return the binary image
        return binary_output

    def threshold(self, img, thresh):
        binary_output =  np.zeros_like(img, dtype = np.uint8)
        binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return binary_output

    def pipe(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        birds_eye = self.beye(undist)
        hsv = cv2.cvtColor(birds_eye, cv2.COLOR_RGB2HSV)
        
        gray = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2HLS)
        s_channel = hls[:,:,2]

        deshadow_r = self.threshold(birds_eye[:,:,0], (120, 255))
        deshadow_g = self.threshold(birds_eye[:,:,1], (120, 255))
        deshadow_b = self.threshold(birds_eye[:,:,2], (120, 255))
        deshadow = np.bitwise_or(deshadow_r, deshadow_g, deshadow_b)
    
        dsx = self.abs_sobel_thresh(gray, (20, 100), orient = 'x')
        mag = self.mag_thresh(s_channel, sobel_kernel = 5, thresh= (20, 100))
        s_threshold = self.threshold(s_channel, (170, 255))
        out = np.bitwise_and(np.bitwise_or(dsx, s_threshold, mag), deshadow)
        return out, undist, birds_eye


    def slide(self, warped):
        x_middle = np.int(np.round(self.xsize/2))
        window_width = 50 
        n_layers = 9
        window_height = np.int(np.floor(self.ysize/n_layers))
        margin = 100 # How much to slide left and right for searching
        
        left_centroids = []
        right_centroids = []
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        left_quarter = warped[int(self.ysize * 3/4):, :x_middle]
        middle_y = int(self.ysize - window_height/2)
        l_sum = np.sum(left_quarter, axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2
        
        l_centroid = (l_center, middle_y)
        
        right_quarter = warped[int(self.ysize * 3/4):, x_middle:]
        r_sum = np.sum(right_quarter, axis=0)
        r_center = np.argmax(np.convolve(window,r_sum)) - window_width/2 + x_middle
        r_centroid = (r_center, middle_y)
        
        # Add what we found for the first layer
        left_centroids.append(l_centroid)
        right_centroids.append(r_centroid)
        
        # Threshold for convolution to avoid empty space centroids
        conv_threshold = 100
        # Go through each layer looking for max pixel locations
        
        for level in range(1, n_layers):
            # convolve the window into the vertical slice of the image
            #top, middle and bottom lines of the window
            bottom_y = int(self.ysize - level * window_height)
            top_y = bottom_y - window_height
            middle_y = bottom_y - window_height/2
            layer = warped[top_y:bottom_y, :]
            layer_sum = np.sum(layer, axis=0)
            conv_signal = np.convolve(window, layer_sum)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center 
                                + offset 
                                - margin, 0))
            l_max_index = int(min(l_center
                                + offset
                                + margin, self.xsize))
            if np.max(conv_signal[ l_min_index : l_max_index]) > conv_threshold:
                l_center = np.argmax(conv_signal[ l_min_index : l_max_index]) + l_min_index - offset
                l_centroid = (l_center, middle_y)
                left_centroids.append(l_centroid)
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center
                                +offset
                                -margin, 0))
            r_max_index = int(min(r_center
                                +offset
                                +margin, self.xsize))
            if np.max(conv_signal[r_min_index : r_max_index]) > conv_threshold:
                r_center = np.argmax(conv_signal[r_min_index : r_max_index]) + r_min_index-offset
                r_centroid = (r_center, middle_y)
                right_centroids.append(r_centroid)
        
    
        left_centroids = np.array(left_centroids)
        right_centroids = np.array(right_centroids)
        left_fit = np.polyfit(left_centroids[:, 1], left_centroids[:, 0], 2)
        right_fit = np.polyfit(right_centroids[:, 1], right_centroids[:, 0], 2)
        return left_fit, right_fit, left_centroids, right_centroids
    
    def quadratic(self, fit):
        y = self.ploty
        a = fit[0]
        b = fit[1]
        c = fit[2]
        out = a * np.square(y) + b * y + c
        return out

    def plot_all(self, warped, undist, birds_eye, left_fitx, right_fitx, left_centroids, right_centroids):
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 255))
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.xsize, self.ysize))
        
        mini_x = self.xsize//3
        mini_y = self.ysize//3
        mini_be = cv2.resize(birds_eye, (mini_x, mini_y), interpolation=cv2.INTER_AREA)
        
        mini_fit = np.copy(warped)
        mini_fit[mini_fit == 1] = 255
        mini_fit = np.dstack((mini_fit, mini_fit, mini_fit))

        for y in range(self.ysize):
            mini_fit = cv2.circle(mini_fit, (left_fitx[y], y), 3, (0, 255, 0 ), -1)
            mini_fit = cv2.circle(mini_fit, (right_fitx[y], y), 3, (0, 255, 0 ), -1)
        for point in left_centroids:
            mini_fit = cv2.circle(mini_fit, tuple(np.int32(point)), 15, (255, 0,0 ), -1)
        for point in right_centroids:
            mini_fit = cv2.circle(mini_fit, tuple(np.int32(point)), 15, (255, 0,0 ), -1)
        
        mini_fit = cv2.resize(mini_fit, (mini_x, mini_y), interpolation=cv2.INTER_AREA)

        undist[0:mini_y, 0:mini_x, :] = mini_be
        undist[0:mini_y, mini_x:2*mini_x] = mini_fit
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result
    
    def sanity(self, img, left_fit, right_fit, left_fitx, right_fitx, left_centroids, right_centroids):
        
        left_diff = self.left.diff(left_fitx, self.ploty)
        right_diff = self.right.diff(right_fitx, self.ploty)
                
        cent_thresh = 4
        n_left_cent = len(left_centroids)
        n_right_cent = len(right_centroids)
        if (n_left_cent < cent_thresh | n_right_cent < cent_thresh):
            file_name = "fail_{:04d}.jpg".format(self.counter)
            bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('failure_imgs/' + file_name, bgr)
        
        self.log.append((self.counter, left_diff, right_diff, 
        left_fit[0], left_fit[1], left_fit[2], 
        right_fit[0], right_fit[1], right_fit[2],
        n_left_cent, n_right_cent))
        
        
    def process(self, img):
        self.xsize = img.shape[1]
        self.ysize = img.shape[0]
        self.counter += 1
        warped, undist, birds_eye = self.pipe(img)
        left_fit, right_fit, left_centroids, right_centroids = self.slide(warped)
        
        if self.ploty is None:
            self.ploty = np.linspace(0, self.ysize-1, self.ysize )
        
        left_fitx = self.quadratic(left_fit)
        right_fitx = self.quadratic(right_fit)

        self.sanity(img, left_fit, right_fit, left_fitx, right_fitx, left_centroids, right_centroids)

        out = self.plot_all(warped, undist, birds_eye, self.left.mean_fit, self.right.mean_fit,
        left_centroids, right_centroids)
        curv_txt = "Curvature: {:6.0f}m".format(self.left.curvature)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, curv_txt, (10, 280), font, 1, (255, 255, 255), 2, cv2.LINE_AA )

        lane_center = left_fitx[-1] + (right_fitx[-1] - left_fitx[-1]) / 2
        car_center = np.int(self.xsize/2)
        deviation = (car_center - lane_center) * self.left.xm_per_pix
        dev_txt = "Deviation from lane center: {:4.2f}".format(deviation)
        cv2.putText(out, dev_txt, (int(self.xsize/2), 280), font, 1, (255, 255, 255), 2, cv2.LINE_AA )
        return out
       