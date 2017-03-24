
# **Project: Vehicle Identification** 

---

 [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
 
 
 ## 1. Introduction
 
 The goal of this project is to develop classifier, which will be able to detect vehicles while driving using images from a dash cam. The vehicles should be identified correctly and performance should be sufficent to allow for  5-6 fps processing.
 
 One important assumption is the choice of algorithms used to identify cars on the road - in this project we use Computer Vision, especially Histogram of Oriented Gradients together with SVM based classification (SVC).
 
 This report is divided into following sections:
 * SVC model training
 * Features engineering
 * Image analysis
 * Video pipeline
 * Summary
 
 
 ## 2. SVC model training
 ### 2.1 Training data
 I decided to use only the datasets provided for this project - did not use the recently released dataset from Udacity.
 
 For each image, I extracted following features (see get_features in hog_utils.py for details):
 * color histogram
 * spatial features
 * HOG descriptor
 Those features will be used to train SVC classified
 
 ### 2.2 Train and test split
 I decided to have 20% of data in the test set. In case of GTI data, there was a problem pointed out in the course materials - some of the pictures were taken in series - there was the same car on them and differences were minimal. This means that in case of spreading such pictures between train and test set, they could disturb training process - classifier will "memorize" them and show better accuracy than with real data.

 To deal with this issue, I manually splitted the GTI sets - for each set i found a picture, which splitted the set with 80/20 rule but without spreading the same pictures across train and test set - see model_check in hog_utils.py for more details.

All test and train data are scaled and shuffled before training

### 2.3 Training

 The SVC classifier is trained using default parameters and I calculated accuracy using test set. This allows me to comapre different models and select the one with best accuracy. An assumption about choosing accuracy as the metric for selecting model should be discussed in the future taking into account the requirements for the classifier.
 
 The classifier is trained using images with size 64x64 pixels, this means that any prediction must be made on image of the same size.
 ## 3 Feature engineering
 each feature has its parameters, which can impact the classifier accuracy, additionally selecting wrong parameters can result in bigger efatures and therefore longer processing time. To select best parameters I did the following:
* color histogram: I experimented manually setting different number of bins and decided to go with 32 bins
* spatial features  - I used similar procedure as in case of color histogram
* HOG has 3 omportant parameters and tuning them manually was too hard, especially with importance of color model selection. Therefore I decided to perform parameter searching.
### 3.1 Searching of parameter space
I made cartesian product of the possible parameter values and for each combination trained a calssifier, calculated its accuracy and features size. Trained classifier and scaler plus parameters used are stored together in a pickle for future use. Parameter values:
* HOG pixels per cell: 8, 12, 16
* HOG cells per block: 2, 3, 4
* HOG orientations: 6, 8, 9, 10, 12
* Color model: HLS and YCbCr - I made preselection using exploratory analysis

Above values gave me 90 combinations to check, here come the top 10 results:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>pix_per_cell</th>
      <th>cells_per_block</th>
      <th>orientations</th>
      <th>size</th>
      <th>color_model</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8</td>
      <td>3</td>
      <td>9</td>
      <td>9612</td>
      <td>HLS</td>
      <td>0.991019</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2</td>
      <td>12</td>
      <td>7920</td>
      <td>YCbCr</td>
      <td>0.990457</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2</td>
      <td>10</td>
      <td>6744</td>
      <td>YCbCr</td>
      <td>0.989896</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2</td>
      <td>12</td>
      <td>7920</td>
      <td>HLS</td>
      <td>0.989615</td>
    </tr>
    <tr>
      <td>16</td>
      <td>2</td>
      <td>12</td>
      <td>2160</td>
      <td>HLS</td>
      <td>0.989054</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2</td>
      <td>10</td>
      <td>6744</td>
      <td>HLS</td>
      <td>0.989054</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3</td>
      <td>12</td>
      <td>12528</td>
      <td>HLS</td>
      <td>0.989054</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3</td>
      <td>12</td>
      <td>12528</td>
      <td>YCbCr</td>
      <td>0.988774</td>
    </tr>
    <tr>
      <td>16</td>
      <td>3</td>
      <td>12</td>
      <td>2160</td>
      <td>HLS</td>
      <td>0.988212</td>
    </tr>
  </tbody>
</table>


## 4. Image processing
### 4.1 Sliding window
I used slided window approach as described in project lectures. I developed a class named slider, which does the sliding window scan and makes predictions based on provided classifier. Slider class is initialized with:
* image to be processed
* area of interest - where to search for vehicles - these values were found by trial and error
* model - classifier, scaler plus parameters for features like HOG etc found during feature engineering
* step for sliding window found by trial and error
* scale: how to scale image before processing
The scale is used to make image smaller while keeping the same scan window size (64x64). This is similar approach to image pyramid method described by Adrian Rosenbrock, however I decided to use 2 sliders:
* 1:1 detecting objects with size 64x64,
* 1:2 detecting objects with size 128x128
An example of output is shown below, blue boxes are results of 128x128 scan.


![png](output_images/output_5_1.png)


### 4.2 Heat maps

To deal with false positives, I implemented heatmap as described in lectures. For each images, I created a heatmpap containing points returned by the classifier, then apply threshold found by trual and error and perform labeling as described in the lectures. A bouding box is then drawn around found cars.

Result for test images is shown below:

![png](output_images/output_7_0.png)



![png](output_images/output_7_1.png)



![png](output_images/output_7_2.png)



![png](output_images/output_7_3.png)



![png](output_images/output_7_4.png)



![png](output_images/output_7_5.png)


## 5. Video pipeline

To process video file, I applied the all elements described in chapter "Image processing"  to each frame. Additionally I also run the lane finding code from Project 4 and draw its output.

Result is shown below, click on it for full version.

[![Result](result.gif)](result.mp4)

## 6. Summary and discussion
### 6.1 Classifier
My model reports very interesting false positives - i.e finds cars in the trees. Selecting areaof interest removes those false positives, however this signals a room for improvement here.


![png](output_images/output_10_1.png)


The classifier shoud be retrained with more samples and also a hard-negative mining could be used to avoid false posives as described by Adrian Rosebrock [here](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/). Also some SVC specific tuning should be performed to ensure best results.

### 6.2 Parameters

The full video shows that a tradeoff between processing time and accuracy is far from being perfect. Especially number of sliding windows and area of interest should be reviewed to better detect vehicles coming into picture. Method of grouping multiple detections into one should be also changed to either [deformable parts model by Pedro F. Felzenszwalb et al](http://cs.brown.edu/~pff/papers/lsvm-pami.pdf) or [Exemplar SVMs by Tomasz Malisiewicz](http://cs.brown.edu/~pff/papers/lsvm-pami.pdf)

One of main outcomes of this project is its sensitiviness to parameter tuning. While I performed basic feature engineering, it should be repated to confirm my findings. The resulting parameter set should be atradeoff between accuracy and features size, because it impact the processing time.

### 6.3 Performance

My code is terribly slow - on Intel Core i5-5300  @ 2.30 GHz it processes one frame in 10 seconds, this is far below expectations of doing realtime identification. There are different ways of dealing with this problem:
* the calculations are done on 1 core, but switching to multicore algorithm should be considered as last resort options, because the code can be optimized
* HOG calculations are responsible for most of the time used for processing, so reducing this should be the first goal. There will be siginificant improvement in case of solution based on HOG subsampling or more general using Piotr Dollar's [paper](https://pdollar.github.io/files/papers/DollarBMVC10FPDW.pdf) 

### 6.4 Final thoughts
Car detection using Computer Vision and models like SVM requires a lot of feature engineering and model tuning, however final results should be good enough to contribute to the Self Driving Car systems responsible for environment mapping. There are other technologies available like U-net and others and final choice should be made on the tradeoff between accuracy (or other metrics) and computational power required for processing.