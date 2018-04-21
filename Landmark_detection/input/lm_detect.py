"""
Kaggle Competition: Humpback Whale Identification Challenge

Author: Karthik Raj Ramachandrappa
Date : 21 April 2018

"""

import os
import time
import math

import numpy as np
import pandas as pd
import cv2


import matplotlib.image as mpimg

from sklearn.decomposition.pca import PCA
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline

import warnings

start_time = time.clock()


#8381 of the images are RGB, so they will need to be converted to greyscale.
def rgb_to_gray(img):
    if len(img.shape) == 2: #already gray
        return(img)
    grayImage = np.zeros(img.shape)
    #These scaling coefficients are evidently standard for RGB->Grayscale conversion
    R = img[:,:,0]*0.299
    G = img[:,:,1]*0.587
    B = img[:,:,2]*0.114
    
    grayImage = R + G + B
    
    return(grayImage)
    
#We will also need to group together nearby pixels in order to 
#reduce image complexity and uniformize image size (so that every image has
#the same number of features).
    
def img_compress(img, x_bins=100,y_bins=100):
    x_splits = np.linspace(0,img.shape[1]-1,x_bins+1, dtype = int)
    y_splits=  np.linspace(0,img.shape[0]-1,y_bins+1, dtype = int)
    
    compressed = np.zeros((y_bins,x_bins))
    
    for i in range(y_bins):
        for j in range(x_bins):
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=RuntimeWarning)
                 temp = np.mean(img[y_splits[i]:y_splits[i+1],
                                      x_splits[j]:x_splits[j+1]])
            if math.isnan(temp):
                if y_splits[i]==y_splits[i+1]:
                    compressed[i,j]=compressed[i-1,j]
                else:
                    compressed[i,j] = compressed[i,j-1]
            else:
                compressed[i,j] = int(temp)
    return(compressed)


train_dir = '../input/train'
test_dir = '../input/test'
#Convert .jpg images into pixel arrays
#imgs_train = [rgb_to_gray(mpimg.imread(train_dir + '/' + file, format = 'JPG')) for file in os.listdir(train_dir)]
imgs_train_1 = [cv2.imread(train_dir + '/' + file) for file in os.listdir(train_dir)]
imgs_train = [cv2.cvtColor( img, cv2.COLOR_RGB2GRAY ) for img in imgs_train_1]
#rows_train = [img.shape[0] for img in imgs_train]
#cols_train = [img.shape[1] for img in imgs_train]
print("Training images read.", "Time:", time.clock()-start_time)


min_cols = 1024
min_rows = 683



#Get indices of those pictures which aren't too small
good_pics = [i for i in range(len(imgs_train)) if (imgs_train[i].shape[0]>=min_rows)and(imgs_train[i].shape[1]>=min_cols)]
#Select only pictures that are sufficiently large
imgs_train = [imgs_train[i] for i in good_pics]


#Compress images. Unfortunately we have to distort the aspect ratio, which may
#lose valuable information. But if we do not do this then features will not
#have consistent meaning across pictures, even if we ensure that all pictures
#get compressed to the same resolution (pixel count)

#We need to find the smallest dimension across all images (train and test) in
#order to properly compress; otherwise the compressing algorithm will generate
#NaN-values.

compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]
print("Training images compressed.", "Time:", time.clock()-start_time)
        
del imgs_train



#Extract filenames to later associate to whale IDs
filenames = [file for file in os.listdir(train_dir)]
filenames = [filenames[i] for i in good_pics]
filenames_test = [file for file in os.listdir(test_dir)]
#Convert images into pandas dataframe format for learning the model
df1 = pd.DataFrame(data = [img.ravel() for img in compressed_train_imgs])
df2 = pd.DataFrame(data = filenames, columns = ['FileName'])
data1 = pd.concat([df2,df1],axis = 1)
#Obtain filenames, indexed by whale IDs
data2 = pd.read_csv('./train.csv',names = ['FileName','Landmark'])

#Map whale IDs to images via filenames (index of data1 is matched to value of data2)
data = data2.merge(data1, on = 'FileName')
data = data.drop('FileName',axis = 1)

del compressed_train_imgs, df1, df2, data1, data2

#Extract training examples and labels, and get test set.
X_train = data.iloc[:,1:]
y_train = data['Landmark']


#Perform principal component analysis for dimensionality reduction
#Try playing around with n_components to get better scores

pca = PCA(random_state = 42, n_components = 100, whiten = True)
pca.fit(X_train)
print("PCA fitting complete", "Time:", time.clock()-start_time)

#Try other classifiers for better scores.
logreg = LogisticRegression(C = 1e-2)

#Define our pipeline. First we transform the data into its principal components,
#then learn the logistic regression model
clf = Pipeline([('pca',pca),
                ('logreg',logreg),
                ])

#Fit model
clf.fit(X_train,y_train)
print('Estimator fitting complete.', "Time:", time.clock()-start_time)
#Score model on training data
print("Score on training set:", clf.score(X_train,y_train))
#We no longer need the training data. Delete to free up memory.
del data, X_train, y_train

#Read and clean test set
#imgs_test = [rgb_to_gray(mpimg.imread(test_dir + '/' + file)) for file in os.listdir(test_dir)]
imgs_test_1 = [cv2.imread(test_dir + '/' + file) for file in os.listdir(test_dir)]
imgs_test = [cv2.cvtColor( img, cv2.COLOR_RGB2GRAY ) for img in imgs_test_1]

print("Test images read.", "Time:", time.clock()-start_time)
#Smallest number of columns and rows across all test pictures
compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]
print("Test images compressed.", "Time:", time.clock()-start_time)
del imgs_test
X_test = pd.DataFrame([img.ravel() for img in compressed_test_imgs])
del compressed_test_imgs

#Extract the top 5 predictions for each test example

y_preds = clf.predict_proba(X_test)
print("Predictions made on test set.", "Time:", time.clock()-start_time)


#Extract top 5 results
results = pd.DataFrame(data = [clf.classes_[np.argsort(y_preds[i,:])[-5:]] for i in range(y_preds.shape[0])],
                       index = filenames_test)
def list_to_str(L):
    string = ""
    for word in L:
        string =  word + " "         
    return(string)
        
results2 = pd.DataFrame(data = [list_to_str(results.iloc[i].values) for i in range(results.shape[0])],
                        index = filenames_test,columns = ['Landmark'])
 
results2.to_csv('results.csv', sep = ',', index_label = 'Image', header = True)

print("Done.", "Time:", time.clock()-start_time)


