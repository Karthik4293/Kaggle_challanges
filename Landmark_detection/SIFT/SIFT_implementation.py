# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy
from scipy.misc import imread
import _pickle as pickle
import random
import os
import matplotlib.pyplot as plt
from collections import Counter


# Feature extractor
def extract_features(image_path, vector_size=64, flag=0):
    image = imread(image_path, mode="RGB")
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    try:
        alg = cv2.KAZE_create()
        # Finding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
     #   print(kps, dsc)
        if (flag==1):
            img=cv2.drawKeypoints(gray,kps,image,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('sift_keypoints.jpg',img)
            cv2.imshow("sift_keypoints.jpg",img)
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error:', e)
        return None

    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f, flag = 0)
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)


class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path,'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract_features(image_path, flag = 1)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()

def show_img(path):
    img = imread(path, mode="RGB")
    plt.imshow(img)
    plt.show()
    
def run():
    images_path = '../resources/images/'
    test_images = '../resources/test/'
    namespace = {'trevi': 'Trevi Fountain  Rome,Italy', 'liberty': 'Statue of Liberty   New York,USA', 'colosseum':'Colosseum   Rome,Italy'}
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    test_files = [os.path.join(test_images, p) for p in sorted(os.listdir(test_images))]
    # getting 3 random images 
    sample = random.sample(test_files, 3)
    batch_extractor(images_path)
    #print(result)
    ma = Matcher('features.pck')
    final_res = []
    
    for s in sample:
        print('Query image ==========================================')
        show_img(s)
        names, match = ma.match(s, topn=3)
        new_names = []      
        for na in names:
            new_name =''
            j = 0
            while na[j].isalpha():
                new_name = new_name + na[j]
                j += 1
            new_names.append(new_name)
            
        data = Counter(new_names)
        res = max(new_names, key=data.get)
        print('Result images ========================================')
        for i in range(3):
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            print('Match %.6s' % (1-match[i]))
            show_img(os.path.join(images_path, names[i]))
        print("The landmark being queried is " + namespace[res])
        final_res.append(res)
    print('\n')
    print('---------------------------------------------')
    k = 0
    for res in final_res:
        print("Query image ", k,': ', namespace[res])
        k += 1

run()
