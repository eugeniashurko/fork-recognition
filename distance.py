""""Classifies images from an already learned database"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from skimage.io import imshow
from skimage.io import imread

from library.data_utils import load_labels
from library.feature_extraction import extract_features
from library.feature_extraction import preprocess_image

from sys import argv

if __name__ == '__main__':

    # Process image
    if len(argv) < 3:
        print("Error: less than 2 pgm files given")
        exit()
    else:
        image1 = imread(argv[1], as_grey=True)
        p_image1 = preprocess_image(image1)
        features1 = extract_features(p_image1)
        
        image2 = imread(argv[2], as_grey=True)
        p_image2 = preprocess_image(image2)
        features2 = extract_features(p_image2)

        f = features1 - features2

        s = 0
        for i in f:
            s += i*i
        print(s)
    

    
