""""Prints the distance detween two images, using our vector of features"""
from skimage.io import imshow
from skimage.io import imread

from library.data_utils import load_labels
from library.feature_extraction import extract_dist_features
from library.feature_extraction import preprocess_image

from sys import argv

if __name__ == '__main__':
    # Check arguents
    if len(argv) < 3:
        print("Error: less than 2 pgm files given")
        exit()
    else:
        # process image 1
        image1 = imread(argv[1], as_grey=True)
        p_image1 = preprocess_image(image1)
        features1 = extract_dist_features(p_image1)

        # process image 2
        image2 = imread(argv[2], as_grey=True)
        p_image2 = preprocess_image(image2)
        features2 = extract_dist_features(p_image2)


        # make the distance
        f = features1 - features2

        s = 0
        for i in f:
            s += i*i
        print(s)
    

    
