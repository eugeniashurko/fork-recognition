""""Classifies images using the QUICK dataset to train the model (features to be used mus be stored into files X,y)."""
import numpy as np
from skimage.io import imshow
from skimage.io import imread

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

from pickle import dump, load

from library.utils import medial_axis_skeleton
from library.utils import skeleton_lines

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale

from library.data_utils import load_labels
from library.feature_extraction import extract_quick_features
from library.feature_extraction import preprocess_image

from sys import argv


if __name__ == '__main__':
    labels = load_labels("database", "classes.csv")

    # load model
    X = np.array([])
    y = np.array(labels)
    X = load(open("X_quick", "rb"))
    y = load(open("y_quick", "rb"))


    # Check arguments
    if len(argv) < 2:
        print("Error: no pgm file given")
        exit()
    else:
        # Process image
        image = imread(argv[1], as_grey=True)
        p_image = preprocess_image(image)
        features = extract_quick_features(p_image)

        # resize the features to "normalize" them compared the feature dataset
        large_X = np.concatenate((X, [features]))
        large_X = scale(large_X)
        
        # re-extract now it's scaled
        features = large_X[-1]
        X = large_X[:-1]

        
        # Build the classifier from binary data loaded
        gamma = 0.1
        C = 10
        clf = OneVsRestClassifier(
            SVC(kernel='rbf', C=C, gamma=gamma, probability=True), n_jobs=4)
        clf.fit(X, y)

        # Classifies the image
        proba = clf.predict_proba([features])
        for p in proba[0]:
            print("%.13f" % p)
    

    
