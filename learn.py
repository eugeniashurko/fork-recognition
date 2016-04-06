"""Performs the training of the model using some dataset and saves features in files X,y."""

import numpy as np
from pickle import dump, load

from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.preprocessing import scale

from library.data_utils import load_database
from library.feature_extraction import extract_features

def custom_test(clf, X, y):
    """Tests if the right class is in the top 10 classes."""
    score = 0.0
    proba = clf.predict_proba(X) # All the probas
    for i, sample_proba in enumerate(proba):
        classes = np.argsort(sample_proba)[::-1][:10] # top 10 higher probas
        labels = clf.classes_[classes]

        # is th right class in the top 10 ?
        if y[i] in labels:
            score += 1.0
    return score / proba.shape[0]

    
if __name__ == '__main__':
    images, labels = load_database("database", "classes.csv")
    X = []
    print("SLOW FEATURE EXTRACTION....")
    for i, image in enumerate(images):
         X.append(extract_features(image))

    # -------------
    # Machine learning
    # First, save features.
    X = np.array(X)
    y = np.array(labels)

    dump(X, open("X_quick", "wb"))
    dump(y, open("y_quick", "wb"))

    X = scale(X)

    print("DATA: %d x %d" % X.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)


    print("\n\nSVC RBF kernel:")
    print("++++++++++++")

    gamma = 0.1
    C = 10
    clf = OneVsRestClassifier(
        SVC(kernel='rbf', C=C, gamma=gamma, probability=True), n_jobs=4)

    print("SCORE:")
    score = cross_validation.cross_val_score(
        clf, X, y, cv=5)
    print(sum(score) / float(len(score)))
    clf.fit(X_train, y_train)

    print("TEACHERS TEST:")
    print(custom_test(clf, X_test, y_test))
