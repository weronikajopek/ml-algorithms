import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:

    # constructor
    def __init__(self, k=3):
        self.k = k

    # training the model
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # making predictions for multiple samples
    def predict(self, X):
        predictions = [self._predict(x) for x in X]  # taking a set of test samples X and calling _predict(x) for each sample to classify it
        return predictions

    # predicting a single sample
    def _predict(self, x):

        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]  # sorting distances and selecting the indices of the knn
        k_nearest_labels = [self.y_train[i] for i in k_indices] # retrieving the labels

        # majority vote - for classification tasks
        most_common = Counter(k_nearest_labels).most_common()   # counting how many times each classes appears in the knn
        return most_common[0][0]