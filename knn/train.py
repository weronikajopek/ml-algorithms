import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

# defining colors corresponding to the 3 classes in the iris dataset
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# loadint the dataser
iris = datasets.load_iris()
X, y = iris.data, iris.target

# splitting the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# training the classifier
clf = KNN(k=5)
clf.fit(X_train, y_train)

# making predictions
predictions = clf.predict(X_test)
print(predictions)

# computing accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)
