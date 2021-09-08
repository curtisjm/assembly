from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, neighbors
import numpy as np
import pandas as pd

df = pd.read_csv("data/breast-cancer-wisconsin.data")
# told that there were ? unknown values, so we should replace them
df.replace("?", -99999, inplace=True)
df.drop(columns="id", axis=1, inplace=True)

# features
X = np.array(df.drop(columns="class", axis=1))
# labels
y = np.array(df["class"])

# shuffle and separate data into training and testing chunks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])

prediction = clf.predict(example_measures)
print(prediction)
