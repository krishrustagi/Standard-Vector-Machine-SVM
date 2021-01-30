from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("Iris.csv")  # Iris Data
print(data)


labels = data["Species"]  # labels are species
print(labels)


# Plot of Sepal length vs Sepal Width
plt.scatter(data[data.Species == 'Iris-setosa']['SepalLengthCm'], data[data.Species ==
                                                                       'Iris-setosa']['SepalWidthCm'], label="Iris-setosa", color='red')
plt.scatter(data[data.Species == 'Iris-virginica']['SepalLengthCm'], data[data.Species ==
                                                                          'Iris-virginica']['SepalWidthCm'], label='Iris-virginica', color='green')
plt.scatter(data[data.Species == 'Iris-versicolor']['SepalLengthCm'], data[data.Species ==
                                                                           'Iris-versicolor']['SepalWidthCm'], label="Iris-versicolor", color='blue')
plt.show()


# Plot of Petal length vs Petal Width
plt.scatter(data[data.Species == 'Iris-setosa']['PetalLengthCm'], data[data.Species ==
                                                                       'Iris-setosa']['PetalWidthCm'], label="Iris-setosa", color='red')
plt.scatter(data[data.Species == 'Iris-virginica']['PetalLengthCm'], data[data.Species ==
                                                                          'Iris-virginica']['PetalWidthCm'], label='Iris-virginica', color='green')
plt.scatter(data[data.Species == 'Iris-versicolor']['PetalLengthCm'], data[data.Species ==
                                                                           'Iris-versicolor']['PetalWidthCm'], label="Iris-versicolor", color='blue')
plt.show()


features = data
features.drop(["Id"], inplace=True, axis=1)
features.drop(["Species"], inplace=True, axis=1)
print(features)  # features are sepals and petals


features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2)  # splitting the data


clf = SVC(kernel='rbf', class_weight='balanced', C=100.)  # model classifier
clf.fit(features_train, labels_train)  # training model


pred = clf.predict(features_test)  # predicting on features test


acc = accuracy_score(pred, labels_test)  # accuracy of model using SVC
print(acc)


# Now we can predict the type of iris based on properties of sepals and petals
# predictions can be made here ⬇
