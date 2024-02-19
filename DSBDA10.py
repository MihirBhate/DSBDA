from sklearn.datasets import load_iris

iris = load_iris()

import matplotlib.pyplot as plt

# Create a list of features
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Create a histogram for each feature
for feature in features:
  plt.hist(iris.data[:, features.index(feature)], bins=50)
  plt.xlabel(feature)
  plt.ylabel("Count")
  plt.title("Histogram of " + feature)
  plt.show()
  
  import matplotlib.pyplot as plt

# Create a list of features
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Create a boxplot for each feature
for feature in features:
  plt.boxplot(iris.data[:, features.index(feature)])
  plt.xlabel(feature)
  plt.ylabel("Value")
  plt.title("Boxplot of " + feature)
  plt.show()