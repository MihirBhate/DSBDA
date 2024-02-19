import pandas as pd
url = "https://raw.githubusercontent.com/StillWork/data/master/titanic_train.csv"
df = pd.read_csv(url)

import seaborn as sns
import matplotlib.pyplot as plt

# Load the titanic dataset
titanic = sns.load_dataset('titanic')

# distribution of age
sns.distplot(titanic['age'].dropna())
plt.show()

# relationship between sex and survival
sns.barplot(x='sex', y='survived', data=titanic)
plt.show()

# relationship between class and survival
sns.barplot(x='class', y='survived', data=titanic)
plt.show()

# relationship between age, sex, and survival
sns.pointplot(x='age', y='survived', hue='sex', data=titanic)
plt.show()

# relationship between class, sex, and survival
sns.pointplot(x='class', y='survived', hue='sex', data=titanic)
plt.show()

import matplotlib.pyplot as plt

fares = titanic["fare"]

# Create a histogram with 20 bins
plt.hist(fares, bins=20)

#labels
plt.xlabel("Fare")
plt.ylabel("Number of passengers")
plt.title("Distribution of ticket prices")

plt.show()
