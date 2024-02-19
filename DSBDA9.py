import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

# Create box plot
sns.boxplot(
    x = 'sex',
    y = 'age',
    hue = 'survived',
    data = titanic
)

plt.show()