import pandas as pd

df = pd.read_csv('Iris.csv')

grouped = df.groupby('Species')

mean_values = grouped.mean()
median_values = grouped.median()
std_dev_values = grouped.std()
min_values = grouped.min()
max_values = grouped.max()
percentiles = grouped.quantile([0.25, 0.50, 0.75])

print("Mean Values:")
print(mean_values)

print("\nMedian Values:")
print(median_values)

print("\nStandard Deviation Values:")
print(std_dev_values)

print("\nMinimum Values:")
print(min_values)

print("\nMaximum Values:")
print(max_values)

print("\nPercentiles:")
print(percentiles)
