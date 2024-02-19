import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('Student_Marks.csv')

# Step 1: Handling missing values and inconsistencies
# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Deal with missing values using suitable techniques (e.g., fill with mean/median, forward fill)
df.fillna(df.mean(), inplace=True)

# Step 2: Handling outliers in numeric variables using the IQR method
# Identify numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns

# Loop through numeric columns and handle outliers using the IQR method
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with NaN
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), np.nan, df[column])

# Fill missing values with the mean or median of the respective columns
df.fillna(df.mean(), inplace=True)

# Step 3: Apply Min-Max Scaling (Normalization) on the 'Marks' variable
# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the data and transform the 'Marks' column
df['Marks_scaled'] = scaler.fit_transform(df[['Marks']])

# Display the DataFrame before and after transformation
print("DataFrame Before Transformation:")
print(df.head())

print("\nDataFrame After Transformation:")
print(df.head())
