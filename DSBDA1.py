import pandas as pd

# Step 1: Import all the required Python Libraries.

# Step 2: Load the Dataset into pandas dataframe.
local_path = '/content/laptop_pricing_dataset.csv'
df = pd.read_csv(local_path)

# Step 3: Data Preprocessing.

# Check for missing values in the data.
missing_values = df.isnull().sum()

# Get some initial statistics.
summary_statistics = df.describe()

# Check the dimensions of the data frame.
data_dimensions = df.shape

# Step 4: Data Formatting and Data Normalization.

# Summarize the types of variables by checking the data types.
variable_types = df.dtypes

# Print variable descriptions and types.
print("\nVariable Types:\n", variable_types)

# Step 5: Convert categorical variables into quantitative variables in Python.
# This step is not explicitly performed in this code snippet, but it can be done using techniques such as one-hot encoding or label encoding.

# Step 6: Additional Data Normalization or Processing.

# Fill missing values using the last observation carried forward (LCOF) method.
lcof = df.ffill()

# Fill missing values using the next observation carried backward (NOCB) method.
nocb = df.bfill()

# Display the results.
print("\nMissing Values:\n", missing_values)
print("\nSummary Statistics:\n", summary_statistics)
print("\nData Dimensions:\n", data_dimensions)
print("\nVariable Types:\n", variable_types)
print("\n LCOF \n", lcof)
print("\n NOCB \n", nocb)
