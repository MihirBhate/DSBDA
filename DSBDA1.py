{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOyZusHNzHf7f9dY3EQhYyL",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MihirBhate/DSBDA/blob/DSBDA/DSBDA1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jxScjbFHynJU",
        "outputId": "02ba99fd-1105-4c20-bb4c-4bca1eedc288"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Variable Types:\n",
            " Unnamed: 0          int64\n",
            "Manufacturer       object\n",
            "Category            int64\n",
            "Screen             object\n",
            "GPU                 int64\n",
            "OS                  int64\n",
            "CPU_core            int64\n",
            "Screen_Size_cm    float64\n",
            "CPU_frequency     float64\n",
            "RAM_GB              int64\n",
            "Storage_GB_SSD      int64\n",
            "Weight_kg         float64\n",
            "Price               int64\n",
            "dtype: object\n",
            "\n",
            "Missing Values:\n",
            " Unnamed: 0        0\n",
            "Manufacturer      0\n",
            "Category          0\n",
            "Screen            0\n",
            "GPU               0\n",
            "OS                0\n",
            "CPU_core          0\n",
            "Screen_Size_cm    4\n",
            "CPU_frequency     0\n",
            "RAM_GB            0\n",
            "Storage_GB_SSD    0\n",
            "Weight_kg         5\n",
            "Price             0\n",
            "dtype: int64\n",
            "\n",
            "Summary Statistics:\n",
            "        Unnamed: 0    Category         GPU          OS    CPU_core  \\\n",
            "count  238.000000  238.000000  238.000000  238.000000  238.000000   \n",
            "mean   118.500000    3.205882    2.151261    1.058824    5.630252   \n",
            "std     68.848868    0.776533    0.638282    0.235790    1.241787   \n",
            "min      0.000000    1.000000    1.000000    1.000000    3.000000   \n",
            "25%     59.250000    3.000000    2.000000    1.000000    5.000000   \n",
            "50%    118.500000    3.000000    2.000000    1.000000    5.000000   \n",
            "75%    177.750000    4.000000    3.000000    1.000000    7.000000   \n",
            "max    237.000000    5.000000    3.000000    2.000000    7.000000   \n",
            "\n",
            "       Screen_Size_cm  CPU_frequency      RAM_GB  Storage_GB_SSD   Weight_kg  \\\n",
            "count      234.000000     238.000000  238.000000      238.000000  233.000000   \n",
            "mean        37.269615       2.360084    7.882353      245.781513    1.862232   \n",
            "std          2.971365       0.411393    2.482603       34.765316    0.494332   \n",
            "min         30.480000       1.200000    4.000000      128.000000    0.810000   \n",
            "25%         35.560000       2.000000    8.000000      256.000000    1.440000   \n",
            "50%         38.100000       2.500000    8.000000      256.000000    1.870000   \n",
            "75%         39.624000       2.700000    8.000000      256.000000    2.200000   \n",
            "max         43.942000       2.900000   16.000000      256.000000    3.600000   \n",
            "\n",
            "             Price  \n",
            "count   238.000000  \n",
            "mean   1462.344538  \n",
            "std     574.607699  \n",
            "min     527.000000  \n",
            "25%    1066.500000  \n",
            "50%    1333.000000  \n",
            "75%    1777.000000  \n",
            "max    3810.000000  \n",
            "\n",
            "Data Dimensions:\n",
            " (238, 13)\n",
            "\n",
            "Variable Types:\n",
            " Unnamed: 0          int64\n",
            "Manufacturer       object\n",
            "Category            int64\n",
            "Screen             object\n",
            "GPU                 int64\n",
            "OS                  int64\n",
            "CPU_core            int64\n",
            "Screen_Size_cm    float64\n",
            "CPU_frequency     float64\n",
            "RAM_GB              int64\n",
            "Storage_GB_SSD      int64\n",
            "Weight_kg         float64\n",
            "Price               int64\n",
            "dtype: object\n",
            "\n",
            " LCOF \n",
            "      Unnamed: 0 Manufacturer  Category     Screen  GPU  OS  CPU_core  \\\n",
            "0             0         Acer         4  IPS Panel    2   1         5   \n",
            "1             1         Dell         3    Full HD    1   1         3   \n",
            "2             2         Dell         3    Full HD    1   1         7   \n",
            "3             3         Dell         4  IPS Panel    2   1         5   \n",
            "4             4           HP         4    Full HD    2   1         7   \n",
            "..          ...          ...       ...        ...  ...  ..       ...   \n",
            "233         233       Lenovo         4  IPS Panel    2   1         7   \n",
            "234         234      Toshiba         3    Full HD    2   1         5   \n",
            "235         235       Lenovo         4  IPS Panel    2   1         5   \n",
            "236         236       Lenovo         3    Full HD    3   1         5   \n",
            "237         237      Toshiba         3    Full HD    2   1         5   \n",
            "\n",
            "     Screen_Size_cm  CPU_frequency  RAM_GB  Storage_GB_SSD  Weight_kg  Price  \n",
            "0            35.560            1.6       8             256       1.60    978  \n",
            "1            39.624            2.0       4             256       2.20    634  \n",
            "2            39.624            2.7       8             256       2.20    946  \n",
            "3            33.782            1.6       8             128       1.22   1244  \n",
            "4            39.624            1.8       8             256       1.91    837  \n",
            "..              ...            ...     ...             ...        ...    ...  \n",
            "233          35.560            2.6       8             256       1.70   1891  \n",
            "234          33.782            2.4       8             256       1.20   1950  \n",
            "235          30.480            2.6       8             256       1.36   2236  \n",
            "236          39.624            2.5       6             256       2.40    883  \n",
            "237          35.560            2.3       8             256       1.95   1499  \n",
            "\n",
            "[238 rows x 13 columns]\n",
            "\n",
            " NOCB \n",
            "      Unnamed: 0 Manufacturer  Category     Screen  GPU  OS  CPU_core  \\\n",
            "0             0         Acer         4  IPS Panel    2   1         5   \n",
            "1             1         Dell         3    Full HD    1   1         3   \n",
            "2             2         Dell         3    Full HD    1   1         7   \n",
            "3             3         Dell         4  IPS Panel    2   1         5   \n",
            "4             4           HP         4    Full HD    2   1         7   \n",
            "..          ...          ...       ...        ...  ...  ..       ...   \n",
            "233         233       Lenovo         4  IPS Panel    2   1         7   \n",
            "234         234      Toshiba         3    Full HD    2   1         5   \n",
            "235         235       Lenovo         4  IPS Panel    2   1         5   \n",
            "236         236       Lenovo         3    Full HD    3   1         5   \n",
            "237         237      Toshiba         3    Full HD    2   1         5   \n",
            "\n",
            "     Screen_Size_cm  CPU_frequency  RAM_GB  Storage_GB_SSD  Weight_kg  Price  \n",
            "0            35.560            1.6       8             256       1.60    978  \n",
            "1            39.624            2.0       4             256       2.20    634  \n",
            "2            39.624            2.7       8             256       2.20    946  \n",
            "3            33.782            1.6       8             128       1.22   1244  \n",
            "4            39.624            1.8       8             256       1.91    837  \n",
            "..              ...            ...     ...             ...        ...    ...  \n",
            "233          35.560            2.6       8             256       1.70   1891  \n",
            "234          33.782            2.4       8             256       1.20   1950  \n",
            "235          30.480            2.6       8             256       1.36   2236  \n",
            "236          39.624            2.5       6             256       2.40    883  \n",
            "237          35.560            2.3       8             256       1.95   1499  \n",
            "\n",
            "[238 rows x 13 columns]\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Step 1: Import all the required Python Libraries.\n",
        "\n",
        "# Step 2: Load the Dataset into pandas dataframe.\n",
        "local_path = '/content/laptop_pricing_dataset.csv'\n",
        "df = pd.read_csv(local_path)\n",
        "\n",
        "# Step 3: Data Preprocessing.\n",
        "\n",
        "# Check for missing values in the data.\n",
        "missing_values = df.isnull().sum()\n",
        "\n",
        "# Get some initial statistics.\n",
        "summary_statistics = df.describe()\n",
        "\n",
        "# Check the dimensions of the data frame.\n",
        "data_dimensions = df.shape\n",
        "\n",
        "# Step 4: Data Formatting and Data Normalization.\n",
        "\n",
        "# Summarize the types of variables by checking the data types.\n",
        "variable_types = df.dtypes\n",
        "\n",
        "# Print variable descriptions and types.\n",
        "print(\"\\nVariable Types:\\n\", variable_types)\n",
        "\n",
        "# Step 5: Convert categorical variables into quantitative variables in Python.\n",
        "# This step is not explicitly performed in this code snippet, but it can be done using techniques such as one-hot encoding or label encoding.\n",
        "\n",
        "# Step 6: Additional Data Normalization or Processing.\n",
        "\n",
        "# Fill missing values using the last observation carried forward (LCOF) method.\n",
        "lcof = df.ffill()\n",
        "\n",
        "# Fill missing values using the next observation carried backward (NOCB) method.\n",
        "nocb = df.bfill()\n",
        "\n",
        "# Display the results.\n",
        "print(\"\\nMissing Values:\\n\", missing_values)\n",
        "print(\"\\nSummary Statistics:\\n\", summary_statistics)\n",
        "print(\"\\nData Dimensions:\\n\", data_dimensions)\n",
        "print(\"\\nVariable Types:\\n\", variable_types)\n",
        "print(\"\\n LCOF \\n\", lcof)\n",
        "print(\"\\n NOCB \\n\", nocb)\n"
      ]
    }
  ]
}