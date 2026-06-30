# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

# Load dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------
# HANDLE MISSING VALUES
# -------------------------

# Fill Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -------------------------
# ENCODE CATEGORICAL DATA
# -------------------------

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# -------------------------
# STANDARDIZATION
# -------------------------

scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(
    df[['Age', 'Fare']]
)

# -------------------------
# CREATE BOXPLOT
# -------------------------

plt.figure(figsize=(8,5))
sns.boxplot(y=df['Fare'])
plt.title("Fare Outliers")

# Save figure automatically
plt.savefig('boxplot.png')

# Close figure automatically
plt.close()

print("\nBoxplot saved as boxplot.png")

# -------------------------
# REMOVE OUTLIERS
# -------------------------

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df['Fare'] >= lower) &
    (df['Fare'] <= upper)
]

print("\nDataset after removing outliers:")
print(df.head())

# -------------------------
# SAVE CLEANED DATA
# -------------------------

df.to_csv('cleaned_Titanic-Dataset.csv', index=False)

print("\ncleaned_Titanic-Dataset.csv created successfully!")
print("boxplot.png created successfully!")
print("\nTASK COMPLETED SUCCESSFULLY!")
