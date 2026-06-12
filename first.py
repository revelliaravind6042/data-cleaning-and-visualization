import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Loading Dataset...\n")

df = pd.read_csv('StudentsPerformance.csv')
# DATASET INFORMATION
print("First 5 Rows:\n")
print(df.head())
print("\nDataset Information:\n")
print(df.info())
print("\nDataset Shape:")
print(df.shape)

# CHECKING MISSING VALUES
print("\nMissing Values:\n")
print(df.isnull().sum())

# Filling missing numerical values with mean
numerical_columns = ['math score', 'reading score', 'writing score']
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].mean())

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# CHECKING DUPLICATES
print("\nDuplicate Rows:", df.duplicated().sum())

# Removing duplicates
df = df.drop_duplicates()
print("Duplicate Rows After Cleaning:", df.duplicated().sum())

# OUTLIER DETECTION
print("\nChecking Outliers Using IQR Method...\n")
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col} -> Number of Outliers: {len(outliers)}")

# STATISTICAL SUMMARY
print("\nStatistical Summary:\n")
print(df.describe())

# CREATING VISUAL REPORTS / DASHBOARD
sns.set_style('whitegrid')
plt.figure(figsize=(18, 12))

# GRAPH 1 — Gender Distribution
plt.subplot(2, 2, 1)
sns.countplot(x='gender', data=df)
plt.title('Gender Distribution')

# GRAPH 2 — Average Scores by Gender
plt.subplot(2, 2, 2)
average_scores = df.groupby('gender')[[
    'math score',
    'reading score',
    'writing score'
]].mean()
average_scores.plot(kind='bar', ax=plt.gca())
plt.title('Average Scores by Gender')
plt.ylabel('Scores')

# GRAPH 3 — Reading vs Writing Scores
plt.subplot(2, 2, 3)
sns.scatterplot(
    x='reading score',
    y='writing score',
    hue='gender',
    data=df
)
plt.title('Reading vs Writing Scores')

# GRAPH 4 — Score Distribution
plt.subplot(2, 2, 4)
sns.histplot(df['math score'], bins=15, kde=True)
plt.title('Math Score Distribution')
plt.tight_layout()
plt.savefig('dashboard.png')
plt.show()

# HEATMAP (CORRELATION)
plt.figure(figsize=(8, 6))
correlation = df[[
    'math score',
    'reading score',
    'writing score'
]].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('heatmap.png')
plt.show()
# FINAL INSIGHTS
print("\nKEY FINDINGS:")
print("1. Dataset cleaned successfully.")
print("2. Missing values handled.")
print("3. Duplicate rows removed.")
print("4. Outliers analyzed using IQR method.")
print("5. Female and male student performance compared.")
print("6. Reading and writing scores show strong correlation.")
print("7. Dashboard-style visual reports generated.")

print("\nData Cleaning and Visualization Completed Successfully!")
