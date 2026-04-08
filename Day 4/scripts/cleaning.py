import pandas as pd

# Load data
df = pd.read_csv("../data/data.csv")

#shape of Data(rows,columns)
print(df.shape)

#Basic Info
print("Dataset Info:\n")
print(df.info())

#Identify missing values
print("\nMissing Values:\n", df.isnull().sum())

#Check duplicate rows
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

#Remove duplicates (if any)
df = df.drop_duplicates()

#Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

#Fill missing Score with 0
df['Score'] = df['Score'].fillna(0)

#Data type check & correction
df['Age'] = df['Age'].astype(int)
df['Score'] = df['Score'].astype(float)

#Outlier check
print("\nStatistical Summary:\n", df.describe())

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

#Save cleaned data (IMPORTANT)
df.to_csv("cleaned_data.csv", index=False)

# Final Output
print("\nCleaned Data:\n", df)