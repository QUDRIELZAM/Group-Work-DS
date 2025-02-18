import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r'C:\Users\Msi\Desktop\R folder\Group Work\Titanic.csv')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handling Missing Values
# Since the dataset doesn't have 'Embarked' or 'Cabin', we only handle 'Age'
df['Age'].fillna(df['Age'].median(), inplace=True)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Convert Data Types
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

# Detect Outliers
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()

# Remove Outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~(df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR))]

# Min-Max Normalization
df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())

# Splitting Dataset into Train (80%), Test (10%), and Validation (10%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Summary statistics
print(df.describe())

# Count of survivors
print(df['Survived'].value_counts())

# Count of passengers by class
print(df['Pclass'].value_counts())

# Survival count by gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival Count by Gender')
plt.show()

# 3D Scatter plot of Age, Fare, and Survival
fig = px.scatter_3d(df, x='Age', y='Fare', z='Survived', color='Survived')
fig.update_layout(title='3D Scatter Plot of Age, Fare, and Survival')
fig.show()