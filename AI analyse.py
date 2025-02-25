import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

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

# Feature Engineering
# Select features and target variable
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'PrCh', 'Fare']]  # Features
y = df['Survived']  # Target variable

# Encode categorical variables (e.g., 'Sex' and 'Pclass')
categorical_features = ['Sex', 'Pclass']
numerical_features = ['Age', 'SibSp', 'PrCh', 'Fare']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features
        ('cat', OneHotEncoder(), categorical_features)  # One-hot encode categorical features
    ])

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))  # Logistic Regression model
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict on the validation set (if needed)
val_predictions = model.predict(val_df[['Pclass', 'Sex', 'Age', 'SibSp', 'PrCh', 'Fare']])
print("Validation Predictions:", val_predictions)

# Visualizations for Final Predictions

# 1. Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 2. Actual vs Predicted Survival Bar Plot
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
comparison_df = comparison_df.reset_index(drop=True)
comparison_df.head(10).plot(kind='bar', figsize=(10, 6))
plt.title('Actual vs Predicted Survival (First 10 Samples)')
plt.xlabel('Passenger Index')
plt.ylabel('Survival Status (0 = Not Survived, 1 = Survived)')
plt.show()

# 3. ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# 4. Prediction Probabilities Distribution
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_proba, bins=20, kde=True)
plt.title('Distribution of Predicted Probabilities for Survival')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Frequency')
plt.show()