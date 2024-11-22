# Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

# Load the dataset
df = pd.read_csv('breast_cancer_dataset.csv')

# Display the first 10 rows
print("First 10 Rows of the Dataset:")
print(df.head(10))

# Display column information
print("\nColumn Information:")
df.info()

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check for zero values
print("\nZero Values in Each Column:")
zero_values = (df == 0).sum()
print(zero_values)

# Replace zero values with the column mean
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if zero_values[col] > 0:  # Only process columns with zero values
        df[col] = df[col].replace(0, df[col].mean())

# Confirm zero values replaced
print("\nZero Values After Replacement:")
print((df == 0).sum())

# Distribution 
# Get the counts of each diagnosis
diagnosis_counts = df['diagnosis'].value_counts()

# Plot a pie chart
plt.figure(figsize=(7, 7))
diagnosis_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Diagnosis Distribution')
plt.ylabel('')  # Hide the y-axis label
plt.show()

# Map 'M' to 1 and 'B' to 0 in the diagnosis column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Check the changes
print("\nUpdated Diagnosis Column:")
print(df['diagnosis'].head())

# Outlier Detection - Z-Score Method
# Compute the Z-scores for all numerical columns
z_scores = zscore(df.select_dtypes(include=['float64', 'int64']))

# Convert the Z-scores to a DataFrame for easier handling
z_scores_df = pd.DataFrame(z_scores, columns=df.select_dtypes(include=['float64', 'int64']).columns)

# Count outliers (Z-score > 3 or < -3) for each column
outliers_zscore_count = (z_scores_df.abs() > 3).sum()

# Replace outliers with column mean for Z-score method using .loc to avoid warning
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df.loc[z_scores_df[col].abs() > 3, col] = df[col].mean()  # Use .loc to modify values

# Display the count of outliers for each column detected using Z-Score
print("\nOutliers count for each column (Z-Score method):")
print(outliers_zscore_count)

# Outlier Detection - IQR Method
# Compute the IQR for each numerical column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outliers (values outside the IQR bounds) for each column
outliers_iqr_count = ((df < lower_bound) | (df > upper_bound)).sum()

# Replace outliers with column mean for IQR method using .loc to avoid warning
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df.loc[(df[col] < lower_bound[col]) | (df[col] > upper_bound[col]), col] = df[col].mean()  # Use .loc to modify values

# Display the count of outliers for each column detected using IQR
print("\nOutliers count for each column (IQR method):")
print(outliers_iqr_count)

# Compute the correlation matrix
correlation_matrix = df.corr()

# Create the heatmap
plt.figure(figsize=(4, 8))
sns.heatmap(df.corr()["diagnosis"].sort_values(ascending=False).abs().to_frame(), annot=True, cmap='coolwarm')
plt.show()

# Set a threshold for correlation (e.g., 0.2)
correlation_threshold = 0.2

# Get the columns that are highly correlated with the target (diagnosis)
highly_correlated_columns = correlation_matrix["diagnosis"][correlation_matrix["diagnosis"].abs() > correlation_threshold].index

# Keep only the relevant columns (including 'diagnosis')
df_filtered = df[highly_correlated_columns]

# Determine the dropped columns by comparing original columns with the filtered ones
dropped_columns = set(df.columns) - set(df_filtered.columns)

# Display the remaining columns and the dropped columns
print("\nRemaining Columns After Dropping Unrelated Columns:")
print(df_filtered.columns)

print("\nDropped Columns:")
print(dropped_columns)

# Display the final processed dataframe
print("\nProcessed DataFrame After All Steps:")
print(df_filtered.head())  # Show the first few rows of the processed dataframe

# Step 1: Splitting the data into features and target variable
X = df_filtered.drop('diagnosis', axis=1)  # Features
y = df_filtered['diagnosis']  # Target variable

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# Predicting with Logistic Regression
y_pred_logreg = logreg.predict(X_test_scaled)

# Step 5: Support Vector Machine (SVM)
svm = SVC(kernel='rbf', random_state=42)  
svm.fit(X_train_scaled, y_train)

# Predicting with SVM
y_pred_svm = svm.predict(X_test_scaled)

# Step 6: Model Evaluation

# Logistic Regression Evaluation
print("Logistic Regression Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))

# SVM Evaluation
print("\nSupport Vector Machine Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Step 1: Learning Curves for Logistic Regression
train_sizes_logreg, train_scores_logreg, test_scores_logreg = learning_curve(
    logreg, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Step 2: Learning Curves for SVM
train_sizes_svm, train_scores_svm, test_scores_svm = learning_curve(
    svm, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Step 3: Plotting the learning curves

# Calculate the mean and standard deviation for training and testing scores
train_mean_logreg = train_scores_logreg.mean(axis=1)
train_std_logreg = train_scores_logreg.std(axis=1)

test_mean_logreg = test_scores_logreg.mean(axis=1)
test_std_logreg = test_scores_logreg.std(axis=1)

train_mean_svm = train_scores_svm.mean(axis=1)
train_std_svm = train_scores_svm.std(axis=1)

test_mean_svm = test_scores_svm.mean(axis=1)
test_std_svm = test_scores_svm.std(axis=1)

# Plotting Learning Curve for Logistic Regression
plt.figure(figsize=(8, 6))
plt.plot(train_sizes_logreg, train_mean_logreg, color='blue', label='Train Accuracy', linestyle='-', marker='o')
plt.plot(train_sizes_logreg, test_mean_logreg, color='orange', label='Test Accuracy', linestyle='--', marker='o')
plt.fill_between(train_sizes_logreg, train_mean_logreg - train_std_logreg, train_mean_logreg + train_std_logreg, color='blue', alpha=0.1)
plt.fill_between(train_sizes_logreg, test_mean_logreg - test_std_logreg, test_mean_logreg + test_std_logreg, color='orange', alpha=0.1)

plt.title('Learning Curve: Logistic Regression', fontsize=14)
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Plotting Learning Curve for SVM
plt.figure(figsize=(8, 6))
plt.plot(train_sizes_svm, train_mean_svm, color='green', label='Train Accuracy', linestyle='-', marker='o')
plt.plot(train_sizes_svm, test_mean_svm, color='red', label='Test Accuracy', linestyle='--', marker='o')
plt.fill_between(train_sizes_svm, train_mean_svm - train_std_svm, train_mean_svm + train_std_svm, color='green', alpha=0.1)
plt.fill_between(train_sizes_svm, test_mean_svm - test_std_svm, test_mean_svm + test_std_svm, color='red', alpha=0.1)

plt.title('Learning Curve: SVM', fontsize=14)
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='best')
plt.grid(True)
plt.show()