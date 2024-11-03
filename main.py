# Activity Recognition Project: Walking vs. Jumping Classification
# Author: Het Buddhdev 
# Date: 2024-06-13
# Description: 
# This project builds a machine learning model to classify walking and jumping activities using accelerometer data.
# The dataset consists of labeled CSV files containing acceleration data along three axes.
# The workflow includes data preprocessing, feature extraction, model training, and evaluation using Random Forest and Logistic Regression classifiers.

# Import necessary libraries
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set plotting style for better aesthetics
sns.set_style('whitegrid')

# Define directories
models_dir = 'models'
plots_dir = 'plots'

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# ----------------------------------------
# Step 1: Data Loading and Combining
# ----------------------------------------

# Set the directory where the dataset CSV files are stored
data_dir = r"C:\Users\hetb0\Desktop\WalkingVrunning\Datasets"  # Update this path as needed

# Check if the specified data directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

# Retrieve all files in the data directory
files = os.listdir(data_dir)

# Filter out only the CSV files for processing
csv_files = [file for file in files if file.endswith('.csv')]

# Initialize a list to hold individual DataFrames from each CSV file
dataframes = []

# Loop through each CSV file and load the data
for file in csv_files:
    file_path = os.path.join(data_dir, file)  # Full path to the CSV file
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
    except Exception as e:
        # Handle any errors that occur during file reading
        print(f"Failed to read {file}: {e}")
        continue  # Skip to the next file if there's an error

    # Standardize column names: remove whitespace and convert to lowercase
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
    df.columns = df.columns.str.lower()  # Convert column names to lowercase

    # Convert filename to lowercase for consistent processing
    filename_lower = file.lower()

    # Label the data based on the activity inferred from the filename
    if 'walk' in filename_lower:
        df['label'] = 0  # 0 represents walking
        df['activity'] = 'Walking'
    elif 'jump' in filename_lower:
        df['label'] = 1  # 1 represents jumping
        df['activity'] = 'Jumping'
    else:
        # Skip files that do not contain recognizable activity labels
        print(f"Unknown activity type in file: {file}. Skipping this file.")
        continue

    # Extract person identifier from the filename for potential future use
    if 'person1' in filename_lower:
        df['person'] = 'Person1'
    elif 'person2' in filename_lower:
        df['person'] = 'Person2'
    elif 'person3' in filename_lower:
        df['person'] = 'Person3'
    else:
        df['person'] = 'Unknown'  # Assign 'Unknown' if person identifier is missing

    # Append the processed DataFrame to the list
    dataframes.append(df)

# Combine all individual DataFrames into a single DataFrame
if dataframes:
    combined_data = pd.concat(dataframes, ignore_index=True)
else:
    raise ValueError("No valid dataframes to combine.")

# ----------------------------------------
# Step 2: Initial Data Exploration
# ----------------------------------------

# (Optional) Uncomment the following lines if you need to perform data exploration
# print("\nSummary Statistics:")
# print(combined_data.describe())

# print("\nData Types:")
# print(combined_data.dtypes)

# print("\nMissing Values:")
# print(combined_data.isnull().sum())

# print("\nLabel Distribution:")
# print(combined_data['label'].value_counts())

# ----------------------------------------
# Step 3: Data Preprocessing
# ----------------------------------------

# Handle missing values by dropping any rows that contain NaN values
combined_data.dropna(inplace=True)

# Identify the numeric columns that contain accelerometer data
numeric_columns = ['acceleration x (m/s^2)', 'acceleration y (m/s^2)', 'acceleration z (m/s^2)']

# Ensure column names are in lowercase to match the standardized DataFrame
numeric_columns = [col.lower() for col in numeric_columns]

# Check for any missing expected columns and raise an error if found
missing_columns = [col for col in numeric_columns if col not in combined_data.columns]
if missing_columns:
    raise KeyError(f"The following columns are missing from the data: {missing_columns}")

# Remove outliers from the numeric columns using the Z-score method
z_scores = np.abs(stats.zscore(combined_data[numeric_columns]))
threshold = 3  # Z-score threshold for identifying outliers
combined_data = combined_data[(z_scores < threshold).all(axis=1)]

# Apply a moving average filter to smooth the accelerometer data and reduce noise
window_size = 5  # Window size for the moving average
for col in numeric_columns:
    combined_data[col] = combined_data[col].rolling(window=window_size).mean()

# Drop rows with NaN values that result from the moving average operation
combined_data.dropna(inplace=True)

# Reset the DataFrame index after preprocessing steps
combined_data.reset_index(drop=True, inplace=True)

# ----------------------------------------
# Step 4: Data Visualization 
# ----------------------------------------

# Visualize a sample of accelerometer data for walking activity
walking_data = combined_data[combined_data['label'] == 0]
plt.figure(figsize=(12, 6))
for col in numeric_columns:
    plt.plot(walking_data[col].values[:1000], label=col)
plt.title('Sample Accelerometer Data for Walking')
plt.xlabel('Samples')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'sample_walking_data.png'))
plt.close()

# Visualize a sample of accelerometer data for jumping activity
jumping_data = combined_data[combined_data['label'] == 1]
plt.figure(figsize=(12, 6))
for col in numeric_columns:
    plt.plot(jumping_data[col].values[:1000], label=col)
plt.title('Sample Accelerometer Data for Jumping')
plt.xlabel('Samples')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'sample_jumping_data.png'))
plt.close()

# ----------------------------------------
# Step 5: Segmenting the Data and Feature Extraction
# ----------------------------------------

# Define sampling rate and window size for segmenting the data
sampling_rate = 100  # Number of samples per second (adjust based on your dataset)
window_duration = 5  # Duration of each window in seconds
window_size_samples = sampling_rate * window_duration  # Total samples per window

def segment_data(data, window_size):
    """
    Segments the data into overlapping windows.

    Parameters:
        data (pd.DataFrame): The preprocessed accelerometer data.
        window_size (int): Number of samples per window.

    Returns:
        segments (list): List of DataFrame segments.
        labels (list): List of labels corresponding to each segment.
    """
    segments = []
    labels = []
    step_size = window_size // 2  # 50% overlap between windows

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        segment = data.iloc[start:end]
        # Ensure that the entire segment contains only one type of activity
        if segment['label'].nunique() == 1:
            segments.append(segment)
            labels.append(segment['label'].iloc[0])
    return segments, labels

# Segment the combined data into windows
segments, labels = segment_data(combined_data, window_size_samples)

def extract_features(segment):
    """
    Extracts statistical features from a data segment.

    Parameters:
        segment (pd.DataFrame): A windowed segment of the accelerometer data.

    Returns:
        features (dict): Dictionary of extracted features.
    """
    features = {}
    for col in numeric_columns:
        data = segment[col]
        features[f'{col}_mean'] = data.mean()
        features[f'{col}_std'] = data.std()
        features[f'{col}_max'] = data.max()
        features[f'{col}_min'] = data.min()
        features[f'{col}_median'] = data.median()
        features[f'{col}_skew'] = data.skew()
        features[f'{col}_kurtosis'] = data.kurtosis()
    return features

# Extract features from each segment and compile them into a list
feature_list = []
for segment, label in zip(segments, labels):
    features = extract_features(segment)
    features['Label'] = label  # Add the corresponding label to the features
    feature_list.append(features)

# Create a DataFrame from the list of feature dictionaries
features_df = pd.DataFrame(feature_list)

# ----------------------------------------
# Step 6: Feature Scaling
# ----------------------------------------

# Separate the features (X) and labels (y) for model training
X = features_df.drop('Label', axis=1)
y = features_df['Label']

# Initialize the StandardScaler for feature normalization
scaler = StandardScaler()

# Fit the scaler on the features and transform them
X_scaled = scaler.fit_transform(X)

# Save the scaler to disk
scaler_filename = os.path.join(models_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_filename)

# ----------------------------------------
# Step 7: Splitting the Data into Training and Testing Sets
# ----------------------------------------

# Split the dataset into training and testing sets with an 80-20 ratio
# Setting random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# Step 8: Model Training with Random Forest Classifier
# ----------------------------------------

# Initialize the Random Forest Classifier with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model on the training data
try:
    rf_model.fit(X_train, y_train)
except Exception as e:
    print(f"Random Forest model training failed: {e}")
    raise

# ----------------------------------------
# Step 9: Model Evaluation
# ----------------------------------------

# Make predictions on the test set using the trained Random Forest model
y_pred_rf = rf_model.predict(X_test)

# ----------------------------------------
# Step 10: Post-Processing to Enforce No Consecutive Jumps
# ----------------------------------------

def enforce_no_consecutive_jumps(predictions):
    """
    Modifies the predictions so that if a 'Jump' (1) is detected,
    the next prediction is set to 'Walk' (0).

    Parameters:
        predictions (np.array): Array of predicted labels.

    Returns:
        modified_predictions (np.array): Array with enforced rules.
    """
    modified_predictions = predictions.copy()
    for i in range(len(modified_predictions) - 1):
        if modified_predictions[i] == 1:
            modified_predictions[i + 1] = 0
    return modified_predictions

# Apply the post-processing function to Random Forest predictions
y_pred_rf_modified = enforce_no_consecutive_jumps(y_pred_rf)

# ----------------------------------------
# Step 11: Continue Model Evaluation with Modified Predictions
# ----------------------------------------

# Generate the classification report for Random Forest with modified predictions
classification_report_rf = classification_report(y_test, y_pred_rf_modified)

# Generate the confusion matrix for Random Forest with modified predictions
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf_modified)

# Calculate the accuracy score for Random Forest as a percentage
rf_accuracy = accuracy_score(y_test, y_pred_rf_modified) * 100

# Display evaluation metrics
print("\nRandom Forest Classification Report (After Post-Processing):")
print(classification_report_rf)

print("Random Forest Confusion Matrix (After Post-Processing):")
print(confusion_matrix_rf)

print(f"Random Forest Accuracy Score (After Post-Processing): {rf_accuracy:.2f}%")

# ----------------------------------------
# Step 12: ROC Curve and AUC for Random Forest
# ----------------------------------------

# Compute the predicted probabilities for the positive class
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR) for ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

# Compute the Area Under the Curve (AUC) for Random Forest
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot the ROC Curve for Random Forest and save it
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.title('Random Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'random_forest_roc_curve.png'))
plt.close()

# ----------------------------------------
# Step 13: Plot Time vs Activity
# ----------------------------------------

# Calculate time for each prediction window
step_size = window_size_samples // 2  # 50% overlap
window_shift_seconds = step_size / sampling_rate  # e.g., 2.5 seconds
window_duration_seconds = window_size_samples / sampling_rate  # e.g., 5 seconds
total_windows = len(y_pred_rf_modified)
time = np.arange(total_windows) * window_shift_seconds + (window_duration_seconds / 2)

# Plot Time vs Activity for Random Forest
plt.figure(figsize=(15, 5))
plt.step(time, y_pred_rf_modified, where='mid', label='Random Forest Predictions', color='red')
plt.yticks([0, 1], ['Walking', 'Jumping'])
plt.xlabel('Time (s)')
plt.ylabel('Activity')
plt.title('Time vs Activity - Random Forest')
plt.legend()
plt.tight_layout()
plt.close()

# ----------------------------------------
# Step 14: Compare with Logistic Regression Model
# ----------------------------------------

# Initialize the Logistic Regression model with increased max_iter for convergence
lr_model = LogisticRegression(max_iter=1000)

# Train the Logistic Regression model on the training data
try:
    lr_model.fit(X_train, y_train)
except Exception as e:
    print(f"Logistic Regression model training failed: {e}")
    raise

# Make predictions on the test set using Logistic Regression
y_pred_lr = lr_model.predict(X_test)

# Apply the post-processing function to Logistic Regression predictions
y_pred_lr_modified = enforce_no_consecutive_jumps(y_pred_lr)

# Generate the classification report for Logistic Regression with modified predictions
classification_report_lr = classification_report(y_test, y_pred_lr_modified)

# Calculate the accuracy score for Logistic Regression as a percentage
lr_accuracy = accuracy_score(y_test, y_pred_lr_modified) * 100

# Display evaluation metrics
print("\nLogistic Regression Classification Report (After Post-Processing):")
print(classification_report_lr)

print(f"Logistic Regression Accuracy Score (After Post-Processing): {lr_accuracy:.2f}%")

# Compute the predicted probabilities for the positive class using Logistic Regression
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR) for ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)

# Compute the Area Under the Curve (AUC) for Logistic Regression
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Plot the ROC Curves for both Random Forest and Logistic Regression for comparison and save it
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'roc_curve_comparison.png'))
plt.close()

# ----------------------------------------
# Step 15: Save the Trained Models
# ----------------------------------------

# Save the trained Random Forest model to disk using joblib
rf_model_filename = os.path.join(models_dir, 'random_forest_model.pkl')
joblib.dump(rf_model, rf_model_filename)

# Save the trained Logistic Regression model to disk using joblib
lr_model_filename = os.path.join(models_dir, 'logistic_regression_model.pkl')
joblib.dump(lr_model, lr_model_filename)

# Final Print Statement to indicate successful execution
print("Model training, evaluation, and saving completed successfully.")
