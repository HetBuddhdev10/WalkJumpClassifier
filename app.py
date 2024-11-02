import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set_style('whitegrid')

# Define directories
models_dir = 'models'
plots_dir = 'plots'

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# ----------------------------------------
# Load the trained Random Forest model
# ----------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
        return None
    model = joblib.load(model_path)
    return model

model = load_model()

# ----------------------------------------
# Load the scaler used during training
# ----------------------------------------
@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at {scaler_path}. Please ensure the scaler is saved correctly.")
        return None
    scaler = joblib.load(scaler_path)
    return scaler

scaler = load_scaler()

# ----------------------------------------
# Streamlit App Title and Description
# ----------------------------------------
st.title("📱 Activity Recognition: Walking vs. Jumping Classification")

st.markdown("""
Welcome to the **Activity Recognition** application! This tool leverages machine learning to classify activities based on accelerometer data collected from your smartphone.

### **About This App**
- **Data Collection:** The accelerometer data used in this app was collected using the [phyphox](https://phyphox.org/) app on a smartphone. Phyphox allows users to access and record various sensor data directly from their phone.
- **Purpose:** Classify segments of your movement data as either **Walking** or **Jumping**.
- **Visualization:** Provides an interactive plot to visualize activity classifications over time.
- **Downloadable Results:** Allows you to download the prediction results for further analysis or record-keeping.

### **How to Use**
1. **Prepare Your Data:** Ensure your CSV file is formatted correctly with the required columns:
    - **Time (s):** Timestamp of each reading.
    - **Acceleration x (m/s²):** Acceleration along the X-axis.
    - **Acceleration y (m/s²):** Acceleration along the Y-axis.
    - **Acceleration z (m/s²):** Acceleration along the Z-axis.
    - **Absolute acceleration (m/s²):** Combined acceleration magnitude.

2. **Upload the CSV File:** Use the uploader below to select and upload your CSV file.

3. **View Results:** After processing, the app will display:
    - **Data Preview:** A snapshot of your uploaded data.
    - **Data Preprocessing:** Information about data cleaning and segmentation.
    - **Feature Extraction:** Overview of extracted features from your data.
    - **Predictions:** Classification results indicating whether each segment is Walking or Jumping.
    - **Activity Plot:** Visual representation of your activities over time.

4. **Download Predictions:** Export your prediction results as a CSV file for your records or further analysis.

**Let's get started and uncover your activity patterns! 🚶‍♂️🏃‍♀️**
""")

# ----------------------------------------
# File Upload Section
# ----------------------------------------
st.header("📥 Upload Accelerometer Data CSV")
st.markdown("""
Please upload a CSV file containing your accelerometer data. Ensure that the file includes the following columns:
- **Time (s)**
- **Acceleration x (m/s²)**
- **Acceleration y (m/s²)**
- **Acceleration z (m/s²)**
- **Absolute acceleration (m/s²)**
""")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Attempt to read the CSV with comma separator
        data = pd.read_csv(uploaded_file)
        delimiter_used = 'comma'
    except Exception:
        try:
            # If comma separator fails, try tab separator
            data = pd.read_csv(uploaded_file, sep='\t')
            delimiter_used = 'tab'
        except Exception:
            try:
                # If tab separator fails, try delim_whitespace
                data = pd.read_csv(uploaded_file, delim_whitespace=True)
                delimiter_used = 'whitespace'
            except Exception as e:
                st.error(f"Error reading the CSV file: {e}")
                st.stop()
    
    # ----------------------------------------
    # Data Preprocessing
    # ----------------------------------------
    st.header("🛠️ Data Preprocessing and Feature Extraction")
    st.markdown("""
    The uploaded data undergoes the following preprocessing steps to ensure it's suitable for analysis:
    - **Column Renaming:** Standardizes column names by converting them to lowercase and replacing special characters.
    - **Handling Missing Values:** Removes any incomplete entries to maintain data integrity.
    - **Segmenting Data:** Breaks down the data into overlapping windows to capture activity patterns effectively.
    """)
        
    # Required columns based on training
    required_columns = ['acceleration x (m/s^2)', 'acceleration y (m/s^2)', 'acceleration z (m/s^2)']
    
    
    # Check if required columns exist
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.error(f"Uploaded CSV does not contain the required columns: {missing_cols}")
        st.stop()
    
    # Handle missing values by dropping rows with NaNs in required columns
    initial_shape = data.shape
    data.dropna(subset=required_columns, inplace=True)
    final_shape = data.shape
    st.write(f"Dropped {initial_shape[0] - final_shape[0]} rows due to missing values.")

    # ----------------------------------------
    # Segmenting the Data into Windows
    # ----------------------------------------
    st.header("📊 Segmenting Data into Windows")
    st.markdown("""
    To capture meaningful activity patterns, the data is segmented into overlapping windows:
    - **Window Duration:** 5 seconds
    - **Sampling Rate:** 100 samples per second
    - **Overlap:** 50% (i.e., each new window starts halfway through the previous window)
    """)

    # Define sampling rate and window size (must match training parameters)
    sampling_rate = 100  # samples per second (adjust if different)
    window_duration = 5  # seconds
    window_size = sampling_rate * window_duration  # total samples per window
    step_size = window_size // 2  # 50% overlap

    # Function to segment data
    def segment_data(data, window_size, step_size):
        segments = []
        for start in range(0, len(data) - window_size + 1, step_size):
            end = start + window_size
            segment = data.iloc[start:end]
            segments.append(segment)
        return segments

    segments = segment_data(data, window_size, step_size)
    st.write(f"Total segments created: {len(segments)}")

    # ----------------------------------------
    # Feature Extraction
    # ----------------------------------------
    st.header("🔍 Extracting Features from Segments")
    st.markdown("""
    From each data segment, we extract statistical features that are instrumental in distinguishing between walking and jumping activities. These features include:
    - **Mean**
    - **Standard Deviation**
    - **Maximum**
    - **Minimum**
    - **Median**
    - **Skewness**
    - **Kurtosis**
    """)

    def extract_features(segment):
        features = {}
        for col in required_columns:
            features[f'{col}_mean'] = segment[col].mean()
            features[f'{col}_std'] = segment[col].std()
            features[f'{col}_max'] = segment[col].max()
            features[f'{col}_min'] = segment[col].min()
            features[f'{col}_median'] = segment[col].median()
            features[f'{col}_skew'] = segment[col].skew()
            features[f'{col}_kurtosis'] = segment[col].kurtosis()
        return features

    feature_list = []
    for segment in segments:
        features = extract_features(segment)
        feature_list.append(features)

    features_df = pd.DataFrame(feature_list)
    st.write("Sample of Extracted Features:")
    st.write(features_df.head())

    # ----------------------------------------
    # Feature Scaling
    # ----------------------------------------
    st.header("⚖️ Scaling Features")
    st.markdown("""
    To ensure that all features contribute equally to the prediction process, we normalize the data using a pre-trained scaler.
    """)

    if scaler is not None:
        try:
            X_scaled = scaler.transform(features_df)
            st.success("Features scaled successfully.")
        except ValueError as ve:
            st.error(f"Feature scaling failed: {ve}")
            st.stop()
    else:
        st.error("Scaler is not loaded. Cannot perform feature scaling.")
        st.stop()

    # ----------------------------------------
    # Making Predictions
    # ----------------------------------------
    st.header("🤖 Making Predictions")
    st.markdown("""
    Utilizing the trained Random Forest model, each data segment is classified as either **Walking** or **Jumping**. The model also provides probabilities indicating the confidence of each prediction.
    """)

    if model is not None:
        try:
            predictions = model.predict(X_scaled)
            prediction_proba = model.predict_proba(X_scaled)
            st.success("Predictions made successfully.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
    else:
        st.error("Model is not loaded. Cannot make predictions.")
        st.stop()

    # ----------------------------------------
    # Mapping Predictions to Time
    # ----------------------------------------
    st.header("🕒 Mapping Predictions to Time")
    st.markdown("""
    To provide temporal context, each prediction is mapped to the corresponding time segment's midpoint. This allows for a clear visualization of activity transitions over time.
    """)

    # Assign each prediction to the middle time point of its window
    if 'time_s' in data.columns:
        time_series = data['time_s']
    else:
        # Attempt to find a column that resembles time
        time_cols = [col for col in data.columns if 'time' in col]
        if time_cols:
            time_series = data[time_cols[0]]
            st.warning(f"Using '{time_cols[0]}' as the time column.")
        else:
            st.error("No time column found in the data.")
            st.stop()

    window_mid_points = []
    for i in range(len(segments)):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        mid_idx = start_idx + window_size // 2
        if mid_idx < len(time_series):
            window_mid_points.append(time_series.iloc[mid_idx])
        else:
            window_mid_points.append(time_series.iloc[-1])

    # Create a DataFrame for predictions
    prediction_df = pd.DataFrame({
        'time_s': window_mid_points,
        'predicted_activity': predictions,
        'prob_walking': prediction_proba[:, 0],
        'prob_jumping': prediction_proba[:, 1]
    })

    st.write("Sample of Prediction Results:")
    st.write(prediction_df.head())

    # ----------------------------------------
    # Plotting the Activities Over Time
    # ----------------------------------------
    st.header("📈 Activity Plot Over Time")
    st.markdown("""
    The following plot visualizes your activity classifications over time, distinguishing between **Walking** and **Jumping** activities.
    """)

    plt.figure(figsize=(15, 5))
    sns.lineplot(x='time_s', y='predicted_activity', data=prediction_df, marker='o')
    plt.yticks([0, 1], ['Walking', 'Jumping'])
    plt.xlabel('Time (s)')
    plt.ylabel('Predicted Activity')
    plt.title('Walking vs. Jumping Over Time')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'activity_plot_over_time.png')
    plt.savefig(plot_path)
    st.pyplot(plt)
    plt.close()

    # ----------------------------------------
    # Download Prediction Results
    # ----------------------------------------
    st.header("💾 Download Prediction Results")
    st.markdown("""
    You can download the prediction results in CSV format for your records or further analysis.
    """)

    csv = prediction_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='activity_predictions.csv',
        mime='text/csv',
    )

else:
    st.info("🕒 Awaiting CSV file to be uploaded.")
