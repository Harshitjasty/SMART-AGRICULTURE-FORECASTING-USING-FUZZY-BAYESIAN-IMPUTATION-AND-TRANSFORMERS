# 🌱 Intelligent Crop Growth Prediction using Fuzzy Bayesian Imputation and Transformer-Based Learning (FICFormer)

## 📌 Overview

Agriculture plays a critical role in global sustainability and food production. Modern smart farming systems generate large amounts of environmental sensor data to monitor crop conditions and support decision-making. However, real-world agricultural datasets often suffer from missing values due to sensor failures, transmission errors, or environmental disturbances, which can significantly reduce prediction accuracy.

This project presents an intelligent crop growth prediction framework that addresses incomplete agricultural data through **Fuzzy Bayesian Imputation** and applies advanced **deep learning techniques** for accurate forecasting.

The system reconstructs missing environmental sensor values and utilizes machine learning models such as **LSTM** and the proposed **FICFormer (Fuzzy Imputation Cross Former)** architecture to predict crop growth based on environmental conditions.

---

## 🎯 Project Objectives

- Develop an intelligent crop growth prediction framework for smart agriculture.
- Reconstruct missing environmental sensor data using Fuzzy Bayesian Imputation.
- Analyze temporal relationships among agricultural variables.
- Implement and compare forecasting models such as LSTM and FICFormer.
- Improve prediction accuracy and reliability for agricultural decision support systems.

---

## 🚀 Key Features

✔ Missing value reconstruction using Fuzzy Bayesian Imputation  
✔ Agricultural multivariate time-series analysis  
✔ Deep learning-based forecasting models  
✔ LSTM implementation for baseline comparison  
✔ FICFormer architecture with attention mechanisms  
✔ Model performance evaluation using multiple metrics  
✔ Real-time crop growth prediction capability  
✔ Extendable architecture for future enhancements

---

## 📂 Dataset

**Dataset Used:** Smart Farming Data 2024 (SF24)

The dataset contains environmental and agricultural parameters affecting crop growth, including:

- Temperature
- Humidity
- CO₂ Concentration
- Radiation Levels
- Environmental Sensor Readings
- Crop Growth Values

Dataset source:

https://www.kaggle.com/datasets/datasetengineer/smart-farming-data-2024-sf24

---

## ⚙️ Methodology

### Step 1: Data Collection
Agricultural sensor data is collected and loaded into the system.

### Step 2: Missing Value Detection
Missing values within environmental features are identified.

### Step 3: Fuzzy Bayesian Imputation
Missing values are reconstructed using probabilistic estimation combined with fuzzy logic.

### Step 4: Data Preprocessing
- Data normalization using MinMaxScaler
- Feature-target separation
- Train-test split (80:20)

### Step 5: Model Training
The following models are trained:

#### LSTM
Used as a baseline model to capture temporal dependencies in agricultural data.

#### FICFormer
Proposed transformer-inspired architecture integrating:

- Feature extraction
- Attention mechanisms
- Sequential learning

### Step 6: Prediction and Evaluation
Model predictions are generated and evaluated using performance metrics.

---

## 📊 Evaluation Metrics

The performance of the models is evaluated using:

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- RSE (Relative Squared Error)

Lower metric values indicate better model performance.

---

## 🧠 Model Architecture

### FICFormer Architecture

Environmental Sensor Data  
↓  
Missing Value Detection  
↓  
Fuzzy Bayesian Imputation  
↓  
Data Preprocessing  
↓  
Feature Extraction  
↓  
Attention Mechanism  
↓  
Prediction Layer  
↓  
Crop Growth Prediction

---

## 🔬 Technologies Used

### Programming Language
- Python

### Libraries & Frameworks

- Pandas
- NumPy
- PySpark
- Scikit-Learn
- TensorFlow
- Keras
- Matplotlib

---

## 📈 Expected Applications

- Smart Farming Systems
- Crop Monitoring
- Agricultural Decision Support
- Precision Agriculture
- Yield Forecasting
- Resource Optimization

---

## 🔮 Future Enhancements

- Hybrid models combining FICFormer with GRU
- Stacked architectures using Bidirectional GRU
- Interactive frontend dashboard
- Real-time IoT sensor integration
- Recommendation system for farmers

---

## 👨‍💻 Team Members

- Harshit Chowdary Jasty
- Praneetha Bolneti
- Anirudh Kuchibhotla
- Sasank Jalluri

---

## 📜 License

This project is developed for academic and research purposes.
