# 🕵️‍♂️ FakeXpose

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

------------------------------------------------------------------------

# 🛡 FakeXpose -- AI Social Media Fraud Detection System

FakeXpose is an **AI-powered social media fraud detection dashboard**
that analyzes profile metadata and behavioral patterns to identify
**fake or bot-controlled accounts**.

The system uses a **Random Forest Machine Learning model** trained on
profile activity features to estimate the probability that an account is
fraudulent.

The project provides both **single profile investigation** and **bulk
dataset analysis** with interactive visualizations and risk indicators.

------------------------------------------------------------------------

# 🚀 Key Features

### 🔍 Single Profile Analysis

Analyze an individual social media profile and detect whether it is
**Fake or Real** using AI.

### 📂 Bulk Profile Analysis

Upload a CSV dataset and analyze **multiple profiles automatically**.

### 📊 Interactive Visualizations

-   Fake vs Real distribution
-   Fraud probability histogram
-   Fraud risk heatmap
-   Feature correlation matrix

### 🎯 AI Confidence Meter

Gauge visualization showing the model's **confidence in prediction**.

### ⭐ Profile Trust Score

A calculated trust score based on fraud probability.

### 🚨 Risk Indicator Detection

Flags suspicious signals such as: - Very low followers - Excessive
following - Low activity - Missing profile picture - Suspicious
follower/following ratio

### 📥 Downloadable Results

Export analyzed profiles as a **CSV report**.

### 🧠 Model Explainability

Displays **feature importance** to explain which signals influenced the
AI decision.

------------------------------------------------------------------------

# 🧠 Machine Learning Model

FakeXpose uses a **Random Forest Classifier** trained on behavioral
signals extracted from social media profile metadata.

### Features used by the model

1.  Followers\
2.  Following\
3.  Posts\
4.  Has Profile Picture\
5.  Username Length\
6.  Followers / Following Ratio

These features help detect **bot-like behavior patterns** often found in
fake accounts.

------------------------------------------------------------------------

# 📊 Dashboard Capabilities

The AI dashboard provides:

• Prediction probability visualization\
• Fraud confidence meter\
• Trust score calculation\
• Fraud indicator analysis\
• Feature importance explanation\
• Dataset correlation analysis\
• Fraud probability distribution\
• Risk level classification

------------------------------------------------------------------------

# 🛠 Tech Stack

  Technology     Purpose
  -------------- ----------------------------
  Python         Core programming language
  Scikit-learn   Machine learning model
  Streamlit      Interactive web dashboard
  Pandas         Data processing
  NumPy          Numerical computation
  Plotly         Interactive visualizations
  Joblib         Model serialization

------------------------------------------------------------------------

# 📂 Project Structure

FakeXpose\
│\
├── app.py \# Main Streamlit dashboard\
├── fakexpose_model.pkl \# Trained ML model\
├── train.py \# Model training script\
├── requirements.txt \# Project dependencies\
├── README.md \# Project documentation\
└── LICENSE \# MIT License

------------------------------------------------------------------------

# ▶️ Running the Project Locally

### 1️⃣ Clone the repository

git clone https://github.com/Arzoo1901/FakeXpose

cd FakeXpose

### 2️⃣ Install dependencies

pip install -r requirements.txt

### 3️⃣ Run the Streamlit application

streamlit run app.py

The dashboard will open in your browser.

------------------------------------------------------------------------

# 📂 CSV Format for Bulk Analysis

Your CSV file must contain the following columns:

followers,following,posts,has_profile_pic,username_length

### Example

1200,150,50,1,8\
350,500,20,0,12\
25,800,2,0,18

------------------------------------------------------------------------

# 🌍 Live Demo

https://fakexpose-nurp6bbxn7dc6mdwbxzzmh.streamlit.app/

------------------------------------------------------------------------

# 👨‍💻 Authors

Arzoo\
Abhinav Vashisth\
Abhinav Katoch

------------------------------------------------------------------------

# 📄 License

This project is licensed under the **MIT License**.\
See the LICENSE file for details.

------------------------------------------------------------------------

# ⚠ Disclaimer

This project is intended for **educational and research purposes**.\
Predictions generated by the model should not be considered definitive
proof of fraud.