# 🕵️‍♂️ FakeXpose

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
---

## AI-Powered Social Media Profile Fraud Detection System

**FakeXpose** is a machine learning web application built with **Python** and **Streamlit** that detects whether a social media profile is **fake** or **real** based on behavioral metrics and engagement patterns. It is designed for researchers, developers, and cybersecurity enthusiasts to analyze suspicious accounts efficiently.

---

## 🚀 Features

* 🔍 **Single Profile Analysis** – Check individual profiles for authenticity.
* 📊 **Bulk CSV Upload Analysis** – Analyze multiple profiles at once.
* 📈 **Fake vs Real Distribution Chart** – Visualize the ratio of fake vs real profiles.
* 📥 **Downloadable Prediction Results** – Export results in CSV format.
* 🎯 **Confidence Score & Risk Level** – See how confident the model is about predictions.
* 🧠 **Intelligent ML Model** – Uses six key behavioral features for detection.

---

## 🧠 Machine Learning Features Used

1. **Followers** – Total number of followers.
2. **Following** – Number of accounts the user follows.
3. **Posts** – Total number of posts.
4. **Has Profile Picture** – Boolean indicating if the account has a profile picture.
5. **Username Length** – Length of the username.
6. **Followers/Following Ratio** – Engagement indicator.

---

## 🛠 Tech Stack

* **Python** – Core programming language.
* **Scikit-learn** – For building the ML model.
* **Pandas & NumPy** – Data manipulation and numerical operations.
* **Matplotlib** – Data visualization.
* **Streamlit** – Web app interface.

---

## ▶️ How to Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/<your-username>/FakeXpose.git
cd FakeXpose
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**

```bash
streamlit run app.py
```

---

## 📂 CSV Format for Bulk Upload

Your CSV file should contain the following columns:

```csv
followers,following,posts,has_profile_pic,username_length
```

**Example:**

```csv
1200,150,50,1,8
350,500,20,0,12
```

---

## 🌍 Live Demo

[https://fakexpose-nurp6bbxn7dc6mdwbxzzmh.streamlit.app/]

---

## 👨‍💻 Authors

* **Arzoo**
* **Abhinav Vashisth**
* **Abhinav Katoch**

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 💡 Tips for Users

* Ensure your CSV file has no empty cells.
* The `has_profile_pic` column should be `1` (Yes) or `0` (No).
* Username length is calculated without special characters or spaces.
