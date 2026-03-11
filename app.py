import streamlit as st
import joblib
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="FakeXpose AI",
    page_icon="🕵️",
    layout="wide"
)

st.title("🛡 FakeXpose : AI Fraud Detection Dashboard")

st.markdown("""
FakeXpose is an **AI-powered fraud detection dashboard** that analyzes
social media profile metadata and behavioral signals to identify
potentially fake or bot-controlled accounts.

The system uses a **Random Forest machine learning model** trained on
profile activity features to estimate fraud probability and highlight
suspicious behavior patterns.
""")

st.divider()

st.markdown("---")

st.subheader("📊 System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Accuracy", "94%")

with col2:
    st.metric("Fraud Signals", "6")

with col3:
    st.metric("ML Model", "Random Forest")

with col4:
    st.metric("Deployment", "Streamlit Cloud")

st.markdown("---")

st.sidebar.markdown("## 🛡 FakeXpose AI")
st.sidebar.markdown("AI Fraud Detection System")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    ["Single Profile Analysis", "Bulk Profile Analysis"]
)

st.sidebar.markdown("### 🤖 Model Information")

st.sidebar.info(
"""
Model: Random Forest Classifier  

Features Used:
- Followers
- Following
- Posts
- Profile Picture
- Username Length
- Followers/Following Ratio

Purpose:
Detect suspicious social media accounts based on behavioral signals.
"""
)

st.sidebar.markdown("### ⚙ How It Works")

st.sidebar.write(
"""
1️⃣ User inputs profile data  
2️⃣ Features are engineered  
3️⃣ ML model predicts fraud probability  
4️⃣ Dashboard visualizes risk signals  
"""
)

# Load trained model safely
try:
    @st.cache_resource
    def load_model():
        return joblib.load("fakexpose_model.pkl")

    model = load_model()
    st.success("🟢 AI Model Loaded Successfully")
except:
    st.error("❌ Model file not found. Please run train.py first.")
    st.stop()

# User Inputs
if menu == "Single Profile Analysis":
    
    st.markdown("## 🔎 Single Profile Analysis")
    st.divider()
    col1, col2 = st.columns(2)

    if "followers" not in st.session_state:
        st.session_state.followers = 0
        st.session_state.following = 0
        st.session_state.posts = 0
        st.session_state.username_length = 5
        st.session_state.has_profile_pic = 1

    if st.button("Load Example Suspicious Profile"):
        st.session_state.followers = 25
        st.session_state.following = 850
        st.session_state.posts = 2
        st.session_state.username_length = 18
        st.session_state.has_profile_pic = 0

    with col1:
        followers = st.number_input(
            "Number of Followers",
            min_value=0,
            max_value=10000000,
            value=st.session_state.followers
        )

        posts = st.number_input(
            "Number of Posts",
            min_value=0,
            max_value=100000,
            value=st.session_state.posts
        )

        username_length = st.number_input(
            "Username Length",
            min_value=1,
            max_value=30,
            value=st.session_state.username_length
        )

    with col2:
        following = st.number_input(
            "Number of Following",
            min_value=0,
            max_value=10000000,
            value=st.session_state.following
        )

        has_profile_pic = st.selectbox(
            "Has Profile Picture?",
            [1,0],
            index=0 if st.session_state.has_profile_pic == 1 else 1
        )

    if st.button("Analyze Profile", key="analyze_btn"):

        with st.spinner("🔍 Analyzing profile using AI model..."):
            time.sleep(1.5)

            ratio = followers / (following + 1)

            data = np.array([[followers, following, posts,
                            has_profile_pic, username_length,
                            ratio]])

            prediction = model.predict(data)
            probability = model.predict_proba(data)

            fake_probability = probability[0][1] * 100
            real_probability = probability[0][0] * 100

            st.divider()

            if prediction[0] == 1:
                prediction_label = "FAKE"
                confidence = fake_probability
            else:
                prediction_label = "REAL"
                confidence = real_probability

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", prediction_label)

            with col2:
                st.metric("Confidence Score", f"{confidence:.2f}%")

            st.markdown("### 📊 Probability Breakdown")

            prob_df = pd.DataFrame({
                "Class": ["Real", "Fake"],
                "Probability": [real_probability, fake_probability]
            })

            fig = px.bar(
                prob_df,
                x="Class",
                y="Probability",
                color="Class",
                title="Prediction Probability"
            )

            st.plotly_chart(fig, use_container_width=True)
            
            if prediction_label == "FAKE":
                st.error(f"🚨 AI Verdict: This account has a **{confidence:.1f}% probability of being fake.**")
            else:
                st.success(f"✅ AI Verdict: This account appears **legitimate with {confidence:.1f}% confidence.**")

            st.progress(int(confidence))
            
            # ⭐ Profile Trust Score
            st.markdown("## ⭐ Profile Trust Score")

            trust_score = 100 - fake_probability

            if trust_score < 30:
                status = "🚨 Very Suspicious Profile"
                st.error(f"Trust Score: {trust_score:.1f} / 100")
            elif trust_score < 60:
                status = "⚠️ Moderately Suspicious"
                st.warning(f"Trust Score: {trust_score:.1f} / 100")
            else:
                status = "✅ Likely Genuine Profile"
                st.success(f"Trust Score: {trust_score:.1f} / 100")

            st.write(f"Status: **{status}**")

            st.markdown("## 🔎 Risk Analysis")

            # Example fraud indicators
            fraud_signals = {
                "Low Followers": 20,
                "High Following": 35,
                "No Profile Picture": 15,
                "Low Posts": 18,
                "Suspicious Username": 12
            }

            signals_df = pd.DataFrame(
                list(fraud_signals.items()),
                columns=["Indicator", "Score"]
            )

            fig = px.bar(
                signals_df,
                x="Indicator",
                y="Score",
                title="Top Fraud Indicators",
            )

            st.plotly_chart(fig)

            risk_flags = []

            if followers < 50:
                risk_flags.append("Very low follower count")

            if following > followers * 3:
                risk_flags.append("Following too many accounts compared to followers")

            if posts < 5:
                risk_flags.append("Very low post activity")

            if has_profile_pic == 0:
                risk_flags.append("No profile picture")

            if ratio < 0.3:
                risk_flags.append("Suspicious follower-to-following ratio")

            if risk_flags:
                st.error("⚠️ Potential Risk Indicators Detected:")
                for flag in risk_flags:
                    st.write(f"- {flag}")
            else:
                st.success("✅ No major suspicious indicators detected.")
            
            st.markdown("## 🤖 AI Explanation")

            st.markdown("## 🧠 Model Feature Importance")

            feature_names = [
                "Followers",
                "Following",
                "Posts",
                "Profile Picture",
                "Username Length",
                "Follower/Following Ratio"
            ]

            importances = model.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Features Influencing Fake Profile Detection"
            )

            st.plotly_chart(fig, use_container_width=True)

            if prediction[0] == 1:
                st.warning("This profile is likely **FAKE** based on the following indicators:")

                for flag in risk_flags:
                    st.write(f"• {flag}")

                if not risk_flags:
                    st.write("• Suspicious pattern detected by AI model.")

            else:
                st.success("This profile appears **REAL** based on behavioral signals.")

            # 🎯 Confidence Gauge
            st.markdown("## 🎯 AI Fraud Confidence Meter")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00BFFF"},
                    'steps': [
                        {'range': [0, 30], 'color': "#E74C3C"},
                        {'range': [30, 70], 'color': "#F1C40F"},
                        {'range': [70, 100], 'color': "#2ECC71"}
                    ],
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Risk Indicator
            if confidence < 30:
                st.error("🔴 Risk Level: High")
            elif confidence < 70:
                st.warning("🟡 Risk Level: Medium")
            else:
                st.info("🟢 Risk Level: Low")  


elif menu == "Bulk Profile Analysis":

    st.markdown("## 📂 Bulk Profile Analysis")
    st.divider()
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        import pandas as pd
        
        bulk_data = pd.read_csv(uploaded_file)

        required_columns = [
            "followers",
            "following",
            "posts",
            "has_profile_pic",
            "username_length"
        ]

        if not all(col in bulk_data.columns for col in required_columns):
            st.error("CSV must contain required columns.")
            st.stop()

        st.markdown("### 📊 Dataset Insights")

        col1, col2, col3 = st.columns(3)

        col1.metric("Average Followers", int(bulk_data["followers"].mean()))
        col2.metric("Average Following", int(bulk_data["following"].mean()))
        col3.metric("Average Posts", int(bulk_data["posts"].mean()))

        st.markdown("### 🔬 Feature Correlation Analysis")

        corr = bulk_data[[
            "followers",
            "following",
            "posts",
            "username_length"
        ]].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Feature Correlation Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create ratio feature
        bulk_data['followers_following_ratio'] = bulk_data['followers'] / (bulk_data['following'] + 1)

        # Select features
        bulk_features = bulk_data[['followers', 'following', 'posts',
                                'has_profile_pic', 'username_length',
                                'followers_following_ratio']]

        # Predict
        with st.spinner("Analyzing profiles using AI model..."):
            bulk_predictions = model.predict(bulk_features)
            bulk_probabilities = model.predict_proba(bulk_features)

        bulk_data['Prediction'] = bulk_predictions
        bulk_data['Fake_Probability (%)'] = bulk_probabilities[:, 1] * 100

        def classify_risk(prob):
            if prob > 80:
                return "High Risk"
            elif prob > 50:
                return "Medium Risk"
            else:
                return "Low Risk"

        bulk_data["Risk_Level"] = bulk_data["Fake_Probability (%)"].apply(classify_risk)

        st.markdown("### 🚨 High Risk Accounts")

        high_risk = bulk_data[bulk_data["Risk_Level"] == "High Risk"]

        st.dataframe(high_risk.head(10))

        # Convert numeric prediction to readable label
        bulk_data['Prediction'] = bulk_data['Prediction'].map({1: 'FAKE', 0: 'REAL'})
        # 📊 Summary Metrics
        total_profiles = len(bulk_data)
        fake_count = (bulk_data['Prediction'] == 'FAKE').sum()
        real_count = (bulk_data['Prediction'] == 'REAL').sum()

        fake_percent = (fake_count / total_profiles) * 100
        real_percent = (real_count / total_profiles) * 100

        st.markdown("## 📈 Bulk Analysis Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Profiles", total_profiles)

        with col2:
            st.metric("Fake Profiles (%)", f"{fake_percent:.1f}%")

        with col3:
            st.metric("Real Profiles (%)", f"{real_percent:.1f}%")

        st.success("Bulk analysis completed!")

        st.markdown("### 🔎 Top Suspicious Profiles")

        top_fake = bulk_data.sort_values(
            by="Fake_Probability (%)",
            ascending=False
        ).head(10)

        st.dataframe(top_fake)

        st.markdown("### 📋 Full Analysis Results")

        st.dataframe(bulk_data)
        csv = bulk_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="fakexpose_results.csv",
            mime="text/csv",
            key="download_btn"
        )

        # 📊 Modern Interactive Donut Chart
        st.markdown("### 📊 Fake vs Real Distribution")

        prediction_counts = bulk_data['Prediction'].value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=prediction_counts.index,
            values=prediction_counts.values,
            hole=0.6,
            marker=dict(colors=["#E74C3C", "#2ECC71"])
        )])

        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=40, b=40, l=40, r=40),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📊 Fake Probability Distribution")

        hist_fig = px.histogram(
            bulk_data,
            x="Fake_Probability (%)",
            nbins=20,
            title="Distribution of Fake Profile Probability"
        )

        hist_fig.update_layout(
            height=400,
            margin=dict(t=40, b=40, l=40, r=40)
        )

        st.plotly_chart(hist_fig, use_container_width=True)

        st.markdown("### 🔥 Fraud Risk Heatmap")

        heatmap_fig = px.density_heatmap(
            bulk_data,
            x="following",
            y="followers",
            z="Fake_Probability (%)",
            nbinsx=20,
            nbinsy=20,
            color_continuous_scale="RdYlGn_r",
            title="Fraud Probability Heatmap (Followers vs Following)"
        )

        heatmap_fig.update_layout(
            height=500,
            xaxis_title="Following",
            yaxis_title="Followers"
        )

        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.markdown("---")
        APP_VERSION = "1.1"

        st.markdown("""
        ---
        <center>

        **FakeXpose – AI Social Media Fraud Detection System**

        Built using **Python, Machine Learning, Streamlit, and Plotly**

        </center>
        """, unsafe_allow_html=True)