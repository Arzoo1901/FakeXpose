import streamlit as st
import joblib
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="FakeXpose AI",
    page_icon="🕵️",
    layout="wide"
)

st.sidebar.title("🕵️ FakeXpose AI")

menu = st.sidebar.radio(
    "Navigation",
    ["Single Profile Analysis", "Bulk Profile Analysis"]
)

# Load trained model safely
try:
    model = joblib.load("fakexpose_model.pkl")
except:
    st.error("❌ Model file not found. Please run train.py first.")
    st.stop()

# User Inputs
if menu == "Single Profile Analysis":
    
    st.title("🕵️ Single Profile Analysis")
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        followers = st.number_input("Number of Followers", min_value=0)
        posts = st.number_input("Number of Posts", min_value=0)
        username_length = st.number_input("Username Length", min_value=1)

    with col2:
        following = st.number_input("Number of Following", min_value=0)
        has_profile_pic = st.selectbox("Has Profile Picture?", [1, 0])
    st.markdown("---")

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

            st.progress(int(confidence))

            # Risk Indicator
            if confidence < 30:
                st.info("🟢 Risk Level: Low")
            elif confidence < 70:
                st.warning("🟡 Risk Level: Medium")
            else:
                st.error("🔴 Risk Level: High")


elif menu == "Bulk Profile Analysis":

    st.title("📂 Bulk Profile Analysis")
    st.divider()
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        import pandas as pd
        
        bulk_data = pd.read_csv(uploaded_file)

        # Create ratio feature
        bulk_data['followers_following_ratio'] = bulk_data['followers'] / (bulk_data['following'] + 1)

        # Select features
        bulk_features = bulk_data[['followers', 'following', 'posts',
                                'has_profile_pic', 'username_length',
                                'followers_following_ratio']]

        # Predict
        bulk_predictions = model.predict(bulk_features)
        bulk_probabilities = model.predict_proba(bulk_features)

        bulk_data['Prediction'] = bulk_predictions
        bulk_data['Fake_Probability (%)'] = bulk_probabilities[:, 1] * 100

        # Convert numeric prediction to readable label
        bulk_data['Prediction'] = bulk_data['Prediction'].map({1: 'FAKE', 0: 'REAL'})

        st.success("Bulk analysis completed!")

        st.dataframe(bulk_data)
        csv = bulk_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="fakexpose_results.csv",
            mime="text/csv",
            key="download_btn"
        )

        import matplotlib.pyplot as plt

        st.markdown("### 📊 Fake vs Real Distribution")

        # Count predictions
        prediction_counts = bulk_data['Prediction'].value_counts()

        # Create pie chart
        fig, ax = plt.subplots()
        ax.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%')
        ax.set_title("Fake vs Real Profiles")

        st.pyplot(fig)