import streamlit as st
import joblib
import numpy as np
import time
import plotly.graph_objects as go

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

            # 🎯 Confidence Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00BFFF"},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ECC71"},
                        {'range': [30, 70], 'color': "#F1C40F"},
                        {'range': [70, 100], 'color': "#E74C3C"}
                    ],
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # 🧠 Feature Importance Section
            st.markdown("## 🧠 Model Feature Importance")

            feature_names = [
                "Followers",
                "Following",
                "Posts",
                "Has Profile Pic",
                "Username Length",
                "Followers/Following Ratio"
            ]

            try:
                importances = model.feature_importances_

                importance_df = {
                    "Feature": feature_names,
                    "Importance": importances
                }

                import pandas as pd
                importance_df = pd.DataFrame(importance_df)
                importance_df = importance_df.sort_values(by="Importance", ascending=True)

                fig_importance = go.Figure(go.Bar(
                    x=importance_df["Importance"],
                    y=importance_df["Feature"],
                    orientation='h',
                    marker=dict(color="#00BFFF")
                ))

                fig_importance.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Importance Score",
                    yaxis_title="Feature"
                )

                st.plotly_chart(fig_importance, use_container_width=True)

            except:
                st.info("Feature importance not available for this model type.")

            # Risk Indicator
            if confidence < 30:
                st.error("🔴 Risk Level: High")
            elif confidence < 70:
                st.warning("🟡 Risk Level: Medium")
            else:
                st.info("🟢 Risk Level: Low")  


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
