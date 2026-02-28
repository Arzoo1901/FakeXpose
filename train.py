import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("data.csv")

# Create new feature: followers/following ratio
data["ratio"] = data["followers"] / (data["following"] + 1)

# Features
data['followers_following_ratio'] = data['followers'] / (data['following'] + 1)
X = data[['followers', 'following', 'posts','has_profile_pic', 'username_length','followers_following_ratio']]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fakexpose_model.pkl")

print("Model retrained and saved successfully!")