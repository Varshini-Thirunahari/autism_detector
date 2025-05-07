import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("autism_synthetic_scaled_final.csv")
df = df.dropna()  # Drop missing values

# Encode categorical features
le_gender = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])

le_ethnicity = LabelEncoder()
df["ethnicity"] = le_ethnicity.fit_transform(df["ethnicity"])

le_asd_type = LabelEncoder()
df["ASD_Type"] = le_asd_type.fit_transform(df["ASD_Type"])

# Save the encoders
label_encoders = {
    "gender": le_gender,
    "ethnicity": le_ethnicity,
    "ASD_Type": le_asd_type
}

# Define features and target
X = df.drop("ASD_Type", axis=1)
y = df["ASD_Type"]

# Save the feature names for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("\n✅ Model Accuracy:", round(accuracy, 2))

# Classification Report
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
feat_importance_df.to_csv("feature_importance.csv", index=False)

# Save model and encoders
joblib.dump(model, "autism_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("\n✅ Model, encoders, and feature names saved successfully.")
