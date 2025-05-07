import pandas as pd

# Step 1: Load the new real-world dataset
df_new = pd.read_csv("Autism-Adult_Data.csv")

# Step 2: Create a new DataFrame with the same structure as your original dataset
df_new_cleaned = pd.DataFrame()

df_new_cleaned["age"] = df_new["age"]
df_new_cleaned["gender"] = df_new["gender"]
df_new_cleaned["ethnicity"] = df_new["ethnicity"]

# Convert screening questions (0/1 scores) into relevant features
df_new_cleaned["eye_contact"] = df_new["A1_Score"]
df_new_cleaned["repetitive_behavior"] = df_new["A2_Score"]
df_new_cleaned["social_interaction"] = df_new["A3_Score"]
df_new_cleaned["communication_skills"] = df_new["A4_Score"]
df_new_cleaned["speech_delay"] = df_new["A5_Score"]
df_new_cleaned["motor_skills"] = df_new["A6_Score"]
df_new_cleaned["family_history"] = df_new["austim"].map({"yes": 1, "no": 0})

# Add dummy IQ (you can replace with actual if available later)
df_new_cleaned["IQ"] = 100

df_new_cleaned["sensory_sensitivity"] = df_new["A7_Score"]
df_new_cleaned["anxiety"] = df_new["A8_Score"]

# Convert ASD target to match your label
df_new_cleaned["ASD_Type"] = df_new["Class/ASD"].map({"YES": 1, "NO": 0})

# Step 3: Drop rows with missing values (optional but recommended)
df_new_cleaned = df_new_cleaned.dropna()

# Step 4: Save the cleaned dataset
df_new_cleaned.to_csv("autism_real_world_cleaned.csv", index=False)

print("✅ New dataset cleaned and saved as autism_real_world_cleaned.csv")
print("📊 Shape:", df_new_cleaned.shape)
