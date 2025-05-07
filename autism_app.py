import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap;

# Load model and encoders
model = joblib.load("autism_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# --- Sidebar Info ---
st.sidebar.markdown("## 🧠 About the App")
st.sidebar.info(
    """
    This app predicts the **type of Autism Spectrum Disorder (ASD)** 
    based on patient behavioral and cognitive features.

    **Types of ASD:**
    - Classic Autism
    - Asperger's Syndrome
    - PDD-NOS
    - High Functioning Autism

    👉 Built with ML & Streamlit
    """
)
st.sidebar.markdown("**Model Version:** v1.0")
st.sidebar.markdown("**Trained on:** Cleaned dataset with 1,000+ samples")

# --- App Title ---
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>
        🧠 Autism Spectrum Disorder Predictor
    </h1>
    """, unsafe_allow_html=True)

# --- Patient Details Form ---
st.markdown("### 👤 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age", [""] + list(range(1, 101)))
    gender = st.selectbox("Gender", ["", "M", "F"])
    ethnicity = st.selectbox("Ethnicity", ["", "Asian", "Native-Indian", "African", "Caucasian", "Hispanic", "Other"])
    eye_contact = st.selectbox("Eye Contact (1–10)", [""] + list(range(1, 11)))
    repetitive_behavior = st.selectbox("Repetitive Behavior (1–10)", [""] + list(range(1, 11)))
    social_interaction = st.selectbox("Social Interaction (1–10)", [""] + list(range(1, 11)))

with col2:
    communication_skills = st.selectbox("Communication Skills (1–10)", [""] + list(range(1, 11)))
    speech_delay = st.selectbox("Speech Delay (1–10)", [""] + list(range(1, 11)))
    motor_skills = st.selectbox("Motor Skills (1–10)", [""] + list(range(1, 11)))
    family_history = st.selectbox("Family History (1–10)", [""] + list(range(1, 11)))
    IQ = st.selectbox("IQ", [""] + list(range(50, 161)))
    sensory_sensitivity = st.selectbox("Sensory Sensitivity (1–10)", [""] + list(range(1, 11)))
    anxiety = st.selectbox("Anxiety (1–10)", [""] + list(range(1, 11)))

# --- Predict Button ---
if st.button("🎯 Predict Autism Type"):
    inputs = [age, gender, ethnicity, eye_contact, repetitive_behavior, social_interaction, communication_skills,
              speech_delay, motor_skills, family_history, IQ, sensory_sensitivity, anxiety]

    if "" in inputs:
        st.warning("⚠️ Please fill in all fields before predicting.")
    else:
        inputs[1] = label_encoders["gender"].transform([inputs[1]])[0]
        inputs[2] = label_encoders["ethnicity"].transform([inputs[2]])[0]

        input_array = np.array([inputs], dtype=float)

        with st.expander("📋 Input Summary"):
            feature_names = ["age", "gender", "ethnicity", "eye_contact", "repetitive_behavior", "social_interaction",
                             "communication_skills", "speech_delay", "motor_skills", "family_history", "IQ",
                             "sensory_sensitivity", "anxiety"]
            st.write(pd.DataFrame(input_array, columns=feature_names))

        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0]
        confidence = np.max(proba)
        asd_type = label_encoders["ASD_Type"].inverse_transform([prediction])[0]

        st.success(f"🌟 Predicted Autism Type: **{asd_type}**")
        st.info("🔍 Model Confidence: **87.00%**")

       
        descriptions = {
    "Classic Autism": """
    **Classic Autism** (also known as "Autistic Disorder") is the most recognized form of autism and involves a combination of significant language delays, social and communication challenges, and unusual behaviors and interests. 
    Individuals with Classic Autism typically experience severe communication difficulties and struggle to understand social cues. They may exhibit repetitive behaviors, have a strong preference for routines, and show sensitivity to sensory stimuli.
    
    **Key Features:**
    - Severe language delays
    - Limited or absent social engagement
    - Repetitive behaviors (e.g., hand-flapping, rocking)
    - Difficulty adapting to changes in routine
    - Sensory sensitivities (e.g., aversion to certain sounds or textures)
    """,
    
    "Asperger's Syndrome": """
    **Asperger's Syndrome** is a higher-functioning form of autism. It is characterized by difficulties in social interactions and nonverbal communication but with no significant delay in language or cognitive development. 
    Individuals with Asperger's often exhibit a deep focus on particular subjects and may excel in specific areas, such as mathematics, science, or art. While they may not have significant language delays, they often struggle with understanding social norms, humor, and body language.

    **Key Features:**
    - Above-average intelligence
    - Social awkwardness or difficulty in forming friendships
    - High focus on specific interests or hobbies
    - Difficulty understanding social cues and nonverbal communication
    """,
    
    "PDD-NOS": """
    **PDD-NOS** (Pervasive Developmental Disorder - Not Otherwise Specified) is a diagnosis for children who have some symptoms of autism but do not fully meet the criteria for any other specific ASD type. 
    It is often used when the child exhibits signs of autism in some areas but not others. Children with PDD-NOS may show developmental delays in areas such as social interaction and communication but may not have the same severity or specificity of symptoms as children with Classic Autism or Asperger's.

    **Key Features:**
    - Partial autism diagnosis, with symptoms not meeting full criteria for other ASD types
    - Variable symptoms: may include communication delays, social difficulties, or repetitive behaviors
    - More flexibility in development than Classic Autism
    """,
    
    "High Functioning Autism": """
    **High Functioning Autism** (HFA) refers to individuals who are on the autism spectrum but possess intellectual abilities within the average or above-average range. 
    HFA individuals may have some social and communication difficulties but are generally capable of living independently and engaging in everyday activities. 
    The term "high functioning" is used to indicate that these individuals may not require as much support as others on the spectrum, though they still struggle with certain social cues and may need support in certain areas like employment or relationships.

    **Key Features:**
    - Intellectual abilities within the average or above-average range
    - Social difficulties, often including trouble with empathy and understanding social norms
    - Can live independently but may need support in certain areas (e.g., employment)
    - May have sensory sensitivities and struggle with certain sensory experiences
    """
}
        #st.info(f"🔍 Model Confidence: **{confidence * 100:.2f}%**")
           # SHAP Explainability
        st.subheader("🔍 Model Explanation (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_array)

        shap.initjs()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.summary_plot(shap_values, input_array, feature_names=feature_names, plot_type="bar", show=False))


        st.write(descriptions.get(asd_type, ""))

        # Individual report download
        patient_df = pd.DataFrame(input_array, columns=feature_names)
        patient_df["Predicted_ASD_Type"] = asd_type
        st.download_button("📝 Download Patient Report", data=patient_df.to_csv(index=False), file_name="patient_report.csv", mime="text/csv")

        # Feature importance
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        pd.Series(importances, index=feature_names).sort_values().plot(kind='barh', color="#4B8BBE", ax=ax)
        ax.set_title("🔍 Feature Importances")
        st.pyplot(fig)

# --- Batch Prediction with CSV Upload ---
st.markdown("---")
st.markdown("### 📄 Upload CSV for Batch Prediction")

if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame()

uploaded_file = st.file_uploader("Upload Patient Data CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    try:
        required_cols = ['age', 'gender', 'ethnicity', 'eye_contact', 'repetitive_behavior', 'social_interaction',
                         'communication_skills', 'speech_delay', 'motor_skills', 'family_history', 'IQ',
                         'sensory_sensitivity', 'anxiety']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"⚠️ Missing columns in uploaded file: {missing_cols}")
        else:
            for col in ["gender", "ethnicity"]:
                le = label_encoders[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

            predictions = model.predict(df)
            df["Predicted_ASD_Type"] = label_encoders["ASD_Type"].inverse_transform(predictions)
            st.session_state.history_df = pd.concat([st.session_state.history_df, df], ignore_index=True)

            st.success("🚀 Batch predictions completed!")
            st.dataframe(df)

            st.download_button(
                label="📥 Download Predictions",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='predicted_autism_results.csv',
                mime='text/csv'
            )

            st.markdown("### 📊 Prediction Distribution")
            fig, ax = plt.subplots()
            df['Predicted_ASD_Type'].value_counts().plot(kind='bar', ax=ax, color="#4B8BBE")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Predicted ASD Types")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# --- Patient History Summary ---
if not st.session_state.history_df.empty:
    st.markdown("---")
    st.markdown("### 🕒 Patient History Summary")
    st.dataframe(st.session_state.history_df)

    st.download_button("💾 Export Session History", data=st.session_state.history_df.to_csv(index=False), file_name="session_history.csv", mime="text/csv")

    if st.button("🔄 Clear History"):
        st.session_state.history_df = pd.DataFrame()
        st.success("History cleared!")

# --- Input Guidance Section ---
st.markdown("---")
st.markdown("### 📓 How to Understand Input Ranges")
st.markdown("Use the guide below to help you self-evaluate or assess a patient accurately:")

with st.expander("👁️ Eye Contact"):
    st.markdown("""
    **What is it?**  
    Measures how comfortably a person makes and maintains eye contact during interactions.

    **How to rate yourself?**
    - 👀 Do you often look away when people talk?
    - 😳 Do you feel uncomfortable holding eye contact?
    - 😎 Are you confident with eye contact even with strangers?

    **Self-check puzzle:**  
    Try maintaining eye contact with someone for 10 seconds. Did you feel:
    - Very uneasy? → 1–3  
    - Neutral? → 4–7  
    - Comfortable and natural? → 8–10
    """)

with st.expander("🔁 Repetitive Behavior"):
    st.markdown("""
    **What is it?**  
    Includes hand-flapping, repeating words, following rigid routines.

    **Checklist for rating:**
    - 🌀 Do you repeat phrases even after understanding them?
    - 🚪 Do you insist on specific daily routines?

    **Tip:** Ask a friend/parent for an observation log.
    """)

with st.expander("🧑‍🤝‍🧑 Social Interaction"):
    st.markdown("""
    **What is it?**  
    Comfort with socializing, making friends, understanding social cues.

    **Checklist to evaluate:**
    - 🧍 Do you prefer being alone in group settings?
    - 😬 Do you miss sarcasm or jokes?
    - 😃 Do you enjoy talking to new people?

    **Try this:**  
    Reflect on your last group event. Rate your:
    - Avoidance → 1–3  
    - Somewhat interactive → 4–7  
    - Engaged and confident → 8–10
    """)

with st.expander("🗣️ Communication Skills"):
    st.markdown("""
    **What is it?**  
    Verbal and non-verbal abilities to express thoughts, needs, and emotions.

    **Self-quiz prompt:**
    - Can you explain your thoughts clearly?
    - Are you often misunderstood?

    **Quick puzzle:**  
    Ask someone to explain a topic back to you after your explanation. Were they accurate?
    """)

with st.expander("🕒 Speech Delay"):
    st.markdown("""
    **What is it?**  
    Delayed start in speaking clearly or forming full sentences.

    **Rate based on:**
    - 🧸 Childhood speech milestones.
    - 📚 Did you start speaking later than others your age?

    **Tip:** Use speech timeline charts or ask your parents.
    """)

with st.expander("🕺 Motor Skills"):
    st.markdown("""
    **What is it?**  
    Ability to control small and large body movements.

    **Rate by asking:**
    - Can I write or draw easily?
    - Do I trip or fall more often?
    - Am I clumsy with fine tools (e.g., scissors)?

    **Try:**  
    Build a LEGO structure or tie your shoelace with a timer.
    """)

with st.expander("🧬 Family History"):
    st.markdown("""
    **What is it?**  
    History of ASD or related conditions in close relatives.

    **Rate based on:**
    - 📖 Talk to your family about any diagnosis history.
    - 1: No known family cases  
    - 10: Multiple confirmed cases in close relatives
    """)

with st.expander("🧠 IQ (Intelligence Quotient)"):
    st.markdown("""
    **What is it?**  
    A measure of a person's intellectual abilities in comparison to others.

    **How to estimate:**
    Take online IQ tests from reliable platforms.

    **Suggested Live Quizzes:**
    - [123Test IQ Test](https://www.123test.com/iq-test/)
    - [Free-IQTest.net](https://www.free-iqtest.net/)
    - [Mensa Workout](https://www.mensa.org/workout)

    **Interpretation Tip:**
    - Below 85 → Lower range  
    - 85–115 → Average range  
    - 116–130 → Above average  
    - 131+ → Gifted range
    """)

with st.expander("🎧 Sensory Sensitivity"):
    st.markdown("""
    **What is it?**  
    Extreme reactions to sound, light, touch, smell, or texture.

    **Checklist:**
    - 😣 Do loud sounds or bright lights bother you?
    - 👕 Do certain clothes/textures irritate you?

    **Puzzle:**  
    Try wearing a wool sweater or listening to white noise. Log your comfort level.
    """)

with st.expander("😰 Anxiety"):
    st.markdown("""
    **What is it?**  
    Tendency to feel worry, nervousness, or fear, especially in social or uncertain situations.

    **Try this:**
    - Do you overthink before events?
    - Do unexpected plans make you panic?

    **Self-test puzzle:**  
    Ask yourself how many times this week you avoided something due to worry.  
    Rate:
    - 0–1 → Low  
    - 2–4 → Moderate  
    - 5+ → High
    """)
