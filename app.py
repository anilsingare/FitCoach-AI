import streamlit as st
from google import genai
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()
st.set_page_config(page_title="FitCoach Pro", page_icon="💪", layout="wide")

# Professional CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button {
        width: 100%; border-radius: 12px; background-color: #00ffcc;
        color: black; font-weight: bold; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #00cca3; transform: scale(1.05); }
    .muscle-card {
        text-align: center; background-color: #1c1f26;
        padding: 15px; border-radius: 15px; border: 1px solid #333;
        margin-bottom: 20px;
    }
    .profile-box {
        background-color: #1c1f26; padding: 20px; border-radius: 15px; border-left: 5px solid #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center; color: #00ffcc;">💪 FITCOACH PRO</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select a muscle group to start your workout</p>", unsafe_allow_html=True)

@st.cache_resource
def get_gemini_client():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API Key not found!")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_gemini_client()
model_id = "gemini-3.6-flash"

@st.cache_data
def load_fitness_assets():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "fitness_data.csv")
        df = pd.read_csv(path, on_bad_lines='skip', sep=None, engine='python').dropna()
        return df
    except:
        return None

df = load_fitness_assets()

def semantic_search(query):
    if df is None: return "No data available."
    mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
    relevant_rows = df[mask]
    return relevant_rows.head(5).to_string(index=False) if not relevant_rows.empty else "No specific data found."

# --- FIXED SESSION STATE ---
# We initialize the values, but we will use 'keys' to update them
if "user_goal" not in st.session_state:
    st.session_state.user_goal = "Muscle Gain"
if "user_level" not in st.session_state:
    st.session_state.user_level = "Beginner"
if "user_weight" not in st.session_state:
    st.session_state.user_weight = "70kg"

st.sidebar.title("⚙️ FitCoach Menu")
page = st.sidebar.radio("Go to:", ["Workout Planner", "My Profile", "Fitness Tips"])

if page == "Workout Planner":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    muscle_images = {
        "Chest": "chest.png", "Back": "back.png", "Shoulders": "shoulders.png",
        "Biceps": "biceps.png", "Triceps": "triceps.png", "Abs": "abs.png",
        "Quads": "quads.png", "Glutes": "glutes.png", "Calves": "calves.png",
    }

    selected_muscle = None
    cols = st.columns(3)
    for i, (muscle, img_name) in enumerate(muscle_images.items()):
        with cols[i % 3]:
            st.markdown('<div class="muscle-card">', unsafe_allow_html=True)
            img_path = os.path.join(base_dir, "workouts", img_name)
            try:
                st.image(img_path, width=100)
            except:
                st.warning("Image missing")
            if st.button(f"Target {muscle}", key=f"btn_{muscle}"):
                selected_muscle = muscle
            st.markdown('</div>', unsafe_allow_html=True)

    if selected_muscle:
        st.markdown(f"## 🎯 Plan for: {selected_muscle}")
        with st.spinner("Designing..."):
            context = semantic_search(selected_muscle)
            # Use the session state keys here
            system_prompt = f"You are FitCoach. User Goal: {st.session_state.user_goal}, Level: {st.session_state.user_level}. Data: {context}. Provide a professional workout plan."
            try:
                response = client.models.generate_content(model=model_id, config={'system_instruction': system_prompt}, contents=f"Create a {selected_muscle} workout.")
                # Typo check: just in case a // slipped in, please delete it!
                st.markdown(response.text)
            except Exception as e:
                st.error(e)

elif page == "My Profile":
    st.markdown("## 👤 User Profile")
    with st.container():
        st.markdown('<div class="profile-box">', unsafe_allow_html=True)
        
        # Use the 'key' parameter to automatically update session_state
        st.selectbox("Goal", ["Muscle Gain", "Weight Loss", "Endurance", "General Fitness"], key="user_goal")
        st.select_slider("Level", options=["Beginner", "Intermediate", "Advanced"], key="user_level")
        st.text_input("Weight (kg)", value=st.session_state.user_weight, key="user_weight")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.success("Profile synced! Your workouts are now personalized.")

elif page == "Fitness Tips":
    st.markdown("## 💡 Pro Fitness Tips")
    tip_category = st.selectbox("Topic", ["Nutrition", "Recovery", "Supplements", "Motivation"])
    if st.button("Get Pro Tip"):
        try:
            # Use the session state keys
            response = client.models.generate_content(model=model_id, contents=f"Give a tip about {tip_category} for a {st.session_state.user_level} athlete.")
            st.info(response.text)
        except Exception as e:
            st.error(e)

if prompt := st.chat_input("Ask FitCoach..."):
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        context = semantic_search(prompt)
        # Use the session state keys
        response = client.models.generate_content(model=model_id, config={'system_instruction': f"You are FitCoach. Data: {context}. Goal: {st.session_state.user_goal}"}, contents=prompt)
        st.markdown(response.text)
