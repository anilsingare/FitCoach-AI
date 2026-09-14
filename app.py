import streamlit as st
from google import genai
import os
from dotenv import load_dotenv
import pandas as pd

# --- 1. Setup & Config ---
load_dotenv()
st.set_page_config(page_title="FitCoach AI", page_icon="💪")
st.title("💪 FitCoach AI (Pro RAG)")
st.markdown("Your Personal Data-Driven Fitness Expert")

# --- 2. SECURE CACHED CLIENT ---
@st.cache_resource
def get_gemini_client():
    try:
        # Try Cloud Secrets (Streamlit Cloud)
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        try:
            # Try .env file (Local Laptop)
            api_key = os.getenv("GOOGLE_API_KEY")
        except:
            api_key = None

    if not api_key:
        st.error("API Key not found! Please check your .env file or Cloud Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_gemini_client()
model_id = "gemini-3.6-flash"

# --- 3. THE DYNAMIC RAG ENGINE ---
@st.cache_data
def load_fitness_df():
    try:
        # --- CLOUD-READY PATHING ---
        # This finds the folder where app.py is located
        base_path = os.path.dirname(__file__) 
        # This joins the folder path with the filename
        file_path = os.path.join(base_path, "fitness_data.csv")
        
        # Read the CSV using the absolute path
        df = pd.read_csv(file_path)
        return df.dropna()
    except Exception as e:
        st.error(f"CSV Error: {e}")
        st.info("Make sure 'fitness_data.csv' is uploaded to the MAIN folder of your GitHub repo.")
        return None

df = load_fitness_df()

def search_fitness_data(query):
    """Searches the CSV for rows that match the user's query."""
    if df is None:
        return "No data available."
    
    query = query.lower()
    # Search for keywords in all columns of the DataFrame
    mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
    relevant_rows = df[mask]
    
    if not relevant_rows.empty:
        return relevant_rows.head(10).to_string(index=False)
    return "No specific data found in the dataset for this query."

# --- 4. Chat Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Interface ---
if prompt := st.chat_input("What's your fitness goal today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # STEP 1: Search the CSV for only relevant data
            relevant_data = search_fitness_data(prompt)
            
            # STEP 2: Create a dynamic system prompt
            dynamic_prompt = f"""
            You are 'FitCoach AI', a certified elite fitness coach.
            
            RELEVANT DATA FROM OUR DATABASE:
            {relevant_data}
            
            RULES:
            1. Use the provided data to answer.
            2. If the data is 'No specific data found', use your general knowledge but tell the user.
            3. Always ask about injuries.
            4. Be professional and motivating.
            """
            
            # STEP 3: Generate content
            response = client.models.generate_content(
                model=model_id,
                config={'system_instruction': dynamic_prompt},
                contents=prompt
            )
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error: {e}")
