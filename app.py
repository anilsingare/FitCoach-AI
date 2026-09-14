import os
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
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("API Key not found!")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_gemini_client()
model_id = "gemini-3.6-flash"

# --- 3. THE DYNAMIC RAG ENGINE ---
@st.cache_data
def load_fitness_df():
    try:
        # We return the actual DataFrame, NOT a string
        df = pd.read_csv("fitness_data.csv")
        return df.dropna()
    except Exception as e:
        st.error(f"CSV Error: {e}")
        return None

df = load_fitness_df()

def search_fitness_data(query):
    """Searches the CSV for rows that match the user's query."""
    if df is None:
        return "No data available."
    
    # Convert query to lowercase for better matching
    query = query.lower()
    
    # Search for the query in all columns of the DataFrame
    # This finds any row that contains the keyword
    mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
    relevant_rows = df[mask]
    
    if not relevant_rows.empty:
        # Return only the top 10 matching rows to save tokens
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
            
            # STEP 2: Create a dynamic system prompt for THIS specific question
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
            
            # STEP 3: Send only the relevant data to the AI
            # We use a new chat session for every request to keep the prompt clean
            response = client.models.generate_content(
                model=model_id,
                config={'system_instruction': dynamic_prompt},
                contents=prompt
            )
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error: {e}")
