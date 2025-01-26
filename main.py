import os
import json
import streamlit as st
from groq import Groq
from typing import List
import re

# Streamlit page configuration
st.set_page_config(
    page_title="Sukhi-Poribar FAQ Bot",
    page_icon="📚",
    layout="centered"
)

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks at sentence boundaries."""
    # Split into sentences (handles both Bengali and English)
    sentences = re.split(r'([।\n]|\.\s)', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Rough estimate of tokens (characters / 3 for Bengali)
        sentence_length = len(sentence) // 3
        
        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def find_relevant_chunks(query: str, chunks: List[str], top_k: int = 2) -> str:
    """Find the most relevant chunks for a given query."""
    # Simple keyword matching (can be improved with better similarity measures)
    query_words = set(query.lower().split())
    chunk_scores = []
    
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        chunk_scores.append((score, chunk))
    
    # Sort by score and get top_k chunks
    relevant_chunks = [chunk for score, chunk in sorted(chunk_scores, reverse=True)[:top_k]]
    return "\n\n".join(relevant_chunks)

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Load and chunk FAQ content
@st.cache_resource
def load_and_chunk_faq():
    try:
        with open(r"book.txt", encoding='utf-8') as file:
            content = file.read()
        return chunk_text(content)
    except FileNotFoundError:
        st.error("FAQ.txt file not found. Please ensure it exists in the working directory.")
        return []

faq_chunks = load_and_chunk_faq()

# Initialize Groq client
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def create_system_prompt(query: str) -> str:
    """Create a system prompt using relevant FAQ chunks."""
    relevant_content = find_relevant_chunks(query, faq_chunks)
    return f"""You are a helpful assistant specialized in answering questions about Bengali family health. 
    Answer based on these relevant FAQ sections:

    {relevant_content}

    Guidelines:
    1. Answer based solely on the provided FAQ content
    2. If the information isn't in the provided sections, say so
    3. Respond in the same language as the user's question (Bengali or English)
    4. Keep responses clear and concise
    5. Stay focused on the specific question asked"""

# UI Elements
st.title("📚 সুখী পরিবার FAQ Bot")
st.markdown("আপনার প্রশ্ন জিজ্ঞাসা করুন। You can ask questions in Bengali or English.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_prompt = st.chat_input("আপনার প্রশ্ন লিখুন... / Type your question...")

if user_prompt:
    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Create system prompt with relevant content
    system_prompt = create_system_prompt(user_prompt)
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Get response from LLM
    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        assistant_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### About This Chatbot")
    st.markdown("""
    This chatbot is designed to answer questions about family health based on the সুখী পরিবার (Sukhi Poribar) FAQ.
    
    You can:
    - Ask questions in Bengali or English
    - Get information about family health topics
    - Receive answers based on the official FAQ content
    """)
    
    # Add debug information in sidebar if needed
    if st.checkbox("Show Debug Info"):
        st.write(f"Number of FAQ chunks: {len(faq_chunks)}")
        if user_prompt:
            st.write("Relevant content length:", len(create_system_prompt(user_prompt)))