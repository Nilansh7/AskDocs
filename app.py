import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys
import streamlit as st
import time
from dotenv import load_dotenv
from groq import Groq

# Import the new processing function from the preprocess.py module
from preprocess import process_document, load_chunks, save_chunks, save_vector_db

# Set a title for the Streamlit app
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("AskMyDocs")
st.markdown("Ask me anything about the provided document.")

@st.cache_resource
def get_embedding_model_and_index():
    """
    Loads the sentence transformer model and the FAISS index.
    """
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        index = faiss.read_index("vectordb/faiss_index.idx")
        return embedding_model, index
    except FileNotFoundError as e:
        st.error(f"Error: Missing RAG data files. Please process a document first.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()

def get_most_similar_chunk(question, chunks, index, embedding_model):
    """
    Finds the most semantically similar chunk to the user's question.
    """
    question_embedding = embedding_model.encode([question])
    D, I = index.search(np.array(question_embedding).astype("float32"), k=1)
    return chunks[I[0][0]]

def ask_groq_stream(question, context, client):
    """
    Generates a streaming response from the Groq API.
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an AI assistant. Use the following context to answer the user's question. "
                f"If you don't know the answer, just say that you don't know, don't try to make up an answer."
                f"Context: {context}"
            ),
        },
        {"role": "user", "content": question},
    ]

    stream = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        stop=None,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# --- Main Application Logic ---

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please set your GROQ_API_KEY in a .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- File Uploader and Preprocessing Section ---
st.markdown("### Step 1: Upload and Process Your Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
process_button = st.button("Process Document")

if uploaded_file and process_button:
    with st.spinner("Processing document..."):
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", "document.pdf")
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        process_document(file_path)
        
        st.session_state.is_processed = True
        st.success("Document processed and RAG data prepared!")
        st.rerun()

if st.session_state.is_processed:
    st.markdown("### Step 2: Start the Chatbot")
 
    embedding_model, index = get_embedding_model_and_index()
    chunks = load_chunks()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = get_most_similar_chunk(prompt, chunks, index, embedding_model)
                
                response_generator = ask_groq_stream(prompt, context, client)
                full_response = st.write_stream(response_generator)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        with st.sidebar:
            st.subheader("Source Document")
            st.write(context)
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("Assignment Details")
        st.markdown(f"**Model in use:** Llama 3 (via Groq API)")
        st.markdown(f"**Indexed Documents:** {len(chunks)} chunks")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
