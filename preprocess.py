import os
import re
import nltk
from pathlib import Path
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

def load_document(file_path):
    """
    Loads text from a PDF document using fitz (PyMuPDF).
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def clean_text(text):
    """
    Cleans up text by removing extra whitespace.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=300):
    """
    Chunks a long text into smaller segments.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Generates semantic embeddings for a list of text chunks.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def save_chunks(chunks, folder='chunks'):
    """
    Saves text chunks to a directory of text files.
    """
    os.makedirs(folder, exist_ok=True)
    # Clear existing chunks
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))
        
    for i, chunk in enumerate(chunks):
        with open(f"{folder}/chunk_{i+1}.txt", 'w', encoding='utf-8') as f:
            f.write(chunk)

def save_vector_db(embeddings, folder='vectordb'):
    """
    Saves the FAISS index to a directory.
    """
    os.makedirs(folder, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, f"{folder}/faiss_index.idx")

def load_chunks(folder_path="chunks"):
    """
    Loads text chunks from a specified folder.
    """
    chunks = []
    try:
        for filename in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks.append(f.read().strip())
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found. Please run `preprocess.py` first.")
        sys.exit(1)
    return chunks
    
def process_document(file_path):
    """
    Main function to process a document and create the RAG data.
    """
    print(f"Starting document processing for {file_path}...")
    text = load_document(file_path)
    clean = clean_text(text)
    chunks = chunk_text(clean, chunk_size=300)
    save_chunks(chunks)
    embeddings = generate_embeddings(chunks)
    save_vector_db(np.array(embeddings).astype("float32"))
    print("Preprocessing complete. Chunks and vector database saved.")
