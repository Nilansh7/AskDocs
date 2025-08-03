## 🎥 Demo GIF</h2>

<p align="center">
  <img src="https://raw.githubusercontent.com/Nilansh7/AskDocs/main/demo.gif" alt="Demo" />
</p>


## Video Source

[Click here to download the full demo video](https://raw.githubusercontent.com/Nilansh7/AskDocs/main/demo.mp4)

# 🤖 AskDocs – RAG-based Document Chatbot

This project is a **chatbot that answers questions based on a specific document (e.g., a PDF file).** It uses a technique called Retrieval-Augmented Generation (RAG) to ensure that its answers are accurate and directly sourced from the document, rather than relying on its general knowledge.
The chatbot has a user-friendly interface built with Streamlit, where you can upload your document and then ask questions.

## Project Architecture and Flow

The chatbot works by following five-step pipeline:
1. **Document Ingestion & Chunking:** The provided PDF document is first loaded and broken down into small, meaningful pieces of text called chunks.
2. **Embedding:** Each text chunk is converted into a numerical vector (a list of numbers) using an embedding model. This allows the computer to understand the meaning of the text.
3. **Vector Database:** These numerical vectors are stored in a special, searchable database called a vector store (using FAISS).
4. **Retrieval:** When you ask a question, the chatbot converts your question into a vector and quickly searches the database to find the most relevant chunk(s) from the original document.
5. **Generation:** The chatbot sends your question and the relevant text chunks to a powerful AI model (LLM). The model's job is to read the chunks and then generate a clear, human-like answer based only on the information it received.

## Technical Choices

**Embedding Model:** all-MiniLM-L6-v2 was chosen because it is an efficient and effective model for creating text embeddings.
**Vector Database:** FAISS was used for its speed and simplicity in performing vector similarity searches.
**LLM (Large Language Model):** Llama 3 (llama3-8b-8192) was selected from the Groq API. This is a powerful, open-source model that provides very fast, free, and stable responses, fulfilling the assignment requirements.

## How to Run the Chatbot

Follow these simple steps to set up and run the application on your local machine.

### Step 1: Set up Your Environment

Install all the required Python libraries:

```bash
pip install streamlit groq sentence-transformers faiss-cpu PyMuPDF python-dotenv
```

### Step 2: Set Your API Key
Create a file named .env in the root of your project folder. In this file, add your Groq API key in the following format:

```bash
GROQ_API_KEY='your_groq_api_key_here'
```
Replace 'your_groq_api_key_here' with your actual Groq API key.

### Step 3: Launch the Application
Run the following command in your terminal to start the Streamlit web application:
```bash
python -m streamlit run app.py
```
A web browser tab will open automatically, and you can start interacting with the chatbot.


## Sample

<p align="center">
  <img 
src="https://raw.githubusercontent.com/Nilansh7/AskDocs/main/Sample_1.jpg"
 alt="Sample Screenshot 1" width="600"/><br><br>

  
<p align="center">
  <img 
src="https://raw.githubusercontent.com/Nilansh7/AskDocs/main/Sample_2.jpg"
 alt="Sample Screenshot 2" width="600"/>
</p>


### Contributing

Got ideas to enhance this project? Found a bug or want to add a new feature?

You're welcome to [open an issue](https://github.com/Nilansh7/AskDocs/issues) or submit a pull request — contributions of all kinds are appreciated!

If you’d like to collaborate, suggest improvements, or need help getting started,  
**feel free to reach out anytime.**

📬 **Let’s connect:**  
📧 **Email:** [s.nilansh07@gmail.com](mailto:s.nilansh07@gmail.com)












