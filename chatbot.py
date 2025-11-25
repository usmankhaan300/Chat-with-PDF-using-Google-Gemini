import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
import time
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- USER CONFIGURATION ---
# TODO: Paste your Google API Key between the quotes below
GOOGLE_API_KEY = "AIzaSyBGhX_poSOGgM1xc4J2hOouOwL-si7L26Y"

# Set the environment variable automatically
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Splits the extracted text into manageable chunks."""
    # Increased chunk size to 2000 to reduce the total number of chunks (API calls)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generates embeddings and creates a FAISS vector store using Google Embeddings."""
    if not text_chunks:
        st.warning("No text extracted from PDFs. Cannot create vector store.")
        return 
    try:
        # Changed to Google's embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # --- BATCH PROCESSING TO PREVENT 429 ERRORS ---
        # The free tier has strict rate limits. We process in small batches with delays.
        batch_size = 10  # Process 10 chunks at a time
        vector_store = None
        
        # Create a progress bar
        progress_text = "Creating Embeddings. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            
            if vector_store is None:
                # Create the store with the first batch
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                # Add subsequent batches to the existing store
                vector_store.add_texts(batch)
            
            # Update progress
            progress = min((i + batch_size) / len(text_chunks), 1.0)
            my_bar.progress(progress, text=f"Processing batch {i // batch_size + 1} of {(len(text_chunks) + batch_size - 1) // batch_size}")
            
            # PAUSE to respect API rate limits (essential for Free Tier)
            time.sleep(2) 

        # Save the final index
        if vector_store:
            vector_store.save_local("faiss_index")
            my_bar.empty()
            st.success("Processing Complete. FAISS index created.")
            
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()
    return vector_store

def get_conversational_chain():
    """Creates the QA chain using Google Gemini."""
    
    prompt_template = """
    Answer the questions as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context just say, "Answer is not available in the context", don't provide a wrong answer.
    
    Context: \n {context} \n
    Question: \n {question} \n
    
    Answer: 
    """

    # Using Google Gemini Flash model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(x["input_documents"])),
            "question": lambda x: x['question']
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain

def user_input(user_question):
    """Handles user input, retrieves relevant documents, and gets an answer."""
    try:
        # Must match the embedding model used during creation
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Enable dangerous deserialization is required for local pickle files
        new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        response = chain.invoke(
            {"input_documents": docs, "question": user_question}
        )

        st.write("Reply: ", response)
        
    except FileNotFoundError:
        st.error("FAISS index not found. Please upload and process your PDFs first.")
    except Exception as e:
        # Check for dimension mismatch (common when switching from OpenAI to Google)
        if "dimension" in str(e).lower() or "size" in str(e).lower():
            st.error("Error: Dimension mismatch. It looks like the existing 'faiss_index' was created with a different model. Please delete the 'faiss_index' folder and re-process your PDFs.")
        else:
            st.error(f"An error occurred: {e}")

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="🤖")
    st.header("Chat with PDF using Google Gemini 🤖")

    # Simple check to ensure key is not empty or default
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("PASTE"):
        st.warning("⚠️ Please open the `app.py` file and paste your Google API Key in line 11 to continue.")
        st.stop()
    
    user_question = st.text_input("Ask a Question from the PDF file.")

    if user_question:
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process your PDF files first.")
        else: 
            user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF file and click on the submit & process button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Optional: Clean up old index before processing
                    if os.path.exists("faiss_index"):
                        shutil.rmtree("faiss_index")
                        
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text: 
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                    else:
                        st.warning("No text could be extracted from the uploaded PDFs.")
            else: 
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()