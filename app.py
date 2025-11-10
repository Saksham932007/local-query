import streamlit as st
from transformers import pipeline
import io
import fitz
import docx

def parse_document(uploaded_file):
    """Parse uploaded document and return text content"""
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        # Handle .txt files
        if file_type == "text/plain":
            try:
                text = uploaded_file.read().decode("utf-8")
                return text
            except UnicodeDecodeError:
                st.error("Could not read the text file. Please ensure it's a valid UTF-8 encoded file.")
                return ""
        
        # Handle .pdf files
        elif file_type == "application/pdf":
            try:
                pdf_bytes = uploaded_file.read()
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                for page in pdf_doc:
                    text += page.get_text()
                pdf_doc.close()
                return text
            except Exception as e:
                st.error(f"Could not read the PDF file: {str(e)}")
                return ""
        
        # Implementation for other file types will be added step by step
        return ""
    return ""

st.set_page_config(
    page_title="LocalQuery: Ask Your Docs",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LocalQuery: Ask Your Docs")

uploaded_file = st.file_uploader(
    "Upload a document", 
    type=["pdf", "docx", "txt"],
    help="Upload a PDF, Word document, or text file to ask questions about"
)