import streamlit as st
from transformers import pipeline
import io

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