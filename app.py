import streamlit as st
from transformers import pipeline
import io
import fitz
import docx

@st.cache_resource
def load_model(model_name_or_path):
    """Load the Q&A model with caching"""
    try:
        st.info(f"Loading model: {model_name_or_path}")
        qa_pipeline = pipeline("question-answering", model=model_name_or_path)
        st.success("Model loaded successfully!")
        return qa_pipeline
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

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
        
        # Handle .docx files
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = docx.Document(uploaded_file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except Exception as e:
                st.error(f"Could not read the DOCX file: {str(e)}")
                return ""
        
        else:
            st.error(f"Unsupported file type: {file_type}")
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

# Process uploaded file
context = ""
if uploaded_file is not None:
    with st.spinner("Processing document..."):
        context = parse_document(uploaded_file)
        if context:
            st.success(f"Document processed successfully! Extracted {len(context)} characters.")

# Display document preview if available
if context:
    with st.expander("📄 Document Preview (First 1000 characters)"):
        st.text(context[:1000] + ("..." if len(context) > 1000 else ""))

# Sidebar for model configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### Model Selection")

use_tuned_model = st.sidebar.checkbox(
    "Use Fine-tuned Model", 
    value=False,
    help="Toggle between base DistilBERT and fine-tuned model (if available)"
)

# Set model path based on checkbox
if use_tuned_model:
    model_path = "./my-custom-model"
    st.sidebar.info("Using fine-tuned model")
else:
    model_path = "distilbert-base-cased-distilled-squad"
    st.sidebar.info("Using base DistilBERT model")

# Load the model
qa_pipeline = load_model(model_path)

# Question answering interface
if context:
    st.markdown("### 🤔 Ask a Question")
    user_question = st.text_input(
        "What would you like to know about this document?",
        placeholder="e.g., What is the main topic of this document?"
    )
    
    # Get answer button with validation
    if st.button("Get Answer", type="primary", disabled=not qa_pipeline or not user_question.strip()):
        if not qa_pipeline:
            st.error("Model not loaded. Please check the model path.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding answer..."):
                try:
                    result = qa_pipeline(question=user_question, context=context)
                    
                    # Display the answer
                    st.success(f"**Answer:** {result['answer']}")
                    st.info(f"**Confidence:** {result['score']:.4f}")
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")