# LocalQuery: Ask Your Docs 🤖📄

**LocalQuery** is a powerful, privacy-focused Question-Answering application that runs entirely on your local machine. Upload documents in multiple formats and get AI-powered answers to your questions - all without your data ever leaving your computer.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20transformers-latest-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### Core Functionality
- 🔒 **100% Local & Private**: Your documents never leave your machine
- 📁 **Multi-Format Support**: PDF, DOCX, and TXT files
- 🧠 **AI-Powered Answers**: Uses state-of-the-art DistilBERT model
- ⚡ **Fast Processing**: Optimized for quick document parsing and inference
- 🎯 **Source Attribution**: Shows exactly where answers were found in your documents
- 📊 **Confidence Scores**: Get reliability metrics for each answer

### Advanced Features
- 🔄 **Model Selection**: Toggle between base and fine-tuned models
- 🚀 **Fine-tuning Capability**: Train custom models on SQuAD dataset
- 💾 **Model Caching**: Intelligent caching for faster subsequent loads
- 🎨 **Beautiful UI**: Clean, intuitive Streamlit interface
- 📈 **Document Preview**: See extracted text before asking questions

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Web interface and user experience |
| **AI/ML** | [🤗 Transformers](https://huggingface.co/transformers/) | Question-answering models |
| **Deep Learning** | [PyTorch](https://pytorch.org/) | Model inference and training |
| **Document Processing** | [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF text extraction |
| **Document Processing** | [python-docx](https://python-docx.readthedocs.io/) | DOCX text extraction |
| **Training Data** | [🤗 Datasets](https://huggingface.co/datasets) | SQuAD dataset for fine-tuning |

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 4GB+ RAM (8GB+ recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saksham932007/local-query.git
   cd local-query
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Basic Question-Answering

1. **Upload a Document**: Click "Browse files" and select a PDF, DOCX, or TXT file
2. **Review Preview**: Check the document preview to ensure proper text extraction
3. **Ask Questions**: Type your question in the text input field
4. **Get Answers**: Click "Get Answer" to receive AI-generated responses with confidence scores

### Example Workflow

```
📁 Upload: "machine_learning_paper.pdf"
❓ Question: "What are the main advantages of transformer models?"
✅ Answer: "Transformer models offer several key advantages including parallel processing capabilities, better handling of long sequences, and superior performance on attention-based tasks..."
📊 Confidence: 0.8542
📍 Source: Found in document at characters 1,245-1,387
```

## 🎯 Advanced Features

### Fine-Tuning Your Own Model

LocalQuery includes a complete training pipeline for creating domain-specific models:

```bash
# Run the training script
python train.py
```

**Training Features:**
- 📚 Automatic SQuAD dataset downloading
- ⚡ Hackathon mode for faster training (reduced dataset)
- 🔧 Configurable hyperparameters
- 💾 Automatic model saving
- 📊 Training progress tracking
- 🛡️ Out-of-memory error handling

**Training Configuration:**
```python
MODEL_CHECKPOINT = "distilbert-base-cased-distilled-squad"
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 2
BATCH_SIZE = 8
HACKATHON_MODE = True  # For faster training
```

### Model Selection

Toggle between models in the sidebar:
- **Base Model**: Pre-trained DistilBERT (faster, general-purpose)
- **Fine-tuned Model**: Your custom-trained model (potentially more accurate for your domain)

## 📁 Project Structure

```
local-query/
├── app.py              # Main Streamlit application
├── train.py            # Model fine-tuning script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── PLAN.md            # Development battle plan
├── .gitignore         # Git ignore rules
└── my-custom-model/   # Fine-tuned model directory (created after training)
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer.json
    └── ...
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom model paths
export CUSTOM_MODEL_PATH="./my-custom-model"
export BASE_MODEL_NAME="distilbert-base-cased-distilled-squad"
```

### Customization Options

**In `app.py`:**
- Modify supported file types
- Adjust confidence score thresholds
- Customize UI appearance
- Add new document processing formats

**In `train.py`:**
- Change base model architecture
- Adjust training hyperparameters
- Modify dataset preprocessing
- Add custom evaluation metrics

## 🎯 Use Cases

### Academic Research
- **Paper Analysis**: Upload research papers and ask about methodologies, results, conclusions
- **Literature Review**: Quick extraction of key information from multiple documents
- **Citation Finding**: Locate specific claims and their sources

### Business Applications
- **Document Analysis**: Process contracts, reports, and proposals
- **Knowledge Management**: Create searchable knowledge bases from internal documents
- **Compliance**: Quick searches through regulatory documents

### Personal Use
- **Book Summaries**: Extract key insights from ebooks and PDFs
- **Study Aid**: Ask questions about textbooks and study materials
- **Research Assistant**: Get quick answers from downloaded articles and papers

## 🛡️ Privacy & Security

- ✅ **Zero Data Transmission**: All processing happens locally
- ✅ **No Internet Required**: Works completely offline after initial model download
- ✅ **No Logging**: Your documents and questions are not stored or transmitted
- ✅ **Open Source**: Full transparency in how your data is handled

## 🚧 Development Roadmap

### Upcoming Features
- [ ] **Multi-language Support**: Support for non-English documents
- [ ] **Batch Processing**: Process multiple documents simultaneously
- [ ] **Export Functionality**: Save Q&A sessions to various formats
- [ ] **Advanced Search**: Semantic search across document collections
- [ ] **API Mode**: REST API for programmatic access

### Model Improvements
- [ ] **Larger Models**: Support for BERT-large and other architectures
- [ ] **Domain-Specific Models**: Pre-trained models for legal, medical, technical domains
- [ ] **Multi-modal Support**: Handle documents with images and tables

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/local-query.git
cd local-query

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/  # If tests are available
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and pre-trained models
- **Streamlit** for the amazing web app framework
- **PyMuPDF & python-docx** for document processing capabilities
- **Stanford NLP** for the SQuAD dataset
- **The open-source community** for making this project possible

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [DistilBERT Model Card](https://huggingface.co/distilbert-base-cased-distilled-squad)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

---

**Built with ❤️ for the hackathon community**

*Made by developers who believe in local-first, privacy-focused AI tools.*