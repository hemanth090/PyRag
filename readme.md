# 🔍 LocalRAG AI Knowledge Assistant - Academic Research Project

**Intelligent Document Analysis and Question Answering System**

A streamlined version of the LocalRAG system optimized for Streamlit Cloud deployment, 
designed for academic research and educational purposes.

## 🎓 Academic Features

- 📁 **Multi-format Document Processing**: PDF, TXT, DOCX, CSV, XLSX, MD, PPTX
- 🤖 **AI-Powered Responses**: State-of-the-art language models for question answering
- 🔍 **Vector Search**: Advanced similarity search using FAISS technology
- 📊 **Research Analytics**: Document collection insights and statistics
- 🌐 **Web Interface**: User-friendly interface for research collaboration

## 🚀 Quick Deployment

1. **Get API Key**
   - Visit [console.groq.com](https://console.groq.com)
   - Sign up for free account
   - Create API key

2. **Deploy to Streamlit Cloud**
   - Fork this repository
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Deploy from your repository
   - Add `GROQ_API_KEY` to app secrets

## 📋 Requirements

All dependencies are in `requirements.txt`:
- streamlit>=1.28.0
- sentence-transformers>=2.2.0
- transformers>=4.35.0
- torch>=2.6.0
- groq>=0.31.0
- faiss-cpu>=1.8.0
- PyPDF2>=3.0.0
- python-docx>=1.1.0
- And more (see requirements.txt)

## 🛠️ Configuration

Add these secrets in Streamlit Cloud:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

## 📚 Usage

1. Upload documents in the "📁 Document Ingestion" tab
2. Ask questions in the "🔍 Intelligent Query" tab
3. View analytics in the "📊 Knowledge Analytics" tab

## 📄 License

MIT License - see LICENSE file for details.
