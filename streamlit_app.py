"""
LocalRAG AI Knowledge Assistant - Streamlit Cloud Version

A version optimized for Streamlit Cloud deployment with robust error handling.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

# Robust path setup for Streamlit Cloud
def setup_paths():
    """Set up Python paths for reliable imports on Streamlit Cloud."""
    current_dir = Path(__file__).parent.resolve()
    src_path = current_dir / "src"
    core_path = src_path / "core"
    
    # Add all possible paths
    paths_to_add = [str(current_dir), str(src_path), str(core_path)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return current_dir, src_path, core_path

current_dir, src_path, core_path = setup_paths()

# Robust import system for Streamlit Cloud
def import_core_modules():
    """Import core modules with multiple fallback strategies for Streamlit Cloud."""
    modules = {}
    
    # Try different import strategies
    import_strategies = [
        # Strategy 1: Standard imports
        lambda: __import__('src.core.document_processor', fromlist=['DocumentProcessor', 'process_directory']),
        lambda: __import__('src.core.vector_store', fromlist=['FAISSVectorStore']),
        lambda: __import__('src.core.llm_handler', fromlist=['LLMHandler', 'create_llm_handler']),
        
        # Strategy 2: Direct core imports
        lambda: __import__('core.document_processor', fromlist=['DocumentProcessor', 'process_directory']),
        lambda: __import__('core.vector_store', fromlist=['FAISSVectorStore']),
        lambda: __import__('core.llm_handler', fromlist=['LLMHandler', 'create_llm_handler']),
    ]
    
    module_names = [
        'document_processor',
        'vector_store', 
        'llm_handler'
    ]
    
    for i, (module_name, strategy) in enumerate(zip(module_names, import_strategies[::2] + import_strategies[1::2])):
        try:
            module = strategy()
            modules[module_name] = module
            st.sidebar.success(f"✅ {module_name} imported successfully")
        except Exception as e:
            st.sidebar.warning(f"⚠️ {module_name} import attempt {i//2 + 1}: {str(e)[:50]}...")
            if i == len(import_strategies) - 1:  # Last attempt
                st.error(f"""
                ❌ Failed to import {module_name} after all attempts.
                
                Common solutions:
                1. Ensure all source files are in your GitHub repository
                2. Check directory structure matches requirements
                3. Verify requirements.txt includes all dependencies
                4. Try the simplified version for debugging
                """)
                st.stop()
    
    return modules

# Try to import modules
try:
    with st.spinner("Initializing core modules..."):
        modules = import_core_modules()
        document_processor_module = modules['document_processor']
        vector_store_module = modules['vector_store']
        llm_handler_module = modules['llm_handler']
        
        # Extract classes and functions
        DocumentProcessor = getattr(document_processor_module, 'DocumentProcessor', None)
        process_directory = getattr(document_processor_module, 'process_directory', None)
        FAISSVectorStore = getattr(vector_store_module, 'FAISSVectorStore', None)
        LLMHandler = getattr(llm_handler_module, 'LLMHandler', None)
        create_llm_handler = getattr(llm_handler_module, 'create_llm_handler', None)
        
except Exception as e:
    st.error(f"Critical import error: {str(e)}")
    st.info("Using fallback mode with limited functionality...")
    DocumentProcessor = None
    FAISSVectorStore = None
    create_llm_handler = None

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Streamlit configuration
st.set_page_config(
    page_title="LocalRAG AI Knowledge Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/localrag/localrag-ai',
        'Report a bug': 'https://github.com/localrag/localrag-ai/issues',
        'About': """
        # 🔍 LocalRAG AI Knowledge Assistant
        
        **Academic Research Project - Streamlit Cloud Version**
        
        A Retrieval-Augmented Generation (RAG) system optimized for cloud deployment.
        """
    }
)

# Display system information
st.sidebar.subheader("Deployment Info")
st.sidebar.write("Environment:", "Streamlit Cloud")
st.sidebar.write("Python paths:", len(sys.path))

# Check if core modules loaded
if not all([DocumentProcessor, FAISSVectorStore, create_llm_handler]):
    st.warning("""
    ⚠️ Running in limited mode due to import issues.
    
    Full functionality requires successful import of core modules.
    If deploying to Streamlit Cloud, ensure:
    1. All source files are in your repository
    2. Directory structure is correct
    3. requirements.txt includes all dependencies
    """)
    
    # Provide a simple demo interface
    st.title("🔍 LocalRAG AI Knowledge Assistant")
    st.markdown("Academic Research Project - Streamlit Cloud Version")
    
    st.info("""
    ## 🚀 Deployment Status
    
    The app structure is correct, but core modules need to be imported for full functionality.
    
    **Next steps:**
    1. Verify all source files are in your GitHub repository
    2. Check that `src/core/` directory contains all required modules
    3. Ensure `requirements.txt` has all necessary dependencies
    4. Refer to STREAMLIT_TROUBLESHOOTING.md for detailed help
    """)
    
    st.stop()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.llm_handler = None
    st.session_state.document_processor = None

@st.cache_resource
def initialize_system():
    """Initialize the LocalRAG system components."""
    try:
        # Initialize document processor
        document_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        # Initialize vector store
        store_path = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        vector_store = FAISSVectorStore(store_path=store_path, embedding_model=embedding_model)
        
        # Initialize LLM handler
        llm_handler = create_llm_handler(llm_type="auto")
        
        return document_processor, vector_store, llm_handler
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None, None, None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #1f77b4; margin-bottom: 0; font-size: 2.5rem;">
            🔍 LocalRAG AI Knowledge Assistant
        </h1>
        <p style="font-size: 1.2rem; color: #666; margin-top: 5px;">
            Academic Research Project - Streamlit Cloud Version
        </p>
        <hr style="margin: 20px 0; border: 1px solid #e0e0e0;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <p style="font-size: 1rem; color: #555;">
            📁 Upload Research Documents • 🤖 Ask Intelligent Questions • 📊 Get AI-Powered Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing LocalRAG AI Knowledge Assistant..."):
            document_processor, vector_store, llm_handler = initialize_system()
            
            if all([document_processor, vector_store, llm_handler]):
                st.session_state.document_processor = document_processor
                st.session_state.vector_store = vector_store
                st.session_state.llm_handler = llm_handler
                st.session_state.initialized = True
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system. Please check your configuration.")
                return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Dashboard")
        
        # System stats
        if st.session_state.initialized:
            stats = st.session_state.vector_store.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("total_documents", 0))
                st.metric("Vectors", stats.get("total_vectors", 0))
            with col2:
                st.metric("Storage (MB)", stats.get("storage_size_mb", 0))
                model_info = st.session_state.llm_handler.get_model_info()
                model_name = model_info.get('model', 'Unknown')[:15] + "..."
                st.metric("Model", model_name)
            
            st.info(f"🤖 **Provider:** {model_info.get('provider', 'Unknown')}")
            st.info(f"🔤 **Embeddings:** {stats.get('embedding_model', 'Unknown')}")
        else:
            st.warning("System not initialized")
        
        st.divider()
        
        # System Management
        st.subheader("🛠️ System Management")
        
        if st.button("🔄 Refresh Stats", type="secondary"):
            st.rerun()
        
        if st.button("🗑️ Clear Knowledge Base", type="secondary"):
            if st.session_state.get("confirm_clear", False):
                if st.session_state.initialized:
                    st.session_state.vector_store.clear_store()
                    st.success("✅ Knowledge base cleared!")
                    st.rerun()
                st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📁 Document Ingestion",
        "🔍 Intelligent Query",
        "📊 Knowledge Analytics"
    ])
    
    with tab1:
        st.header("📁 Document Ingestion Center")
        st.markdown("*Process and index documents for intelligent analysis*")
        
        if not st.session_state.initialized:
            st.warning("Please wait for system initialization to complete.")
            return
        
        # File upload
        uploaded_files = st.file_uploader(
            "Select files to upload and process",
            type=['pdf', 'txt', 'docx', 'csv', 'xlsx', 'md', 'pptx'],
            accept_multiple_files=True,
            help="Supported: PDF, TXT, DOCX, CSV, XLSX, MD, PPTX"
        )
        
        if uploaded_files:
            st.write(f"**Selected {len(uploaded_files)} files:**")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size:,} bytes)")
            
            if st.button("🚀 Process Files", type="primary"):
                processed_files = []
                failed_files = []
                total_chunks = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    for i, file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"Processing {file.name}...")
                            progress_bar.progress((i + 1) / len(uploaded_files))
                            
                            # Save uploaded file
                            file_path = temp_path / file.name
                            with open(file_path, "wb") as f:
                                f.write(file.getvalue())
                            
                            # Process the file
                            doc_data = st.session_state.document_processor.process_file(file_path)
                            
                            # Add to vector store
                            st.session_state.vector_store.add_documents([doc_data])
                            
                            processed_files.append(file.name)
                            total_chunks += doc_data['num_chunks']
                            
                        except Exception as e:
                            failed_files.append(f"{file.name}: {str(e)}")
                
                status_text.empty()
                progress_bar.empty()
                
                if processed_files:
                    st.success(f"✅ Successfully processed {len(processed_files)} files")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Files Processed", len(processed_files))
                    with col2:
                        st.metric("Chunks Created", total_chunks)
                    
                    with st.expander("✅ Successfully Processed"):
                        for file in processed_files:
                            st.write(f"- {file}")
                
                if failed_files:
                    st.error("❌ Some files failed to process")
                    with st.expander("❌ Failed to Process"):
                        for file in failed_files:
                            st.write(f"- {file}")
    
    with tab2:
        st.header("🔍 Intelligent Query Interface")
        st.markdown("*Ask questions and get AI-powered answers from your knowledge base*")
        
        if not st.session_state.initialized:
            st.warning("Please wait for system initialization to complete.")
            return
        
        # Query input
        query = st.text_area(
            "**Your Question:**",
            placeholder="Ask anything about your documents...",
            height=100,
            help="Enter your question in natural language"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Query Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                k = st.slider("Number of Sources", 1, 10, 5, help="How many relevant documents to retrieve")
                score_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.0, 0.1, help="Minimum similarity score")
            
            with col2:
                temperature = st.slider("Response Creativity", 0.0, 2.0, 1.0, 0.1, help="Higher = more creative")
                max_tokens = st.slider("Max Response Length", 100, 8192, 2000, 100, help="Maximum response tokens")
        
        if query and st.button("🔍 Search & Analyze", type="primary"):
            with st.spinner("🤖 AI is analyzing your question..."):
                try:
                    # Search vector store
                    search_results = st.session_state.vector_store.search(
                        query=query,
                        k=k,
                        score_threshold=score_threshold
                    )
                    
                    # Generate AI answer
                    result = st.session_state.llm_handler.generate_answer(
                        query=query,
                        retrieved_chunks=search_results,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=1.0,
                        reasoning_effort="medium",
                        enable_streaming=True
                    )
                    
                    # Display answer
                    st.subheader("🤖 AI Response")
                    st.markdown(result['answer'])
                    
                    # Display sources
                    if result['sources']:
                        st.subheader(f"📚 Sources ({result['num_sources']})")
                        
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {i}: {source['file_name']} (Score: {source['similarity_score']:.3f})"):
                                st.write(f"**File:** {source['file_name']}")
                                st.write(f"**Path:** {source['file_path']}")
                                st.write(f"**Similarity:** {source['similarity_score']:.3f}")
                                st.write(f"**Chunk ID:** {source['chunk_id']}")
                    
                    # Display metadata
                    with st.expander("🔍 Query Metadata"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**LLM Provider:** {result['llm_type']}")
                            st.write(f"**Model Used:** {result['model_used']}")
                        with col2:
                            st.write(f"**Sources Found:** {result['num_sources']}")
                            st.write(f"**Query:** {query}")
                            
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
    
    with tab3:
        st.header("📊 Knowledge Base Analytics")
        st.markdown("*Comprehensive insights into your document collection*")
        
        if not st.session_state.initialized:
            st.warning("Please wait for system initialization to complete.")
            return
        
        stats = st.session_state.vector_store.get_stats()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Documents", 
                stats.get("total_documents", 0),
                help="Number of documents in knowledge base"
            )
        
        with col2:
            st.metric(
                "Vector Embeddings", 
                stats.get("total_vectors", 0),
                help="Total text chunks indexed"
            )
        
        with col3:
            st.metric(
                "Storage Size", 
                f"{stats.get('storage_size_mb', 0)} MB",
                help="Disk space used by vector store"
            )
        
        with col4:
            avg_chunks = (stats.get("total_vectors", 0) / max(stats.get("total_documents", 1), 1))
            st.metric(
                "Avg Chunks/Doc", 
                f"{avg_chunks:.1f}",
                help="Average chunks per document"
            )
        
        st.divider()
        
        # Document list
        if stats.get("documents"):
            st.subheader("📁 Document Collection")
            
            for i, doc_path in enumerate(stats["documents"], 1):
                doc_name = Path(doc_path).name
                st.write(f"{i}. **{doc_name}**")
                st.caption(f"Path: {doc_path}")
        else:
            st.info("📭 No documents in knowledge base yet. Upload some documents to get started!")
        
        # System information
        st.divider()
        st.subheader("🛠️ System Configuration")
        
        model_info = st.session_state.llm_handler.get_model_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Embedding Model:** {stats.get('embedding_model', 'Unknown')}")
            st.write(f"**LLM Provider:** {model_info.get('provider', 'Unknown')}")
        with col2:
            st.write(f"**Current Model:** {model_info.get('model', 'Unknown')}")
            st.write(f"**System Version:** 2.0.0")

if __name__ == "__main__":
    main()
