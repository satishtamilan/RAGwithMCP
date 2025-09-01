# streamlit_app.py
import streamlit as st
import os
from agentic_rag_mcp import RAGOrchestrator

st.set_page_config(page_title="Agentic PDF RAG", layout="wide")
st.title("Agentic RAG PDF QA")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Multiple PDF upload
uploaded_files = st.file_uploader(
    "Upload PDF files", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="Select one or more PDF files to upload and process"
)

if uploaded_files:
    st.write(f"📁 **{len(uploaded_files)} file(s) selected:**")
    for i, file in enumerate(uploaded_files):
        st.write(f"  {i+1}. {file.name} ({file.size:,} bytes)")
    
    if st.button("🔄 Ingest All PDFs", type="primary"):
        # Save uploaded files temporarily
        temp_files = []
        try:
            with st.spinner(f"Processing and indexing {len(uploaded_files)} PDF(s)…"):
                # Save all files temporarily
                for i, file in enumerate(uploaded_files):
                    temp_filename = f"temp_pdf_{i}.pdf"
                    with open(temp_filename, "wb") as f:
                        f.write(file.read())
                    temp_files.append(temp_filename)
                
                # Initialize RAG and ingest all files
                rag = RAGOrchestrator()
                rag.ingest_multiple(temp_files)
                st.session_state.rag = rag
                st.session_state.uploaded_files = [f.name for f in uploaded_files]
                
            st.success(f"📚 Successfully ingested {len(uploaded_files)} PDF(s)!")
            st.info(f"📊 Total document chunks processed: {len(rag.text_chunks)}")
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

# Display current document status
if st.session_state.uploaded_files:
    st.sidebar.write("📚 **Currently loaded documents:**")
    for i, filename in enumerate(st.session_state.uploaded_files):
        st.sidebar.write(f"  {i+1}. {filename}")
    
    if st.sidebar.button("🗑️ Clear All Documents"):
        st.session_state.rag = None
        st.session_state.uploaded_files = []
        st.rerun()

if st.session_state.rag:
    st.markdown("---")
    st.subheader("💬 Ask Questions About Your Documents")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main topics discussed in these documents?",
        help="The AI will search through all uploaded documents and may also search the web if needed"
    )
    
    if question:
        with st.spinner("🤔 Analyzing documents and generating response…"):
            answer = st.session_state.rag.query(question)
        
        st.markdown("### 📝 **Answer:**")
        st.write(answer)
        
        # Show some stats
        with st.expander("📊 Document Stats"):
            st.write(f"**Total documents loaded:** {len(st.session_state.uploaded_files)}")
            st.write(f"**Total text chunks:** {len(st.session_state.rag.text_chunks)}")
            st.write(f"**Search candidates used:** {st.session_state.rag.n_candidates}")
else:
    st.info("👆 Please upload and ingest PDF files to start asking questions.")