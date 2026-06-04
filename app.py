import streamlit as st
import tempfile
from ingest import load_pdf, chunk_text, embed_texts, store_chunks
from query import embed_query, retrieve_chunks, rerank_chunks, build_prompt, ask

st.title("DocMind - Ask your PDF")

file = st.file_uploader("Upload a PDF file", type="pdf")
question = st.text_input("Ask a question")
clicked = st.button("Ask")

if clicked:
    if file is not None and question:
        # Ingest the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        text = load_pdf(tmp_path)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        store_chunks(chunks,embeddings)
        
        # Query the PDF
        with st.spinner("Thinking..."):
            answer, sources = ask(question)
            
        st.subheader("Answer")
        st.write(answer)
        
        st.subheader("Sources")
        for i, chunk in enumerate(sources):
            with st.expander(f"Source {i+1}"):
                st.write(chunk)