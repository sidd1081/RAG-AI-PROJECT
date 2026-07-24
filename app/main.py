import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from app.ui.interface import render_upload_ui, get_session_id
from app.rag.pipeline import ingest_documents, load_existing_index
from app.vectorstore.retriever import HybridRetriever
from app.llm.prompt import build_prompt
from app.llm.generator import AnswerGenerator

session_id = get_session_id()

if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "bm25_store" not in st.session_state:
    st.session_state.bm25_store = None

# Pick up this session's previously-saved index from disk, if any, so a
# restarted session doesn't lose everything already ingested. Scoped to
# session_id - never loads another session's documents.
if st.session_state.vector_store is None:
    loaded_vector_store, loaded_bm25_store = load_existing_index(session_id)
    if loaded_vector_store:
        st.session_state.vector_store = loaded_vector_store
        st.session_state.bm25_store = loaded_bm25_store
        st.session_state.embedded_chunks = loaded_bm25_store.chunks
        st.info(f"Loaded {len(loaded_bm25_store.chunks)} previously indexed chunks for this session.")

uploaded_documents = render_upload_ui()

if uploaded_documents:
    with st.spinner("Parsing, chunking, describing images, and embedding new files..."):
        embedded_chunks, vector_store, bm25_store, stats = ingest_documents(
            uploaded_documents, session_id
        )
        if vector_store is not None:
            st.session_state.embedded_chunks = embedded_chunks
            st.session_state.vector_store = vector_store
            st.session_state.bm25_store = bm25_store

    st.divider()
    st.write("### Ingestion Summary")
    st.write("Total chunks indexed:", len(embedded_chunks))
    if embedded_chunks:
        st.write(
            "Table chunks:",
            sum(1 for c in embedded_chunks if c["metadata"]["content_type"] == "table"),
        )
        st.write(
            "Image chunks:",
            sum(1 for c in embedded_chunks if c["metadata"]["content_type"] == "image"),
        )
    if stats.get("skipped_duplicates"):
        st.info(f"Skipped {stats['skipped_duplicates']} file(s) already indexed for this session.")
    if stats.get("ocr_pages_used"):
        st.info(f"Ran OCR fallback on {stats['ocr_pages_used']} scanned page(s).")
    if stats.get("images_failed"):
        st.warning(
            f"{stats['images_failed']} of {stats.get('images_found', 0)} image(s) "
            "could not be captioned and were left out of the searchable index."
        )

if st.session_state.embedded_chunks:
    st.divider()
    query = st.text_input("Ask a question about the document")

    if query:
        retriever = HybridRetriever(
            st.session_state.vector_store,
            st.session_state.bm25_store,
        )
        results = retriever.retrieve(query)

        st.write("### Retrieved Chunks")
        for res in results:
            meta = res["metadata"]
            st.write(
                f"📄 {meta['source_pdf']} | Page {meta['page_number']} | "
                f"Section: {meta.get('section_title') or 'N/A'} | "
                f"Type: {meta['content_type']}"
            )
            if meta["content_type"] == "image" and meta.get("image_path"):
                st.image(meta["image_path"], width=300)

        prompt = build_prompt(query, results)

        with st.spinner("Generating answer..."):
            generator = AnswerGenerator()
            answer = generator.generate(prompt)

        st.divider()
        st.write("### Final Answer")
        st.write(answer)

        st.write("### Sources")
        sources = set()
        for chunk in results:
            meta = chunk["metadata"]
            sources.add((meta["source_pdf"], meta["page_number"], meta.get("section_title")))
        for pdf, page, section in sources:
            st.write(f"- 📄 {pdf} (Page {page}, Section: {section or 'N/A'})")
