import streamlit as st
from sentence_transformers import SentenceTransformer

from app.config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, NORMALIZE_EMBEDDINGS


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)


class EmbeddingGenerator:
    def __init__(self):
        self.model = load_embedding_model()

    def count_tokens(self, text: str) -> int:
        """
        Token count using the embedding model's own tokenizer, so chunk
        sizing matches what actually gets truncated at embedding time.
        """
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def embed_texts(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )

    def embed_chunks(self, chunks):
        texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        embedded_chunks = []
        for chunk, vector in zip(chunks, embeddings):
            embedded_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "embedding": vector,
                "chunk_text": chunk["chunk_text"],
                "metadata": {
                    "session_id": chunk["session_id"],
                    "source_pdf": chunk["source_pdf"],
                    "doc_hash": chunk["doc_hash"],
                    "page_number": chunk["page_number"],
                    "section_title": chunk.get("section_title"),
                    "content_type": chunk.get("content_type", "text"),
                    "chunk_index": chunk.get("chunk_index"),
                    "total_chunks_in_doc": chunk.get("total_chunks_in_doc"),
                    "image_path": chunk.get("image_path"),
                    "source": chunk.get("source", "native"),
                },
            })

        return embedded_chunks
