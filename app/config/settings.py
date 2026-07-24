"""
Central configuration file for the Multi-Document RAG System.
"""

from pathlib import Path

# =====================================================
# APPLICATION SETTINGS
# =====================================================

APP_NAME = "Multi-Document Conversational RAG"
ENVIRONMENT = "development"  # development | production


# =====================================================
# BASE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_INDEX_DIR = DATA_DIR / "vector_index"
IMAGE_DIR = DATA_DIR / "images"


def session_vector_dir(session_id: str) -> Path:
    """
    Every session gets its own index directory under this. Retrieval must
    only ever load/query a session's own directory - never the base
    VECTOR_INDEX_DIR directly - or documents leak across users.
    """
    return VECTOR_INDEX_DIR / session_id


def session_image_dir(session_id: str) -> Path:
    return IMAGE_DIR / session_id


# =====================================================
# DOCUMENT PROCESSING SETTINGS (token-aware chunking)
# =====================================================

CHUNK_TOKEN_SIZE = 400       # target tokens per chunk (embedding-model tokenizer)
CHUNK_TOKEN_OVERLAP = 60     # sentence-level overlap carried into the next chunk
MIN_CHUNK_TOKENS = 15        # discard chunks smaller than this

OCR_FALLBACK_MIN_CHARS = 20  # pages with less native text than this get OCR'd instead

EMBEDDING_MAX_TOKENS = 480    # hard safety cap, under bge-small's 512 limit -
                               # any chunk over this gets force-split rather
                               # than silently truncated by the model


# =====================================================
# VECTOR INDEX SETTINGS (scalability + versioning)
# =====================================================

FAISS_INDEX_TYPE = "hnsw"     # "hnsw" (approximate, scales to large corpora)
                               # or "flat" (exact, fine for small corpora)
FAISS_HNSW_M = 32             # graph connectivity - higher = better recall, more memory
FAISS_HNSW_EF_CONSTRUCTION = 200
FAISS_HNSW_EF_SEARCH = 64     # higher = better recall, slower search

INDEX_SCHEMA_VERSION = 1      # bump whenever the manifest format changes


# =====================================================
# EMBEDDING MODEL SETTINGS
# =====================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"     # cpu | cuda (if available)
EMBEDDING_DIMENSION = 384
NORMALIZE_EMBEDDINGS = True  # required for cosine similarity via inner product


# =====================================================
# VECTOR RETRIEVAL SETTINGS
# =====================================================

TOP_K_RETRIEVAL = 5          # final chunks handed to the LLM
HYBRID_CANDIDATE_K = 20      # candidates pulled from vector + BM25 before reranking
RRF_K = 60                   # reciprocal rank fusion constant


# =====================================================
# RERANKING SETTINGS
# =====================================================

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True


# =====================================================
# IMAGE / VISION SETTINGS
# =====================================================

# Groq vision model. Groq's multimodal lineup changes frequently and this
# model may be marked "preview" - check https://console.groq.com/docs/vision
# before relying on it in production.
VISION_MODEL_NAME = "qwen/qwen3.6-27b"
MIN_IMAGE_DIMENSION_PX = 80   # skip tiny images (icons/logos)
DESCRIBE_IMAGES = True


# =====================================================
# LLM GENERATION SETTINGS
# =====================================================

LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2         # low temperature to reduce hallucinations
MAX_RESPONSE_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer ONLY using the provided document context. "
    "If the answer is not found in the documents, say so clearly."
)


# =====================================================
# CONVERSATION MEMORY SETTINGS
# =====================================================

MAX_CONVERSATION_TURNS = 5
MEMORY_STRATEGY = "window"


# =====================================================
# VALIDATION FLAGS
# =====================================================

STRICT_SOURCE_ATTRIBUTION = True
ALLOW_OUT_OF_CONTEXT_ANSWER = False
