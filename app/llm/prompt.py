def build_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Strict RAG prompt: model must answer ONLY from context.
    Each source is labeled with its section and content type so the model
    can reason correctly about tables/images instead of treating every
    source as generic prose.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        section = meta.get("section_title") or "Unlabeled section"
        content_type = meta.get("content_type", "text")

        label = f"[Source {i} - Section: \"{section}\" - Type: {content_type}]"
        context_blocks.append(f"{label}\n{chunk['chunk_text']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful AI assistant.

RULES:
- Answer ONLY using the information in the context.
- Do NOT use outside knowledge.
- If a source is a Table, read its rows/columns carefully before answering.
- If a source is an Image, treat its description as a factual account of the image.
- If the answer is not present, say:
  "Answer not found in the provided document."

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt
