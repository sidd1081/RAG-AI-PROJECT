import re


def clean_text(text: str) -> str:
    """
    Cleans extracted Markdown text while preserving Markdown syntax
    (table pipes, heading '#', list '-') needed downstream by the
    parser's table/heading detection and by the chunker.
    """
    text = re.sub(r"[^\x20-\x7E\n|#\-]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
