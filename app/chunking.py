"""Text normalisation, hashing and chunking. Hashes are the basis of every dedup decision."""
import hashlib
import html
import re


def strip_html(text: str) -> str:
    """Feeds routinely double-escape their HTML (&amp;lt;p&amp;gt;), so unescape FIRST, then strip tags.
    Only things that look like real tags (a letter right after '<' or '</') are removed, so ordinary
    text such as '5 < 6 and 7 > 3' survives instead of being eaten as a fake tag."""
    text = html.unescape(text or "")
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def normalize_text(text: str) -> str:
    """Case/whitespace-insensitive form used ONLY for hashing, so cosmetic edits don't count as changes."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks
