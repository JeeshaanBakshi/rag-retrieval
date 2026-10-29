"""Split a document into overlapping fixed-width character chunks."""


def chunk_text(text: str, size: int = 100, overlap: int = 40) -> list[str]:
    """Return overlapping windows of `size` characters.

    Each new chunk starts `size - overlap` characters after the previous one,
    so an idea cut at one boundary still appears whole in the next chunk.

    >>> chunk_text("abcdefghij", size=4, overlap=2)
    ['abcd', 'cdef', 'efgh', 'ghij']
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")

    step = size - overlap
    chunks = [text[0:size]]
    for i in range(size, len(text), step):
        start = i - overlap
        chunks.append(text[start : start + size])
    return chunks
