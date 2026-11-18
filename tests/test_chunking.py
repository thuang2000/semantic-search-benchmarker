from pathlib import Path
import pytest

from src.chunking import FixedSizeChunker, StructureAwareChunker, html_to_markdown, tokens, sections


def test_overlap_integrity():
    pytest.importorskip("tiktoken")
    text = " ".join(f"word{i}" for i in range(100)) + "\n"
    ids = tokens(text)
    chunks = FixedSizeChunker(24, 6).split(text, "source.md")
    assert len(chunks) > 2
    reconstructed = chunks[0].metadata["token_ids"].copy()
    for chunk, following in zip(chunks, chunks[1:]):
        assert chunk.metadata["token_ids"][-6:] == following.metadata["token_ids"][:6]
        reconstructed.extend(following.metadata["token_ids"][6:])
    assert reconstructed == ids
    for c in chunks:
        assert c.content == text[c.metadata["char_start"]:c.metadata["char_end"]]
        assert c.metadata["token_count"] <= 24


def test_unicode_no_character_loss():
    pytest.importorskip("tiktoken")
    text = "Data café 🚀 中文 — example. " * 30
    chunks = FixedSizeChunker(48, 8).split(text, "unicode.md")
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[-1].metadata["char_end"] == len(text)
    for c, nxt in zip(chunks, chunks[1:]):
        assert c.metadata["token_ids"][-8:] == nxt.metadata["token_ids"][:8]
        assert c.metadata["char_end"] >= nxt.metadata["char_start"]
        assert c.content == text[c.metadata["char_start"]:c.metadata["char_end"]]


def test_markdown_fence_preservation():
    pytest.importorskip("tiktoken")
    text = "# Guide\n\n## First\n```python\n# not a heading\nx = 1\n```\n## Next\nDone\n"
    chunks = StructureAwareChunker().split(text, "doc.md")
    first = [c for c in chunks if "## First" in c.content][0]
    assert "```python\n# not a heading\nx = 1\n```" in first.content
    assert first.metadata["header_hierarchy"] == [["Guide", "First"]]
    assert len(chunks) == 3


def test_html_ingestion_preserves_code_and_headings():
    html = "<h1>Guide</h1><p>Text</p><h2>Example</h2><pre># not a heading\nx = 1</pre><script>hidden</script>"
    text = html_to_markdown(html)
    assert "hidden" not in text
    assert "```\n# not a heading\nx = 1\n```" in text
    assert [h for _, _, h in sections(text)] == [["Guide"], ["Guide", "Example"]]


def test_invalid_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(32, 32)
