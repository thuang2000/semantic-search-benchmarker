"""Document ingestion and token/Markdown chunking with source-span metadata."""
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
import re

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict
    parent_doc: str


@lru_cache(maxsize=1)
def encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def tokens(text: str) -> list[int]:
    return encoding().encode(text, disallowed_special=())


def html_to_markdown(html: str) -> str:
    """Translate headings and preformatted code; preserve other visible text."""
    from bs4 import BeautifulSoup, NavigableString, Tag
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    def render(node) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""
        if node.name == "pre":
            code = node.get_text()
            longest = max((len(x) for x in re.findall(r"`+", code)), default=2)
            fence = "`" * max(3, longest + 1)
            return f"\n\n{fence}\n{code}\n{fence}\n\n"
        if re.fullmatch(r"h[1-6]", node.name):
            return "\n\n" + "#" * int(node.name[1]) + " " + node.get_text(" ", strip=True) + "\n\n"
        body = "".join(render(child) for child in node.children)
        if node.name in {"p", "div", "section", "article", "li", "tr", "br", "table"}:
            return "\n" + body + "\n"
        if node.name in {"td", "th"}:
            return body + " | "
        return body
    return render(soup.body or soup).strip() + "\n"


def load_corpus(folder: Path) -> dict[str, str]:
    docs = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".md", ".html", ".htm"}:
            continue
        raw = path.read_text(encoding="utf-8")
        docs[path.relative_to(folder).as_posix()] = (
            html_to_markdown(raw) if path.suffix.lower() != ".md" else raw
        )
    if not docs:
        raise ValueError("No Markdown or HTML documents found in corpus/.")
    return docs


def sections(text: str) -> list[tuple[int, int, list[str]]]:
    """Find heading ranges, ignoring heading-like lines inside fenced code."""
    starts = [(0, [])]
    hierarchy: list[tuple[int, str]] = []
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if marker:
            mark, rest = marker.groups()
            if fence is None:
                fence = (mark[0], len(mark))
            elif mark[0] == fence[0] and len(mark) >= fence[1] and not rest.strip():
                fence = None
        elif fence is None:
            head = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*$", line)
            if head:
                level = len(head.group(1))
                title = head.group(2).strip().rstrip("#").strip()
                hierarchy = [(n, h) for n, h in hierarchy if n < level]
                hierarchy.append((level, title))
                current = (offset, [h for _, h in hierarchy])
                if starts[-1][0] == offset:
                    starts[-1] = current
                else:
                    starts.append(current)
        offset += len(line)
    return [(start, starts[i + 1][0] if i + 1 < len(starts) else len(text), headers)
            for i, (start, headers) in enumerate(starts) if start < len(text)]


def make_chunk(text: str, source: str, strategy: str, start: int, end: int,
               section_ranges: list, token_ids: list[int] | None = None) -> DocumentChunk:
    content = text[start:end]
    ids = tokens(content) if token_ids is None else token_ids
    headers = [h for a, b, h in section_ranges if a < end and b > start and h]
    return DocumentChunk(
        chunk_id=f"{source}:{strategy}:{start}-{end}",
        content=content,
        parent_doc=source,
        metadata={
            "char_start": start, "char_end": end,
            "line_start": text.count("\n", 0, start) + 1,
            "line_end": text.count("\n", 0, max(start, end - 1)) + 1,
            "header_hierarchy": headers,
            "token_count": len(ids), "token_ids": ids,
            "strategy": strategy,
        },
    )


class FixedSizeChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require 0 <= overlap < chunk size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, source: str) -> list[DocumentChunk]:
        """Slice the original token stream; use valid UTF-8 boundaries.

        Overlap is exact in original token IDs. A Unicode boundary may shorten
        a window by a few tokens; no source bytes are dropped or replaced.
        """
        enc = encoding()
        ids = tokens(text)
        if not ids:
            return []
        raw = text.encode("utf-8")
        byte_offsets = [0]
        for token in ids:
            byte_offsets.append(byte_offsets[-1] + len(enc.decode_single_token_bytes(token)))
        assert byte_offsets[-1] == len(raw)
        # A boundary must not point inside a multi-byte UTF-8 character.
        def safe(i: int) -> bool:
            b = byte_offsets[i]
            return b == len(raw) or (raw[b] & 0xC0) != 0x80

        chunks = []
        ranges = sections(text)
        start = 0
        while start < len(ids):
            end = min(start + self.chunk_size, len(ids))
            if end < len(ids):
                while end > start + self.chunk_overlap and (
                    not safe(end) or not safe(end - self.chunk_overlap)
                ):
                    end -= 1
                if end <= start + self.chunk_overlap:
                    raise ValueError("Unable to split at a UTF-8 boundary; increase chunk size.")
            char_start = len(raw[:byte_offsets[start]].decode("utf-8"))
            char_end = len(raw[:byte_offsets[end]].decode("utf-8"))
            chunks.append(make_chunk(text, source, f"fixed-{self.chunk_size}-{self.chunk_overlap}",
                                     char_start, char_end, ranges, ids[start:end]))
            if end == len(ids):
                break
            start = end - self.chunk_overlap
        return chunks


class StructureAwareChunker:
    def split(self, text: str, source: str) -> list[DocumentChunk]:
        """A full Markdown section is one chunk; code fences stay intact."""
        ranges = sections(text)
        return [make_chunk(text, source, "structural", a, b, ranges)
                for a, b, _ in ranges if text[a:b].strip()]
