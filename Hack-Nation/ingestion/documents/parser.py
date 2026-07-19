"""Document parsing into pages of text and tables.

Why an adapter boundary: PDF extraction is the flakiest dependency in the whole
project — the same report parses differently across pdfplumber versions. Pinning
the *contract* (pages, text, tables, page numbers) rather than the library means
the extraction rules and their tests are stable even when the PDF backend is
not, and the test suite runs with no PDF library installed at all.

Page numbers are first-class rather than incidental: every extracted lab value
must be traceable to the page a human can turn to, and a value that cannot be
grounded to a page is dropped (see ``lab_extractor.py``).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

#: Page break marker used by the committed text fixtures.
PAGE_MARKER = re.compile(r"^\s*(?:---\s*)?\[?PAGE\s+(\d+)\]?(?:\s*---)?\s*$", re.IGNORECASE)

TABLE_START = re.compile(r"^\s*\[TABLE\]\s*$", re.IGNORECASE)
TABLE_END = re.compile(r"^\s*\[/TABLE\]\s*$", re.IGNORECASE)


class DocumentTable(BaseModel):
    """A table as rows of cell strings, with the page it was found on."""

    page_number: int
    rows: list[list[str]] = Field(default_factory=list)
    header: list[str] = Field(default_factory=list)


class DocumentPage(BaseModel):
    """One page of a parsed document.

    ``char_offset`` is the page's start position inside
    :attr:`ParsedDocument.text`, so a span found in the whole-document string can
    be resolved back to a page without re-searching.
    """

    page_number: int
    text: str
    char_offset: int = 0
    tables: list[DocumentTable] = Field(default_factory=list)

    def line_spans(self) -> list[tuple[str, int, int]]:
        """Return ``(line, char_start, char_end)`` in document coordinates."""
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for line in self.text.split("\n"):
            start = self.char_offset + cursor
            spans.append((line, start, start + len(line)))
            cursor += len(line) + 1
        return spans


class ParsedDocument(BaseModel):
    """A whole document: pages, tables, and a single character space."""

    document_id: str
    source_path: str | None = None
    source_hash: str | None = None
    pages: list[DocumentPage] = Field(default_factory=list)
    parser: str = "unknown"
    parser_version: str = "0.0.0"

    @property
    def text(self) -> str:
        """The full document text, pages joined by a newline."""
        return "\n".join(page.text for page in self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def tables(self) -> list[DocumentTable]:
        return [table for page in self.pages for table in page.tables]

    def page_for_char(self, char_index: int) -> int | None:
        """Resolve a document character offset to its page number."""
        for page in self.pages:
            if page.char_offset <= char_index < page.char_offset + len(page.text):
                return page.page_number
        return None

    def text_at(self, char_start: int, char_end: int) -> str:
        return self.text[char_start:char_end]


class DocumentParser(ABC):
    """Contract every document backend must satisfy."""

    name: str = "abstract"
    version: str = "0.0.0"

    @abstractmethod
    def parse(self, source: Any, *, document_id: str | None = None) -> ParsedDocument:
        """Parse ``source`` into pages carrying text and tables."""


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _as_existing_path(source: object) -> Path | None:
    """Return ``source`` as a path if it names an existing file, else ``None``.

    Callers legitimately pass raw document text here, and raw text is neither a
    valid path nor safely testable as one: a multi-line or very long string makes
    ``Path.exists()`` raise OSError rather than return False. So the obviously
    non-path cases are rejected first and any remaining OS error is treated as
    "not a path".
    """
    if isinstance(source, Path):
        candidate = source
    elif isinstance(source, str):
        if "\n" in source or len(source) > 4096:
            return None
        candidate = Path(source)
    else:
        return None
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


class TextFixtureParser(DocumentParser):
    """Parses the committed plain-text report fixtures.

    Fixture format — deliberately trivial so the fixtures stay reviewable in a
    diff, which a binary PDF never is::

        [PAGE 1]
        HORMONE PANEL
        Total Testosterone: 78 ng/dL   (Ref: 15 - 70)
        [TABLE]
        Test | Result | Units | Reference Range
        LH   | 12.4   | mIU/mL | 1.9 - 12.5
        [/TABLE]
        [PAGE 2]
        ...

    A document with no ``[PAGE n]`` marker is treated as a single page 1.
    """

    name = "text_fixture"
    version = "1.0.0"

    def parse(self, source: Any, *, document_id: str | None = None) -> ParsedDocument:
        """Parse a path or a raw string into a :class:`ParsedDocument`."""
        path = _as_existing_path(source)
        if path is not None:
            raw = path.read_text()
            doc_id = document_id or path.stem
            source_path: str | None = str(path)
        else:
            raw = str(source)
            doc_id = document_id or "inline"
            source_path = None

        pages = self._to_pages(raw)
        return ParsedDocument(
            document_id=doc_id,
            source_path=source_path,
            pages=pages,
            parser=self.name,
            parser_version=self.version,
        )

    def _to_pages(self, raw: str) -> list[DocumentPage]:
        page_number = 1
        current_lines: list[str] = []
        collected: list[tuple[int, list[str]]] = []

        for line in raw.split("\n"):
            marker = PAGE_MARKER.match(line)
            if marker:
                if current_lines:
                    collected.append((page_number, current_lines))
                page_number = int(marker.group(1))
                current_lines = []
                continue
            current_lines.append(line)
        if current_lines:
            collected.append((page_number, current_lines))
        if not collected:
            collected = [(1, [""])]

        pages: list[DocumentPage] = []
        char_offset = 0
        for number, lines in collected:
            tables, body_lines = self._extract_tables(lines, number)
            text = "\n".join(body_lines)
            pages.append(
                DocumentPage(
                    page_number=number,
                    text=text,
                    char_offset=char_offset,
                    tables=tables,
                )
            )
            char_offset += len(text) + 1
        return pages

    @staticmethod
    def _extract_tables(
        lines: list[str], page_number: int
    ) -> tuple[list[DocumentTable], list[str]]:
        """Pull ``[TABLE]`` blocks out as structured rows.

        The table lines stay in the page text as well. That is intentional: the
        line-oriented extractor and the table extractor must both be able to see
        them, and grounding always resolves against the page text.
        """
        tables: list[DocumentTable] = []
        body: list[str] = []
        rows: list[list[str]] = []
        in_table = False

        for line in lines:
            if TABLE_START.match(line):
                in_table = True
                rows = []
                continue
            if TABLE_END.match(line):
                in_table = False
                if rows:
                    tables.append(
                        DocumentTable(page_number=page_number, header=rows[0], rows=rows[1:])
                    )
                continue
            if in_table:
                rows.append(_split_row(line))
                body.append(line)
            else:
                body.append(line)

        if in_table and rows:  # unterminated block; keep what we saw
            tables.append(DocumentTable(page_number=page_number, header=rows[0], rows=rows[1:]))
        return tables, body


class PdfPlumberParser(DocumentParser):
    """pdfplumber backend, imported lazily so tests never need a PDF library."""

    name = "pdfplumber"
    version = "0.1.0"

    def parse(
        self, source: Any, *, document_id: str | None = None
    ) -> ParsedDocument:  # pragma: no cover - requires optional dependency
        """Parse a PDF path into pages with text and tables."""
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "PdfPlumberParser requires pdfplumber, which is not installed. Install it "
                "with `pip install '.[documents]'`, or use TextFixtureParser for the "
                "committed synthetic reports."
            ) from exc

        path = Path(str(source))
        pages: list[DocumentPage] = []
        char_offset = 0
        with pdfplumber.open(str(path)) as pdf:
            for index, raw_page in enumerate(pdf.pages, start=1):
                text = raw_page.extract_text() or ""
                tables: list[DocumentTable] = []
                for raw_table in raw_page.extract_tables() or []:
                    cleaned = [[(cell or "").strip() for cell in row] for row in raw_table]
                    if cleaned:
                        tables.append(
                            DocumentTable(page_number=index, header=cleaned[0], rows=cleaned[1:])
                        )
                pages.append(
                    DocumentPage(
                        page_number=index,
                        text=text,
                        char_offset=char_offset,
                        tables=tables,
                    )
                )
                char_offset += len(text) + 1

        return ParsedDocument(
            document_id=document_id or path.stem,
            source_path=str(path),
            pages=pages,
            parser=self.name,
            parser_version=self.version,
        )
