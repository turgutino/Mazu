"""Tests for mazu/files/document_extract.py -- generates real
.docx/.pdf/.xlsx/.pptx fixtures in-process with the same libraries the module
itself uses (python-docx, pypdf's writer, openpyxl, python-pptx), so these are
genuine round-trip tests, not fixtures copied from disk.
"""

import io

import pytest

from mazu.files.document_extract import (
    MAX_EXTRACTED_CHARS,
    DocumentExtractionError,
    extract_document_text,
)

# ---------------------------------------------------------------------------
# plain text / code files -- no extraction library involved at all
# ---------------------------------------------------------------------------


def test_plain_text_file_returned_as_is():
    assert extract_document_text("notes.txt", b"hello world") == "hello world"


def test_code_file_returned_as_is():
    content = b"def add(a, b):\n    return a + b\n"
    assert extract_document_text("add.py", content) == content.decode()


def test_extension_matching_is_case_insensitive():
    content = b"hello"
    assert extract_document_text("notes.TXT", content) == "hello"


def test_non_utf8_binary_file_with_unknown_extension_raises():
    # A genuinely binary payload (not valid UTF-8) with no recognized extension
    # must be rejected with a clear error, not silently mangled into replacement
    # characters and sent to the model as if it were real text.
    garbage = bytes([0xFF, 0xFE, 0x00, 0x01, 0x80, 0x81])
    with pytest.raises(DocumentExtractionError, match="not a text file"):
        extract_document_text("mystery.bin", garbage)


def test_truncates_extremely_long_plain_text():
    long_text = "a" * (MAX_EXTRACTED_CHARS + 500)
    result = extract_document_text("big.txt", long_text.encode())
    assert len(result) < len(long_text)
    assert "truncated" in result
    assert result.startswith("a" * 100)


# ---------------------------------------------------------------------------
# .docx
# ---------------------------------------------------------------------------


def _make_docx_bytes(paragraphs: list[str], table_rows: list[list[str]] | None = None) -> bytes:
    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for i, row in enumerate(table_rows):
            for j, cell in enumerate(row):
                table.cell(i, j).text = cell
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_docx_paragraph_text_extracted():
    data = _make_docx_bytes(["First paragraph.", "Second paragraph."])
    result = extract_document_text("report.docx", data)
    assert "First paragraph." in result
    assert "Second paragraph." in result


def test_docx_table_content_extracted():
    data = _make_docx_bytes(["intro"], table_rows=[["Name", "Age"], ["Alice", "30"]])
    result = extract_document_text("report.docx", data)
    assert "Name" in result and "Age" in result
    assert "Alice" in result and "30" in result


def test_corrupted_docx_raises_clean_error():
    with pytest.raises(DocumentExtractionError, match="Could not read"):
        extract_document_text("broken.docx", b"this is not a real docx file")


# ---------------------------------------------------------------------------
# .pdf
# ---------------------------------------------------------------------------


def _make_pdf_bytes(text: str) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    # pypdf's writer has no built-in text-drawing API -- reportlab would be a
    # heavier optional dependency just for test fixtures. Since this test only
    # needs to prove extract_document_text() dispatches to _extract_pdf() and
    # returns SOMETHING well-formed (not that pypdf's own extraction algorithm
    # is correct -- that's pypdf's own test suite's job), a real, valid,
    # zero-text PDF is enough to exercise the real code path end-to-end.
    del page  # unused -- the blank page itself is the fixture
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_extraction_returns_page_markers_for_a_real_pdf():
    data = _make_pdf_bytes("irrelevant -- see helper docstring")
    result = extract_document_text("doc.pdf", data)
    assert "--- page 1 ---" in result


def test_corrupted_pdf_raises_clean_error():
    with pytest.raises(DocumentExtractionError, match="Could not read"):
        extract_document_text("broken.pdf", b"this is not a real pdf file")


# ---------------------------------------------------------------------------
# .xlsx
# ---------------------------------------------------------------------------


def _make_xlsx_bytes(sheet_name: str, rows: list[list]) -> bytes:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for row in rows:
        sheet.append(row)
    buf = io.BytesIO()
    workbook.save(buf)
    return buf.getvalue()


def test_xlsx_sheet_and_cell_values_extracted():
    data = _make_xlsx_bytes("Budget", [["Item", "Cost"], ["Rent", 1200], ["Food", 300]])
    result = extract_document_text("budget.xlsx", data)
    assert "--- sheet: Budget ---" in result
    assert "Item,Cost" in result
    assert "Rent,1200" in result
    assert "Food,300" in result


def test_corrupted_xlsx_raises_clean_error():
    with pytest.raises(DocumentExtractionError, match="Could not read"):
        extract_document_text("broken.xlsx", b"this is not a real xlsx file")


# ---------------------------------------------------------------------------
# .pptx
# ---------------------------------------------------------------------------


def _make_pptx_bytes(title: str, body: str, notes: str | None = None, table_rows: list[list[str]] | None = None) -> bytes:
    import pptx
    from pptx.util import Inches

    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = title
    slide.placeholders[1].text_frame.text = body
    if notes is not None:
        slide.notes_slide.notes_text_frame.text = notes
    if table_rows:
        table_shape = slide.shapes.add_table(
            len(table_rows), len(table_rows[0]), Inches(1), Inches(3), Inches(4), Inches(2)
        )
        for i, row in enumerate(table_rows):
            for j, cell in enumerate(row):
                table_shape.table.cell(i, j).text = cell
    buf = io.BytesIO()
    presentation.save(buf)
    return buf.getvalue()


def test_pptx_title_and_body_extracted():
    data = _make_pptx_bytes("Q1 Results", "Revenue grew 20 percent")
    result = extract_document_text("deck.pptx", data)
    assert "--- slide 1 ---" in result
    assert "Q1 Results" in result
    assert "Revenue grew 20 percent" in result


def test_pptx_speaker_notes_extracted():
    data = _make_pptx_bytes("Title", "Body", notes="Remember to mention the budget cut")
    result = extract_document_text("deck.pptx", data)
    assert "[speaker notes] Remember to mention the budget cut" in result


def test_pptx_table_content_extracted():
    data = _make_pptx_bytes("Title", "Body", table_rows=[["Name", "Age"], ["Bob", "42"]])
    result = extract_document_text("deck.pptx", data)
    assert "Name" in result and "Age" in result
    assert "Bob" in result and "42" in result


def test_pptx_multiple_slides_each_get_their_own_marker():
    import pptx

    presentation = pptx.Presentation()
    for i in range(3):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = f"Slide {i + 1} title"
    buf = io.BytesIO()
    presentation.save(buf)
    result = extract_document_text("deck.pptx", buf.getvalue())
    for i in range(3):
        assert f"--- slide {i + 1} ---" in result
        assert f"Slide {i + 1} title" in result


def test_corrupted_pptx_raises_clean_error():
    with pytest.raises(DocumentExtractionError, match="Could not read"):
        extract_document_text("broken.pptx", b"this is not a real pptx file")
