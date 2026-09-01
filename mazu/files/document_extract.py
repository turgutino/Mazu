"""Extracts plain text from uploaded documents (Word, PDF, Excel, PowerPoint) so
their content can be injected into a chat message as an ordinary text block. No
provider's canonical message format (see mazu/llm/providers/*) has a native
content-block type for .docx/.xlsx/.pptx, and while Anthropic *does* support a
native base64 PDF "document" block, this module deliberately extracts text
uniformly for every file type instead -- one code path that works identically
across all 4 providers (including DeepSeek/Gemini/OpenAI, which have no native
Office/PDF block at all), rather than a native fast-path for Anthropic-only
PDFs plus a fallback for everything else.

Any file whose extension isn't one of the known binary formats below is treated
as plain UTF-8 text -- this is what makes source code, .txt, .md, .csv, .json,
etc. work with zero extra code: they're already text, so no extraction step is
needed at all.
"""

import io
from pathlib import Path

# ~100k characters is a generous cap (roughly 25-30k tokens) that still keeps a
# single attached document from silently blowing out a turn's context budget --
# matches the same "keep one attachment bounded" reasoning as the image size cap
# in mazu_web/app.py's _validate_chat_images, just for extracted text instead of
# raw bytes.
MAX_EXTRACTED_CHARS = 100_000

# Extensions this module actively parses (as opposed to treating as plain text).
BINARY_DOCUMENT_EXTENSIONS = {".docx", ".pdf", ".xlsx", ".pptx"}


class DocumentExtractionError(ValueError):
    """Raised for a file this module cannot turn into text -- a corrupted
    docx/pdf/xlsx, or a genuinely binary file with none of the three known
    extensions (e.g. a .zip or .exe someone tried to attach). Callers should
    surface `str(exception)` directly; it's already a clean, user-facing message.
    """


def _truncate(text: str) -> str:
    if len(text) <= MAX_EXTRACTED_CHARS:
        return text
    omitted = len(text) - MAX_EXTRACTED_CHARS
    return text[:MAX_EXTRACTED_CHARS] + f"\n\n... [truncated -- {omitted} more characters not shown]"


def _extract_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as e:
        raise DocumentExtractionError(
            "The python-docx package isn't installed. Run `pip install mazu[documents]` "
            "(or `pip install python-docx`) to attach .docx files."
        ) from e

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _extract_pdf(data: bytes) -> str:
    try:
        import pypdf
    except ImportError as e:
        raise DocumentExtractionError(
            "The pypdf package isn't installed. Run `pip install mazu[documents]` "
            "(or `pip install pypdf`) to attach .pdf files."
        ) from e

    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = [f"--- page {i + 1} ---\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages)]
    return "\n\n".join(pages)


def _extract_xlsx(data: bytes) -> str:
    try:
        import openpyxl
    except ImportError as e:
        raise DocumentExtractionError(
            "The openpyxl package isn't installed. Run `pip install mazu[documents]` "
            "(or `pip install openpyxl`) to attach .xlsx files."
        ) from e

    workbook = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    sheets = []
    for sheet in workbook.worksheets:
        rows = [
            ",".join("" if cell is None else str(cell) for cell in row)
            for row in sheet.iter_rows(values_only=True)
        ]
        sheets.append(f"--- sheet: {sheet.title} ---\n" + "\n".join(rows))
    return "\n\n".join(sheets)


def _extract_pptx(data: bytes) -> str:
    try:
        import pptx
    except ImportError as e:
        raise DocumentExtractionError(
            "The python-pptx package isn't installed. Run `pip install mazu[documents]` "
            "(or `pip install python-pptx`) to attach .pptx files."
        ) from e

    presentation = pptx.Presentation(io.BytesIO(data))
    slides = []
    for i, slide in enumerate(presentation.slides):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text:
                parts.append(shape.text_frame.text)
            elif shape.has_table:
                for row in shape.table.rows:
                    parts.append("\t".join(cell.text for cell in row.cells))
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text
            if notes_text:
                parts.append(f"[speaker notes] {notes_text}")
        slides.append(f"--- slide {i + 1} ---\n" + "\n".join(parts))
    return "\n\n".join(slides)


def extract_document_text(filename: str, data: bytes) -> str:
    """Returns extracted plain text for `data`, dispatching on `filename`'s
    extension. Raises DocumentExtractionError (a ValueError subclass) with a
    clean, user-facing message for a corrupted file of a known type, or a
    genuinely binary file of an unknown type -- never for a plain text file,
    which always succeeds via UTF-8 decode with `errors="replace"` (matching
    every other place in this codebase that reads possibly-non-UTF-8 text, e.g.
    mazu/tools/fs.py's read_file).
    """
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".docx":
            text = _extract_docx(data)
        elif ext == ".pdf":
            text = _extract_pdf(data)
        elif ext == ".xlsx":
            text = _extract_xlsx(data)
        elif ext == ".pptx":
            text = _extract_pptx(data)
        else:
            # Strict, not "replace": a genuinely binary file with an unrecognized
            # extension (a .zip, .exe, or an image someone attached as a generic
            # file rather than through the image-upload path) must be rejected
            # with a clear error here, not silently mangled into a wall of
            # replacement characters that then gets sent to the model as if it
            # were real text.
            text = data.decode("utf-8", errors="strict")
    except DocumentExtractionError:
        raise
    except UnicodeDecodeError as e:
        raise DocumentExtractionError(
            f"Could not read '{filename}': not a text file and not a recognized "
            "document format (supported: .docx, .pdf, .xlsx, .pptx, or any UTF-8 text/code file)."
        ) from e
    except Exception as e:
        raise DocumentExtractionError(f"Could not read '{filename}': {e}") from e
    return _truncate(text)
