"""
utils/extractor.py
Extract text from uploaded PDF or TXT files
"""


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        import fitz
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        return "PyMuPDF not installed. Run: pip install PyMuPDF"
    except Exception as e:
        return f"Error reading PDF: {e}"


def extract_text_from_txt(uploaded_file) -> str:
    try:
        return uploaded_file.read().decode("utf-8").strip()
    except Exception as e:
        return f"Error reading file: {e}"
