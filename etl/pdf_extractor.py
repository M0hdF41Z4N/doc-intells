import pdfplumber
import unicodedata
from typing import List, Dict, Any

class PDFTextExtractor:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def normalize_text(self, text: str) -> str:
        """Normalizes unicode characters and removes excessive whitespace."""
        if not text:
            return ""
        text = unicodedata.normalize('NFKD', text)
        # Replacing common problematic characters if needed, or keeping it simple
        text = " ".join(text.split())
        return text

    def extract_text(self) -> List[Dict[str, Any]]:
        """
        Extracts text from each page.
        Returns a list of dicts: {'page': int, 'text': str, 'type': 'text'}
        """
        extracted_pages = []
        try:
            with pdfplumber.open(self.filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        normalized_text = self.normalize_text(text)
                        if normalized_text:
                            extracted_pages.append({
                                'page': i + 1,
                                'text': normalized_text,
                                'type': 'text'
                            })
        except Exception as e:
            print(f"Error extracting text from {self.filepath}: {e}")
            raise
        return extracted_pages
