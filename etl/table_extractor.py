import camelot
import pandas as pd
from typing import List, Dict, Any

class PDFTableExtractor:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def validate_table(self, df: pd.DataFrame) -> bool:
        """
        Validates the extracted table structure.
        """
        if df.empty:
            return False
        
        # Check for essentially empty rows or columns that might indicate poor extraction
        if df.isnull().all().all():
            return False
            
        return True

    def extract_tables(self) -> List[Dict[str, Any]]:
        """
        Extracts tables from the PDF.
        Returns a list of dicts: {'table_id': str, 'page': int, 'rows': List[Dict], 'type': 'table'}
        """
        extracted_tables = []
        try:
            # Using 'lattice' or 'stream'. 'lattice' is good for tables with explicit lines.
            # Using 'all' pages can be memory intensive, might need to batch in production.
            tables = camelot.read_pdf(self.filepath, pages='all', flavor='lattice')
            
            for i, table in enumerate(tables):
                df = table.df
                if not self.validate_table(df):
                    continue
                
                # Assume first row is header
                headers = df.iloc[0].tolist()
                data_df = df[1:]
                
                rows = []
                for _, row in data_df.iterrows():
                    # Create a dictionary for each row mapping header to value
                    row_data = {}
                    for j, val in enumerate(row):
                        header_key = str(headers[j]).strip() if j < len(headers) and str(headers[j]).strip() else f"col_{j}"
                        row_data[header_key] = str(val).strip()
                    rows.append(row_data)

                extracted_tables.append({
                    'table_id': f"table_{table.page}_{i}",
                    'page': table.page,
                    'rows': rows,
                    'type': 'table'
                })
                
        except Exception as e:
            print(f"Error extracting tables from {self.filepath}: {e}")
            # Depending on strictness, we might want to log this and continue or raise.
        
        return extracted_tables
