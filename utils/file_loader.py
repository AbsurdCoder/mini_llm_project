"""
Enhanced file loading utilities for Mini LLM project.
Supports multiple file formats: .txt, .html, .pdf, .json
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyPDF2 not installed. PDF support will be disabled.")

try:
    from bs4 import BeautifulSoup
    HTML_SUPPORT = True
except ImportError:
    HTML_SUPPORT = False
    logger.warning("BeautifulSoup not installed. HTML support will be disabled.")

class FileLoader:
    """
    Unified file loader that handles multiple file formats.
    Supports: .txt, .html, .pdf, .json
    """
    
    @staticmethod
    def load_file(file_path: str, encoding: str = 'utf-8') -> str:
        """
        Load content from a file based on its extension.
        
        Args:
            file_path: Path to the file to load
            encoding: Text encoding to use (for text-based files)
            
        Returns:
            String content extracted from the file
            
        Raises:
            ValueError: If file format is unsupported or required dependency is missing
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"Loading file: {file_path} (format: {file_ext})")
        
        if file_ext == '.txt':
            return FileLoader._load_text_file(file_path, encoding)
        elif file_ext == '.html' or file_ext == '.htm':
            return FileLoader._load_html_file(file_path, encoding)
        elif file_ext == '.pdf':
            return FileLoader._load_pdf_file(file_path)
        elif file_ext == '.json':
            return FileLoader._load_json_file(file_path, encoding)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def _load_text_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Load content from a text file."""
        logger.debug(f"Loading text file: {file_path}")
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def _load_html_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Extract text content from an HTML file."""
        if not HTML_SUPPORT:
            raise ValueError("HTML support requires BeautifulSoup. Install with: pip install beautifulsoup4")
        
        logger.debug(f"Loading HTML file: {file_path}")
        with open(file_path, 'r', encoding=encoding) as f:
            html_content = f.read()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    @staticmethod
    def _load_pdf_file(file_path: str) -> str:
        """Extract text content from a PDF file."""
        if not PDF_SUPPORT:
            raise ValueError("PDF support requires PyPDF2. Install with: pip install PyPDF2")
        
        logger.debug(f"Loading PDF file: {file_path}")
        text = ""
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            logger.debug(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text
    
    @staticmethod
    def _load_json_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Extract text content from a JSON file."""
        logger.debug(f"Loading JSON file: {file_path}")
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, str):
            # JSON contains a single string
            return data
        elif isinstance(data, list):
            # JSON contains a list of items
            if all(isinstance(item, str) for item in data):
                # List of strings
                return "\n\n".join(data)
            else:
                # List of objects - convert to string representation
                return "\n\n".join(json.dumps(item, ensure_ascii=False) for item in data)
        elif isinstance(data, dict):
            # JSON contains a dictionary
            if "text" in data and isinstance(data["text"], str):
                # Common format with a "text" field
                return data["text"]
            elif "content" in data and isinstance(data["content"], str):
                # Common format with a "content" field
                return data["content"]
            else:
                # Generic dictionary - convert to string representation
                return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            # Fallback - convert to string representation
            return json.dumps(data, ensure_ascii=False, indent=2)


class DataSplitter:
    """
    Utility for splitting text data into training samples using various strategies.
    """
    
    @staticmethod
    def split_by_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs (separated by double newlines)."""
        return [p.strip() for p in text.split("\n\n") if p.strip()]
    
    @staticmethod
    def split_by_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - not perfect but works for most cases
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def split_by_chunks(text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks of specified size.
        
        Args:
            text: The text to split
            chunk_size: Size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Adjust end to avoid cutting words
            if end < len(text):
                # Find the last space before the end
                while end > start and text[end] != ' ':
                    end -= 1
                
                # If we couldn't find a space, just use the original end
                if end == start:
                    end = min(start + chunk_size, len(text))
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return [c for c in chunks if c.strip()]
    
    @staticmethod
    def split_by_headings(text: str) -> List[str]:
        """
        Split text by headings (lines that look like titles).
        This is a heuristic approach and may not work for all texts.
        """
        import re
        
        # Define patterns for headings
        heading_patterns = [
            r'^#+\s+.+$',  # Markdown headings
            r'^.+\n[=]+$',  # Underlined headings with =
            r'^.+\n[-]+$',  # Underlined headings with -
            r'^[A-Z][^.!?]*$'  # All caps lines (potential headings)
        ]
        
        # Combine patterns
        combined_pattern = '|'.join(f'({p})' for p in heading_patterns)
        
        # Split by headings
        sections = []
        current_section = ""
        
        for line in text.splitlines():
            if re.match(combined_pattern, line):
                # Found a heading
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        # Add the last section
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections
    
    @staticmethod
    def split_by_regex(text: str, pattern: str) -> List[str]:
        """
        Split text using a custom regex pattern.
        
        Args:
            text: The text to split
            pattern: Regex pattern to use for splitting
            
        Returns:
            List of text segments
        """
        import re
        segments = re.split(pattern, text)
        return [s.strip() for s in segments if s.strip()]
