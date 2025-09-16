import aiofiles
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FileProcessor:
    """
    Handle file processing for different document types
    """
    
    @staticmethod
    async def extract_text_from_file(file_content: bytes, filename: str) -> str:
        """
        Extract text from uploaded file based on file type
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Returns:
            Extracted text content
        """
        file_extension = Path(filename).suffix.lower()
        
        try:
            if file_extension == '.txt':
                return file_content.decode('utf-8')
            
            elif file_extension == '.pdf':
                # For PDF processing, you'd typically use PyPDF2 or pdfplumber
                # For now, we'll use a simple fallback
                try:
                    return file_content.decode('utf-8', errors='ignore')
                except:
                    return "PDF processing not implemented yet. Please convert to text file."
            
            elif file_extension in ['.docx', '.doc']:
                # For DOCX processing, you'd typically use python-docx
                # For now, we'll use a simple fallback
                try:
                    return file_content.decode('utf-8', errors='ignore')
                except:
                    return "DOCX processing not implemented yet. Please convert to text file."
            
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise
    
    @staticmethod
    async def save_temp_file(content: bytes, suffix: str = ".txt") -> str:
        """
        Save content to temporary file
        
        Args:
            content: File content
            suffix: File extension
            
        Returns:
            Path to temporary file
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            return tmp_file.name
