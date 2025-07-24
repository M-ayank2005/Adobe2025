import os
import json
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List

# PDF Processing
import PyPDF2
import pdfplumber


@dataclass
class OutlineItem:
    """Represents a single item in the document outline"""
    level: str  # H1, H2, H3, etc.
    text: str
    page: int


@dataclass 
class DocumentStructure:
    """Represents the complete structure of a document"""
    title: str
    outline: List[OutlineItem]


class SimplePDFProcessor:
    """Simplified PDF processor for Challenge 1a"""
    
    def __init__(self):
        # Common heading patterns
        self.heading_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.?\s+[A-Z]',   # Numbered sections
            r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*:?$',  # Title Case
            r'^(CHAPTER|Chapter|Section|SECTION)\s+\d+',  # Chapter/Section
        ]
    
    def extract_text_with_fonts(self, pdf_path: Path):
        """Extract text with font information"""
        text_blocks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Get characters with font info
                    chars = page.chars
                    if not chars:
                        continue
                    
                    # Group characters into lines
                    lines = {}
                    for char in chars:
                        y = round(char['y0'], 1)
                        if y not in lines:
                            lines[y] = []
                        lines[y].append(char)
                    
                    # Process each line
                    for y, line_chars in sorted(lines.items(), reverse=True):
                        if not line_chars:
                            continue
                        
                        text = ''.join(char['text'] for char in line_chars).strip()
                        if not text or len(text) < 3:
                            continue
                        
                        # Get font info
                        font_size = line_chars[0].get('size', 12)
                        font_name = line_chars[0].get('fontname', '').lower()
                        
                        text_blocks.append({
                            'text': text,
                            'page': page_num,
                            'font_size': font_size,
                            'font_name': font_name,
                            'is_bold': 'bold' in font_name,
                            'y_position': y
                        })
        
        except Exception as e:
            print(f"Error extracting with pdfplumber: {e}")
            # Fallback to PyPDF2
            return self.extract_text_fallback(pdf_path)
        
        return text_blocks
    
    def extract_text_fallback(self, pdf_path: Path):
        """Fallback extraction using PyPDF2"""
        text_blocks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 2:
                                text_blocks.append({
                                    'text': line,
                                    'page': page_num,
                                    'font_size': 12,  # Default
                                    'font_name': 'default',
                                    'is_bold': False,
                                    'y_position': 0
                                })
        
        except Exception as e:
            print(f"Error with PyPDF2 fallback: {e}")
        
        return text_blocks
    
    def is_likely_heading(self, text: str, font_size: float, is_bold: bool, avg_font_size: float) -> tuple:
        """Determine if text is likely a heading with improved accuracy"""
        text = text.strip()
        
        # Skip if too short or too long
        if len(text) < 3 or len(text) > 200:
            return False, "length"
        
        # Skip financial/numeric data
        if re.search(r'^\$[\d,]+|^\d+,\d+|^\d+\.\d{2}$', text):
            return False, "financial"
        
        # Skip if mostly punctuation or numbers
        if len(re.sub(r'[^\w\s]', '', text)) < len(text) * 0.5:
            return False, "punctuation"
        
        # Skip page numbers and common footers
        if re.search(r'^page \d+$|^p\.\s*\d+$|^\d+$|copyright|©', text.lower()):
            return False, "page_number"
        
        confidence = 0.0
        
        # Font size analysis - main indicator
        font_size_ratio = font_size / avg_font_size if avg_font_size > 0 else 1
        if font_size_ratio > 1.3:
            confidence += 0.5
        elif font_size_ratio > 1.1:
            confidence += 0.3
        
        # Bold text indicator
        if is_bold:
            confidence += 0.3
        
        # Pattern matching for common heading structures
        heading_patterns = [
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS headings
            r'^\d+\.?\s+[A-Z]',      # Numbered sections (1. Introduction)
            r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*:?$',  # Title Case
            r'^(CHAPTER|Chapter|Section|SECTION|Part|PART)\s+\d+',  # Chapter/Section
            r'^[IVX]+\.\s+[A-Z]',    # Roman numerals
            r'^[A-Z]\.\s+[A-Z]',     # A. Introduction
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                confidence += 0.4
                break
        
        # Additional formatting clues
        if text.isupper() and 3 <= len(text.split()) <= 8:
            confidence += 0.2
        
        if text.endswith(':') and text.count(':') == 1:
            confidence += 0.2
        
        # First word capitalization check
        words = text.split()
        if len(words) > 1 and all(word[0].isupper() for word in words if word.isalpha()):
            confidence += 0.1
        
        # Avoid very long sentences (likely content, not headings)
        if len(words) > 15:
            confidence -= 0.3
        
        return confidence > 0.5, f"confidence_{confidence:.2f}"
    
    def extract_title(self, text_blocks: List[dict]) -> str:
        """Extract document title"""
        if not text_blocks:
            return "Document"
        
        # Look for title in first few blocks
        for block in text_blocks[:10]:
            text = block['text'].strip()
            if len(text) > 10 and len(text) < 100:
                # Skip common non-title patterns
                if not re.search(r'page \d+|copyright|©|\d{4}', text.lower()):
                    return text
        
        return "Document"
    
    def process_single_document(self, pdf_path: Path) -> DocumentStructure:
        """Process a single PDF document"""
        print(f"Processing {pdf_path.name}...")
        
        # Extract text blocks
        text_blocks = self.extract_text_with_fonts(pdf_path)
        
        if not text_blocks:
            return DocumentStructure(title="Document", outline=[])
        
        # Calculate average font size
        font_sizes = [block['font_size'] for block in text_blocks if block['font_size'] > 0]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        
        # Extract title
        title = self.extract_title(text_blocks)
        
        # Find headings
        outline = []
        seen_headings = set()
        
        for block in text_blocks:
            text = block['text'].strip()
            
            # Skip duplicates
            if text.lower() in seen_headings:
                continue
            
            is_heading, reason = self.is_likely_heading(
                text, block['font_size'], block['is_bold'], avg_font_size
            )
            
            if is_heading:
                # Improved heading level determination
                level = "H3"  # Default
                font_size_ratio = block['font_size'] / avg_font_size if avg_font_size > 0 else 1
                
                # Determine level based on multiple factors
                if font_size_ratio > 1.5 or text.isupper() and len(text.split()) <= 4:
                    level = "H1"
                elif font_size_ratio > 1.2 or block['is_bold'] and font_size_ratio > 1.0:
                    level = "H2"
                elif re.match(r'^\d+\.\d+\.\d+', text):  # 1.2.3 style
                    level = "H3"
                elif re.match(r'^\d+\.\d+', text):       # 1.2 style
                    level = "H2"
                elif re.match(r'^\d+\.', text):          # 1. style
                    level = "H1"
                
                outline_item = OutlineItem(
                    level=level,
                    text=text,
                    page=block['page']
                )
                outline.append(outline_item)
                seen_headings.add(text.lower())
        
        print(f"Found {len(outline)} headings")
        return DocumentStructure(title=title, outline=outline)

def process_pdfs():
    # Check if running in Docker container or local development
    if Path("/app/input").exists():
        # Docker container paths
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
    else:
        # Local development paths
        current_dir = Path(__file__).parent
        input_dir = current_dir / "sample_dataset" / "pdfs"
        output_dir = current_dir / "sample_dataset" / "outputs"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the simplified processor
    processor = SimplePDFProcessor()
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file.name}...")
            
            # Process the PDF
            result = processor.process_single_document(pdf_file)
            
            # Convert to the required JSON format
            json_data = {
                "title": result.title,
                "outline": [
                    {
                        "level": item.level,
                        "text": item.text,
                        "page": item.page
                    }
                    for item in result.outline
                ]
            }
            
            # Create output JSON file
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Processed {pdf_file.name} -> {output_file.name} ({len(json_data['outline'])} headings)")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs() 
    print("completed processing pdfs")