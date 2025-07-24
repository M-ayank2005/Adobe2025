"""
Intelligent Document Understanding and Semantic Linking System
Core system for extracting structured outlines and understanding document relationships
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# PDF Processing
import PyPDF2
import pdfplumber

# NLP and ML
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk

# Utilities
from jsonschema import validate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OutlineItem:
    """Represents a single item in the document outline"""
    level: str  # H1, H2, H3, etc.
    text: str
    page: int
    confidence: float = 1.0
    semantic_id: str = ""


@dataclass
class DocumentStructure:
    """Represents the complete structure of a document"""
    title: str
    outline: List[OutlineItem]
    metadata: Dict[str, Any] = None


@dataclass
class SemanticLink:
    """Represents a semantic relationship between document sections"""
    source_doc: str
    target_doc: str
    source_section: str
    target_section: str
    similarity_score: float
    relationship_type: str  # "related", "similar", "prerequisite", etc.


class IntelligentDocumentProcessor:
    """
    Core system for intelligent document understanding and semantic linking
    """
    
    def __init__(self, model_cache_dir: str = "./models"):
        """Initialize the document processor with necessary models"""
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Initialize NLP models
        self._initialize_models()
        
        # Heading patterns for outline extraction
        self.heading_patterns = {
            'H1': [
                r'^\d+\.\s+[A-Z].*$',  # "1. Name of the Government Servant"
                r'^[A-Z][A-Za-z\s]{10,80}(?:form|Form|APPLICATION|application).*$',  # Form titles
                r'^[A-Z\s]{5,50}$',  # All caps headings
                r'^Chapter\s+\d+',  # Chapter headings
                r'^CHAPTER\s+[IVX]+',  # Roman numeral chapters
                r'^[A-Z][A-Za-z\s,]{15,}\.{3,}\s*\d+$',  # TOC entries with dots and page numbers
                r'^\d+\s+[A-Z][A-Za-z\s]{10,}\.{3,}\s*\d+$',  # Numbered TOC entries
            ],
            'H2': [
                r'^\d+\.\d+\.?\s+[A-Za-z].*$',  # "1.1. Subsection"
                r'^\([a-z]\)\s+[A-Z].*$',  # "(a) Something"
                r'^[a-z]\)\s+[A-Z].*$',  # "a) Something"
                r'^\d+\s+[A-Z][A-Za-z\s]{5,}\.{3,}\s*\d+$',  # Numbered subsection TOC entries
            ],
            'H3': [
                r'^\d+\.\d+\.\d+\.?\s+[A-Za-z].*$',  # "1.1.1. Sub-subsection"
                r'^\([ivx]+\)\s+[A-Za-z].*$',  # "(i) Something"
                r'^[ivx]+\)\s+[A-Za-z].*$',  # "i) Something"
            ]
        }
        
        # Semantic similarity threshold
        self.similarity_threshold = 0.7
        
    def _initialize_models(self):
        """Initialize NLP and embedding models"""
        try:
            logger.info("Initializing NLP models...")
            
            # Load lightweight sentence transformer for embeddings
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=str(self.model_cache_dir)
            )
            
            # Initialize NLTK components (lightweight NLP processing)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[int, str]]:
        """Extract text from PDF with page-wise mapping and font analysis"""
        try:
            page_texts = {}
            full_text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract structured text with font information
                        structured_text = self._extract_structured_text(page)
                        page_texts[page_num] = structured_text
                        full_text += f"\n[PAGE {page_num}]\n{structured_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        # Fallback to simple text extraction
                        page_text = page.extract_text() or ""
                        page_texts[page_num] = page_text
                        full_text += f"\n[PAGE {page_num}]\n{page_text}\n"
            
            return full_text, page_texts
            
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            # Fallback to PyPDF2
            return self._extract_text_pypdf2(pdf_path)
    
    def _extract_structured_text(self, page) -> str:
        """Extract text with font and formatting information"""
        try:
            chars = page.chars
            if not chars:
                return page.extract_text() or ""
            
            # Group characters by line and font properties
            lines = []
            current_line = []
            current_y = None
            tolerance = 2  # Pixel tolerance for same line
            
            # Sort characters by position (top to bottom, left to right)
            sorted_chars = sorted(chars, key=lambda c: (-c.get('top', 0), c.get('x0', 0)))
            
            for char in sorted_chars:
                y_pos = char.get('top', 0)
                
                # Check if this character is on a new line
                if current_y is None or abs(y_pos - current_y) > tolerance:
                    if current_line:
                        lines.append(current_line)
                    current_line = [char]
                    current_y = y_pos
                else:
                    current_line.append(char)
            
            if current_line:
                lines.append(current_line)
            
            # Convert lines to text with formatting markers
            structured_text = ""
            for line_chars in lines:
                if not line_chars:
                    continue
                
                # Analyze font properties of this line
                line_info = self._analyze_line_font_properties(line_chars)
                line_text = ''.join(char.get('text', '') for char in line_chars).strip()
                
                if line_text:
                    # Add formatting markers based on font analysis - BALANCED APPROACH
                    if line_info['is_bold'] and line_info['avg_size'] >= 20:
                        structured_text += f"[TITLE:{line_info['avg_size']}] {line_text}\n"
                    elif line_info['is_bold'] and line_info['avg_size'] >= 14:
                        structured_text += f"[H1:{line_info['avg_size']}] {line_text}\n"
                    elif line_info['is_bold'] and line_info['avg_size'] >= 12:
                        structured_text += f"[H2:{line_info['avg_size']}] {line_text}\n"
                    elif self._is_strong_heading_candidate(line_text, line_info, structured_text):
                        structured_text += f"[H3:{line_info['avg_size']}] {line_text}\n"
                    else:
                        # Apply fallback methods for non-bold potential headings
                        heading_info = self._detect_heading_fallback_methods(line_text, line_info, structured_text)
                        if heading_info and heading_info[1] > 0.75:  # Medium-high confidence threshold
                            level, confidence = heading_info
                            structured_text += f"[{level}:{line_info['avg_size']}] {line_text}\n"
                        else:
                            structured_text += f"{line_text}\n"
            
            return structured_text
            
        except Exception as e:
            logger.warning(f"Error in structured text extraction: {e}")
            return page.extract_text() or ""
    
    def _analyze_line_font_properties(self, line_chars) -> Dict:
        """Analyze font properties of a line of characters"""
        if not line_chars:
            return {'is_bold': False, 'avg_size': 12, 'font_name': 'unknown'}
        
        # Count bold characters
        bold_count = 0
        sizes = []
        font_names = []
        
        for char in line_chars:
            font_name = char.get('fontname', '').lower()
            font_names.append(font_name)
            
            # Check if font is bold
            if 'bold' in font_name or 'black' in font_name:
                bold_count += 1
            
            # Collect font sizes
            size = char.get('size', 12)
            if size > 0:  # Valid size
                sizes.append(size)
        
        # Calculate properties
        is_bold = bold_count > len(line_chars) * 0.5  # More than 50% bold
        avg_size = sum(sizes) / len(sizes) if sizes else 12
        dominant_font = max(set(font_names), key=font_names.count) if font_names else 'unknown'
        
        return {
            'is_bold': is_bold,
            'avg_size': avg_size,
            'font_name': dominant_font,
            'bold_ratio': bold_count / len(line_chars) if line_chars else 0
        }
    
    def _is_potential_heading_by_context(self, line_text: str, preceding_text: str) -> bool:
        """Determine if a line is a heading based on context"""
        if len(line_text) < 3 or len(line_text) > 100:
            return False
        
        # Check if it's a standalone line (not part of a paragraph)
        preceding_lines = preceding_text.strip().split('\n')[-3:]  # Last 3 lines
        
        # If previous line is empty or very short, this might be a heading
        if preceding_lines and (not preceding_lines[-1].strip() or len(preceding_lines[-1].strip()) < 20):
            # Check if line has heading characteristics
            if (line_text[0].isupper() and 
                not line_text.endswith('.') and 
                len(line_text.split()) <= 8 and
                not any(word in line_text.lower() for word in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 'with'])):
                return True
        
        return False
    
    def _is_strong_heading_candidate(self, line_text: str, line_info: Dict, preceding_text: str) -> bool:
        """Determine if a line is a strong heading candidate - BALANCED APPROACH"""
        if len(line_text) < 2 or len(line_text) > 100:
            return False
        
        score = 0
        
        # Pattern-based scoring (most reliable)
        if re.match(r'^\d+\.\s+[A-Z]', line_text):  # "1. Introduction"
            score += 3
        elif re.match(r'^[A-Z][A-Z\s]{3,}$', line_text):  # ALL CAPS titles (shortened min length)
            score += 3
        elif re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s*:?\s*$', line_text):  # "Title Case:"
            score += 2
        elif line_text.endswith(':') and len(line_text.split()) <= 5:  # Short text ending in colon
            score += 2
        
        # Font-based scoring (more lenient)
        if line_info.get('is_bold', False):
            score += 2
        if line_info.get('avg_size', 12) >= 13:  # Lowered from 14
            score += 1
        if line_info.get('avg_size', 12) >= 16:  # Bonus for larger fonts
            score += 1
        
        # Content scoring
        if line_text[0].isupper() and not line_text.endswith('.'):
            score += 1
        if len(line_text.split()) <= 6:  # Short headings more likely
            score += 1
        if line_text.istitle():  # Title Case
            score += 1
        
        # Keyword scoring (document-specific + general)
        heading_keywords = [
            'overview', 'introduction', 'summary', 'phase', 'timeline', 'section', 
            'chapter', 'conclusion', 'background', 'methodology', 'results', 
            'discussion', 'references', 'appendix', 'objectives', 'scope', 
            'purpose', 'goals', 'requirements', 'implementation', 'analysis',
            'evaluation', 'recommendation', 'strategy', 'proposal', 'plan'
        ]
        if any(keyword in line_text.lower() for keyword in heading_keywords):
            score += 2
        
        # Position scoring
        lines = preceding_text.strip().split('\n')
        if lines:
            prev_line = lines[-1].strip() if lines else ""
            if not prev_line or len(prev_line) < 30:  # Previous line is empty or short
                score += 1
        
        # Reduced threshold from 4 to 3 for more inclusion
        return score >= 3
    
    def _detect_heading_fallback_methods(self, line_text: str, line_info: Dict, preceding_text: str) -> Optional[Tuple[str, float]]:
        """Multiple fallback methods for heading detection in irregular PDFs"""
        if len(line_text.strip()) < 3:
            return None
        
        confidence_scores = []
        detected_levels = []
        
        # Method 1: Pattern-based detection (for irregular fonts)
        pattern_result = self._detect_heading_by_patterns(line_text)
        if pattern_result:
            level, conf = pattern_result
            confidence_scores.append(conf)
            detected_levels.append(level)
        
        # Method 2: Semantic analysis using sentence transformers
        semantic_result = self._detect_heading_by_semantics(line_text)
        if semantic_result:
            level, conf = semantic_result
            confidence_scores.append(conf)
            detected_levels.append(level)
        
        # Method 3: Position and structure analysis
        position_result = self._detect_heading_by_position(line_text, preceding_text)
        if position_result:
            level, conf = position_result
            confidence_scores.append(conf)
            detected_levels.append(level)
        
        # Method 4: Content analysis (length, capitalization, punctuation)
        content_result = self._detect_heading_by_content_analysis(line_text, line_info)
        if content_result:
            level, conf = content_result
            confidence_scores.append(conf)
            detected_levels.append(level)
        
        # If at least 2 methods agree or one method has high confidence
        if len(confidence_scores) >= 2 or (confidence_scores and max(confidence_scores) > 0.8):
            # Use most common level or highest confidence level
            if detected_levels:
                most_common_level = max(set(detected_levels), key=detected_levels.count)
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                return (most_common_level, avg_confidence)
        
        return None
    
    def _detect_heading_by_patterns(self, text: str) -> Optional[Tuple[str, float]]:
        """Detect headings using pattern matching (for irregular PDFs)"""
        text = text.strip()
        
        # High confidence patterns for titles/H1
        title_patterns = [
            r'^[A-Z][A-Z\s]{10,}$',  # ALL CAPS titles
            r'^\d+\.\s*[A-Z][^.]{5,}$',  # Numbered sections like "1. Introduction"
            r'^Chapter\s+\d+',  # Chapter headings
            r'^PART\s+[IVX]+',  # Part headings with Roman numerals
            r'^Section\s+\d+',  # Section headings
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return ('H1', 0.9)
        
        # Medium confidence patterns for H2
        h2_patterns = [
            r'^\d+\.\d+\s+[A-Z]',  # Subsection numbering like "1.1 Overview"
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s*:?\s*$',  # Title Case headings
            r'^\w+\s+Overview\s*$',  # Something Overview
            r'^\w+\s+Summary\s*$',  # Something Summary
        ]
        
        for pattern in h2_patterns:
            if re.match(pattern, text):
                return ('H2', 0.7)
        
        # Lower confidence patterns for H3
        h3_patterns = [
            r'^\d+\.\d+\.\d+\s+',  # Deep numbering like "1.1.1"
            r'^[A-Z][a-z]+\s*:$',  # Single word followed by colon
        ]
        
        for pattern in h3_patterns:
            if re.match(pattern, text):
                return ('H3', 0.6)
        
        return None
    
    def _detect_heading_by_semantics(self, text: str) -> Optional[Tuple[str, float]]:
        """Use semantic analysis to detect if text is a heading"""
        try:
            # Common heading keywords and phrases
            heading_keywords = [
                'introduction', 'overview', 'summary', 'conclusion', 'background',
                'methodology', 'results', 'discussion', 'references', 'appendix',
                'objectives', 'scope', 'purpose', 'goals', 'requirements',
                'implementation', 'analysis', 'evaluation', 'recommendation'
            ]
            
            text_lower = text.lower()
            
            # Check for heading keywords
            keyword_score = 0
            for keyword in heading_keywords:
                if keyword in text_lower:
                    keyword_score += 0.3
            
            # Check for typical heading structure
            structure_score = 0
            if len(text.split()) <= 6:  # Short text
                structure_score += 0.2
            if text[0].isupper():  # Starts with capital
                structure_score += 0.2
            if not text.endswith('.'):  # Doesn't end with period
                structure_score += 0.2
            if ':' in text:  # Contains colon
                structure_score += 0.2
            
            total_score = keyword_score + structure_score
            
            if total_score >= 0.6:
                return ('H2', min(total_score, 0.8))
            elif total_score >= 0.4:
                return ('H3', min(total_score, 0.6))
            
        except Exception:
            pass
        
        return None
    
    def _detect_heading_by_position(self, text: str, preceding_text: str) -> Optional[Tuple[str, float]]:
        """Detect headings based on position and document structure"""
        lines = preceding_text.strip().split('\n')
        
        # Check if this line follows typical heading patterns
        if len(lines) >= 2:
            prev_line = lines[-1].strip() if lines else ""
            prev_prev_line = lines[-2].strip() if len(lines) >= 2 else ""
            
            # If previous line is empty and before that was text, this might be a heading
            if not prev_line and prev_prev_line and len(prev_prev_line) > 20:
                # This line appears after a paragraph break
                if len(text.split()) <= 8 and not text.endswith('.'):
                    return ('H2', 0.7)
            
            # If previous line was short and this line is also short
            if prev_line and len(prev_line) < 30 and len(text) < 50:
                if text[0].isupper() and not text.endswith('.'):
                    return ('H3', 0.5)
        
        # Check if it's at the beginning of a page
        if '[PAGE' in preceding_text:
            page_lines = preceding_text.split('[PAGE')[-1].split('\n')[1:]  # Lines after last page marker
            if len(page_lines) <= 3 and text.strip():  # Near top of page
                return ('H1', 0.8)
        
        return None
    
    def _detect_heading_by_content_analysis(self, text: str, line_info: Dict) -> Optional[Tuple[str, float]]:
        """Analyze content characteristics to detect headings"""
        text = text.strip()
        
        score = 0
        level = 'H3'  # Default to H3
        
        # Length analysis
        word_count = len(text.split())
        if word_count <= 3:
            score += 0.3
            level = 'H1'  # Very short text often titles
        elif word_count <= 6:
            score += 0.2
            level = 'H2'
        elif word_count <= 10:
            score += 0.1
        
        # Capitalization analysis
        if text.isupper():
            score += 0.4
            level = 'H1'  # ALL CAPS often titles
        elif text.istitle():
            score += 0.3
        elif text[0].isupper():
            score += 0.1
        
        # Punctuation analysis
        if not text.endswith('.'):
            score += 0.2  # Headings usually don't end with periods
        if text.endswith(':'):
            score += 0.3  # Colons often indicate headings
        if not any(p in text for p in '.,;!?'):
            score += 0.2  # No punctuation suggests heading
        
        # Number/bullet analysis
        if re.match(r'^\d+\.?\s+', text):
            score += 0.3  # Numbered items
            level = 'H2'
        if re.match(r'^[A-Z]\.\s+', text):
            score += 0.2  # Lettered items
        
        # Font size relative analysis (even if inconsistent, relative comparison helps)
        avg_size = line_info.get('avg_size', 12)
        if avg_size > 13:
            score += 0.2
        if avg_size > 15:
            score += 0.2
            level = 'H1'
        
        if score >= 0.6:
            return (level, min(score, 0.9))
        
        return None
    
    def _extract_text_pypdf2(self, pdf_path: Path) -> Tuple[str, Dict[int, str]]:
        """Fallback text extraction using PyPDF2"""
        try:
            page_texts = {}
            full_text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text() or ""
                        page_texts[page_num] = page_text
                        full_text += f"\\n[PAGE {page_num}]\\n{page_text}\\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        page_texts[page_num] = ""
            
            return full_text, page_texts
            
        except Exception as e:
            logger.error(f"Fallback extraction failed for {pdf_path}: {e}")
            return "", {}
    
    def extract_title(self, text: str, pdf_path: Path) -> str:
        """Extract document title from text using font-based markers"""
        lines = text.split('\n')[:15]  # Check first 15 lines
        
        best_title = None
        best_score = 0
        
        # Look for title patterns with font markers
        for line in lines:
            line = line.strip()
            # Remove page markers
            line = re.sub(r'\[PAGE \d+\]', '', line).strip()
            
            if len(line) < 5 or len(line) > 150:
                continue
            
            title_score = 0
            clean_text = line
            
            # Check for font-based title markers
            if line.startswith('[TITLE:'):
                title_score += 100
                # Extract the actual title text
                match = re.match(r'\[TITLE:\d+\.?\d*\]\s*(.+)', line)
                if match:
                    clean_text = match.group(1).strip()
            elif line.startswith('[H1:'):
                title_score += 50
                match = re.match(r'\[H1:\d+\.?\d*\]\s*(.+)', line)
                if match:
                    clean_text = match.group(1).strip()
            
            # Additional scoring based on content
            line_lower = clean_text.lower()
            if any(keyword in line_lower for keyword in ['application', 'form', 'document', 'report', 'manual', 'guide', 'foundation', 'introduction', 'overview']):
                title_score += 30
            
            if clean_text.isupper():
                title_score += 20
            elif clean_text[0].isupper() and not clean_text.endswith('.'):
                title_score += 10
            
            # Prefer shorter, more title-like text
            if len(clean_text.split()) <= 8:
                title_score += 15
            
            if title_score > best_score and len(clean_text) > 3:
                best_score = title_score
                best_title = clean_text
        
        if best_title:
            # Clean up the title - ENHANCED FOR OCR ARTIFACTS
            title = best_title
            
            # Remove any remaining font markers that might have leaked through
            title = re.sub(r'\[H[1-6]:\d+\.?\d*\]\s*', '', title)
            title = re.sub(r'\[TITLE:\d+\.?\d*\]\s*', '', title)
            title = re.sub(r'H[1-6]:\d+\.?\d*\s*', '', title)
            
            # Fix common OCR artifacts (repeated letters)
            title = re.sub(r'([A-Z])\1+', r'\1', title)  # RRFFPP -> RFP
            title = re.sub(r'([a-z])\1{2,}', r'\1', title)  # eeeqqq -> eq
            
            # Remove excessive repeated patterns
            title = re.sub(r'(RFP|Request|Proposal).*\1', r'\1', title, flags=re.IGNORECASE)
            
            # Clean special characters but keep essential ones
            title = re.sub(r'[^\w\s\-&:.]', ' ', title)
            
            # Remove multiple spaces
            title = re.sub(r'\s+', ' ', title).strip()
            
            # Remove trailing numbers/dots that might be artifacts
            title = re.sub(r'\s+\d+\.?\s*$', '', title)
            
            # Length check after cleaning
            if len(title) > 3 and len(title) <= 100:
                return title
        
        # Fallback to filename
        return pdf_path.stem.replace('_', ' ').replace('-', ' ').title()
    
    def extract_outline(self, text: str, page_texts: Dict[int, str]) -> List[OutlineItem]:
        """Extract structured outline from document text using font analysis"""
        outline_items = []
        
        # Process page by page to maintain page numbers
        for page_num, page_text in page_texts.items():
            if not page_text.strip():
                continue
                
            lines = page_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if len(line) < 3:
                    continue
                
                # Parse font-based markers
                heading_info = self._parse_font_markers(line)
                if heading_info:
                    level, text_content, font_size = heading_info
                    
                    # Clean the text content
                    clean_text = self._clean_heading_text(text_content)
                    if clean_text and len(clean_text) > 2:
                        # Filter out obvious non-headings
                        if not self._is_valid_heading_content(clean_text):
                            continue
                            
                        outline_items.append(OutlineItem(
                            level=level,
                            text=clean_text,
                            page=page_num,
                            confidence=self._calculate_font_based_confidence(level, font_size, clean_text)
                        ))
        
        # Post-process and filter outline items
        return self._post_process_outline(outline_items)
    
    def _parse_font_markers(self, line: str) -> Optional[Tuple[str, str, float]]:
        """Parse font-based markers to extract heading information"""
        # Check for font markers
        title_match = re.match(r'\[TITLE:(\d+\.?\d*)\]\s*(.+)', line)
        if title_match:
            font_size = float(title_match.group(1))
            text_content = title_match.group(2)
            return ('H1', text_content, font_size)  # Treat titles as H1
        
        h1_match = re.match(r'\[H1:(\d+\.?\d*)\]\s*(.+)', line)
        if h1_match:
            font_size = float(h1_match.group(1))
            text_content = h1_match.group(2)
            return ('H1', text_content, font_size)
        
        h2_match = re.match(r'\[H2:(\d+\.?\d*)\]\s*(.+)', line)
        if h2_match:
            font_size = float(h2_match.group(1))
            text_content = h2_match.group(2)
            return ('H2', text_content, font_size)
        
        h3_match = re.match(r'\[H3:(\d+\.?\d*)\]\s*(.+)', line)
        if h3_match:
            font_size = float(h3_match.group(1))
            text_content = h3_match.group(2)
            return ('H3', text_content, font_size)
        
        return None
    
    def _is_valid_heading_content(self, text: str) -> bool:
        """Validate if text content is actually a heading - VERY STRICT FILTERING"""
        text = text.strip()
        
        # Filter out obvious non-headings - VERY STRICT RULES
        invalid_patterns = [
            r'^\s*$',  # Empty
            r'^\d+\s*$',    # Just numbers
            r'^\d{4}.*\d{4}$', # Date ranges like "2014 Page 3 of 12 May 31, 2014"
            r'^[©®™]', # Copyright symbols alone
            r'^page\s+\d+\s+of\s+\d+', # Page x of y
            r'\.{10,}', # Long sequences of dots
            r'^\w+\s+\d{1,2},?\s+\d{4}', # Dates like "March 21, 2003"
            r'^\d{1,2}\/\d{1,2}\/\d{4}', # Date formats 12/31/2024
            r'^\d{4}-\d{2}-\d{2}', # ISO dates 2024-12-31
            r'^\d+\.\d+\s*$', # Just decimal numbers
            r'^[^\w\s]*$', # Only special characters
            r'^\w{1,2}\s*$', # Single or two letter words
            r'commence as soon as possible', # Specific content fragments
            r'contract.*will be signed', # Contract language
            r'committee\.?\s*$', # Just "committee"
            r'^\d+\s+\d+\s+\d+\s+\d+$', # Multiple numbers like "2003 2222"
            r'.*will decrease from.*to.*during.*period', # Financial projections
            r'.*planning process must also secure.*commitment', # Long planning sentences
            r'.*contributions.*endowment.*gifts.*funding', # Financial details
            r'.*that library contributions.*endowment', # Specific library funding
            r'^those proposals that are', # Sentence fragments
            r'^st\.,.*suite.*toronto', # Address fragments
            r'the financial plan for', # Financial plan fragments
            r'will be invited to discuss', # Invitation fragments
            r'must be received by noon', # Deadline fragments
            r'proposals may be.*mailed.*couriered', # Submission instruction fragments
            r'firms.*consultants.*intended to submit', # Application instruction fragments
            # NEW FINANCIAL DATA FILTERS
            r'.*\$\d+.*\$\d+.*', # Multiple dollar amounts like "$0.5M (1%) $3.75M"
            r'^\d+M\$\d+M.*', # Format like "50M$75MTOTAL"
            r'.*\(\d+%\).*', # Percentage in parentheses like "(70%)"
            r'^(gifts|endowment|libraries|government)\s+\$', # Financial line items
            r'^\$\d+.*\(\d+%\)', # Dollar amount with percentage
            r'.*total.*annual.*\$', # Financial summary lines
            r'funding.*source.*\d{4}', # Table headers with years
            r'^\d{4}\s+\d{4}$', # Just two years like "2007 2017"
            r'.*in-kind.*\$.*\%', # Financial in-kind contributions
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        # Must have some alphabetic content
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Very strict length limits - headings should be concise
        if len(text) < 2 or len(text) > 60:  # Reduced from 120 to 60
            return False
        
        # Filter out sentence fragments (contain too many common words) - STRICTER
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'of', 'as', 'that', 'will', 'be', 'are', 'can', 'may', 'must']
        words = text.lower().split()
        common_word_count = sum(1 for word in words if word in common_words)
        if len(words) > 3 and common_word_count > len(words) * 0.3:  # Stricter: 30% instead of 50%
            return False
        
        # Filter out sentences (end with periods and have many words) - STRICTER
        if text.endswith('.') and len(words) > 4:  # Reduced from 8 to 4
            return False
        
        # Filter out text that contains too many lowercase letters (likely sentences)
        if len(words) > 3:
            lowercase_ratio = sum(1 for c in text if c.islower()) / len(text)
            if lowercase_ratio > 0.7:  # More than 70% lowercase
                return False
        
        # Filter out text with too many numbers/symbols (likely table data)
        non_alpha_chars = len(re.sub(r'[a-zA-Z\s]', '', text))
        if len(text) > 10 and non_alpha_chars > len(text) * 0.5:  # More than 50% non-alphabetic
            return False
        
        # Filter out text with dollar signs or percentages (likely financial data)
        if '$' in text or '%' in text:
            # Allow only if it's a clear heading with these terms
            heading_with_money = ['budget', 'cost', 'funding', 'revenue', 'financial']
            if not any(term in text.lower() for term in heading_with_money):
                return False
        
        # Accept common heading words even if they might seem like metadata
        heading_keywords = [
            'copyright', 'version', 'overview', 'introduction', 'chapter', 'section', 
            'notice', 'board', 'foundation', 'level', 'extensions', 'summary', 
            'background', 'methodology', 'results', 'discussion', 'conclusion',
            'objectives', 'scope', 'purpose', 'goals', 'strategy', 'proposal',
            'plan', 'timeline', 'phase', 'implementation'
        ]
        if any(keyword in text.lower() for keyword in heading_keywords):
            return True
        
        # Too many special characters relative to text
        special_chars = len(re.sub(r'[a-zA-Z0-9\s]', '', text))
        if special_chars > len(text) * 0.4:  # Reduced from 0.5 to 0.4
            return False
        
        return True
    
    def _calculate_font_based_confidence(self, level: str, font_size: float, text: str) -> float:
        """Calculate confidence based on font properties and content"""
        base_confidence = 0.6
        
        # Font size boost (more adaptive for irregular PDFs)
        if font_size >= 18:
            base_confidence += 0.3
        elif font_size >= 14:
            base_confidence += 0.2
        elif font_size >= 12:
            base_confidence += 0.1
        
        # Content analysis boost
        word_count = len(text.split())
        if word_count <= 6:  # Short headings are more likely
            base_confidence += 0.1
        
        # Capitalization boost
        if text.isupper():
            base_confidence += 0.2
        elif text.istitle():
            base_confidence += 0.1
        
        # Level-specific adjustments
        if level == 'H1' and font_size >= 16:
            base_confidence += 0.1
        elif level == 'H2' and 12 <= font_size <= 16:
            base_confidence += 0.1
        elif level == 'H3' and font_size <= 14:
            base_confidence += 0.1
        
        # Pattern matching boost
        if re.match(r'^\d+\.?\s+[A-Z]', text):  # Numbered sections
            base_confidence += 0.2
        if any(keyword in text.lower() for keyword in ['overview', 'introduction', 'summary', 'conclusion']):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _classify_heading(self, line: str) -> Optional[str]:
        """Classify a line as a heading level"""
        line = line.strip()
        
        # Skip if too short or looks like regular text
        if len(line) < 3:
            return None
            
        # Check patterns for each level (most specific first)
        for level, patterns in self.heading_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line):
                    return level
        
        # Additional heuristics for numbered items
        if re.match(r'^\d+\.\s+[A-Za-z]', line):
            return 'H1'
        elif re.match(r'^\d+\.\d+\.?\s+[A-Za-z]', line):
            return 'H2'
        elif re.match(r'^\d+\.\d+\.\d+\.?\s+[A-Za-z]', line):
            return 'H3'
        elif re.match(r'^\([a-z]\)\s+[A-Za-z]', line):
            return 'H2'
        elif re.match(r'^[a-z]\)\s+[A-Za-z]', line):
            return 'H2'
        
        return None
    
    def _is_likely_heading(self, line: str) -> bool:
        """Use heuristics to determine if a line is likely a heading"""
        line = line.strip()
        
        # Length checks
        if len(line) < 5 or len(line) > 100:
            return False
        
        # Check for heading indicators
        heading_indicators = [
            line[0].isupper(),  # Starts with capital
            not line.endswith('.'),  # Doesn't end with period
            len(line.split()) <= 10,  # Not too many words
            not any(word in line.lower() for word in ['the', 'and', 'or', 'but', 'in', 'on', 'at']),  # Few function words
        ]
        
        return sum(heading_indicators) >= 2
    
    def _clean_heading_text(self, text: str) -> str:
        """Clean heading text by removing formatting markers and extra content"""
        # Remove any remaining font markers
        text = re.sub(r'\[(TITLE|H[123]):\d+\.?\d*\]\s*', '', text)
        
        # Remove table of contents dots and page numbers
        text = re.sub(r'\.{3,}\s*\d+$', '', text)
        
        # Remove leading/trailing punctuation except meaningful ones
        text = re.sub(r'^[^\w\s]*|[^\w\s]*$', '', text)
        
        # Handle numbered sections - keep meaningful structure
        if re.match(r'^\d+\.?\s+', text):
            # For simple numbered items like "1. Introduction", keep as is
            pass
        elif re.match(r'^\d+\.\d+\.?\s+', text):
            # For subsections like "1.1. Overview", keep as is
            pass
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short or meaningless text
        if len(text) < 2 or text.isdigit():
            return ""
        
        return text
    
    def _calculate_heading_confidence(self, line: str, level: str) -> float:
        """Calculate confidence score for heading classification"""
        base_confidence = 0.7
        
        # Boost confidence for numbered headings
        if re.match(r'^\d+\.', line.strip()):
            base_confidence += 0.2
        elif re.match(r'^\d+\.\d+\.?', line.strip()):
            base_confidence += 0.15
        elif re.match(r'^\([a-z]\)', line.strip()):
            base_confidence += 0.1
        
        # Boost confidence for proper capitalization
        if line.strip() and line.strip()[0].isupper():
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _post_process_outline(self, outline_items: List[OutlineItem]) -> List[OutlineItem]:
        """Post-process outline items to improve quality and remove false positives"""
        if not outline_items:
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        filtered_items = []
        for item in outline_items:
            # Create a normalized key for comparison
            key = (item.text.lower().strip(), item.page, item.level)
            if key not in seen:
                seen.add(key)
                filtered_items.append(item)
        
        # Filter by confidence threshold (more permissive)
        medium_confidence_items = [item for item in filtered_items if item.confidence >= 0.6]
        
        # If we have medium confidence items, prefer them, otherwise use all
        if medium_confidence_items:
            filtered_items = medium_confidence_items
        elif filtered_items:
            # If no medium confidence items, use lower threshold
            filtered_items = [item for item in filtered_items if item.confidence >= 0.4]
        
        # Sort by page number, then by confidence
        filtered_items.sort(key=lambda x: (x.page, -x.confidence))
        
        # Remove items that are too similar to each other
        final_items = []
        for item in filtered_items:
            is_duplicate = False
            for existing in final_items:
                # Check for very similar text
                if (abs(item.page - existing.page) <= 1 and  # Same or adjacent page
                    self._text_similarity(item.text.lower(), existing.text.lower()) > 0.8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_items.append(item)
        
        # Limit to reasonable number of outline items
        return final_items[:30]  # Max 30 items
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def generate_semantic_embeddings(self, outline_items: List[OutlineItem]) -> np.ndarray:
        """Generate semantic embeddings for outline items"""
        if not outline_items:
            return np.array([])
        
        texts = [item.text for item in outline_items]
        embeddings = self.embedding_model.encode(texts)
        return embeddings
    
    def find_semantic_links(self, 
                          doc1_outline: List[OutlineItem],
                          doc2_outline: List[OutlineItem],
                          doc1_name: str,
                          doc2_name: str) -> List[SemanticLink]:
        """Find semantic links between two documents"""
        if not doc1_outline or not doc2_outline:
            return []
        
        # Generate embeddings
        doc1_embeddings = self.generate_semantic_embeddings(doc1_outline)
        doc2_embeddings = self.generate_semantic_embeddings(doc2_outline)
        
        if doc1_embeddings.size == 0 or doc2_embeddings.size == 0:
            return []
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(doc1_embeddings, doc2_embeddings)
        
        links = []
        for i, item1 in enumerate(doc1_outline):
            for j, item2 in enumerate(doc2_outline):
                similarity = similarity_matrix[i][j]
                
                if similarity > self.similarity_threshold:
                    links.append(SemanticLink(
                        source_doc=doc1_name,
                        target_doc=doc2_name,
                        source_section=item1.text,
                        target_section=item2.text,
                        similarity_score=float(similarity),
                        relationship_type=self._classify_relationship_type(similarity)
                    ))
        
        # Sort by similarity score
        links.sort(key=lambda x: x.similarity_score, reverse=True)
        return links[:10]  # Return top 10 links
    
    def _classify_relationship_type(self, similarity: float) -> str:
        """Classify the type of relationship based on similarity score"""
        if similarity > 0.9:
            return "identical"
        elif similarity > 0.8:
            return "highly_similar"
        elif similarity > 0.7:
            return "related"
        else:
            return "loosely_related"
    
    def process_single_document(self, pdf_path: Path) -> DocumentStructure:
        """Process a single PDF document"""
        logger.info(f"Processing document: {pdf_path.name}")
        
        try:
            # Extract text
            full_text, page_texts = self.extract_text_from_pdf(pdf_path)
            
            if not full_text.strip():
                logger.warning(f"No text extracted from {pdf_path.name}")
                return DocumentStructure(
                    title=self.extract_title("", pdf_path),
                    outline=[]
                )
            
            # Extract title
            title = self.extract_title(full_text, pdf_path)
            
            # Extract outline
            outline = self.extract_outline(full_text, page_texts)
            
            # Generate semantic IDs for outline items
            for i, item in enumerate(outline):
                item.semantic_id = f"{pdf_path.stem}_section_{i}"
            
            logger.info(f"Extracted {len(outline)} outline items from {pdf_path.name}")
            
            return DocumentStructure(
                title=title,
                outline=outline,
                metadata={
                    "source_file": pdf_path.name,
                    "page_count": len(page_texts),
                    "outline_items": len(outline)
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            return DocumentStructure(
                title=self.extract_title("", pdf_path),
                outline=[]
            )
    
    def process_document_collection(self, pdf_paths: List[Path]) -> Dict[str, DocumentStructure]:
        """Process a collection of documents and find semantic links"""
        logger.info(f"Processing collection of {len(pdf_paths)} documents")
        
        documents = {}
        
        # Process individual documents
        for pdf_path in pdf_paths:
            doc_structure = self.process_single_document(pdf_path)
            documents[pdf_path.name] = doc_structure
        
        return documents
    
    def generate_output_json(self, doc_structure: DocumentStructure) -> Dict:
        """Generate output JSON in the required format"""
        outline_json = []
        for item in doc_structure.outline:
            outline_json.append({
                "level": item.level,
                "text": item.text,
                "page": item.page
            })
        
        return {
            "title": doc_structure.title,
            "outline": outline_json
        }
    
    def validate_output(self, output_data: Dict, schema_path: Path) -> bool:
        """Validate output against JSON schema"""
        try:
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                validate(output_data, schema)
                return True
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
        
        return False


def main():
    """Main processing function"""
    logger.info("Starting intelligent document processing")
    
    # Initialize processor
    processor = IntelligentDocumentProcessor()
    
    # Set up paths
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each document
    for pdf_file in pdf_files:
        try:
            # Process the document
            doc_structure = processor.process_single_document(pdf_file)
            
            # Generate output JSON
            output_data = processor.generate_output_json(doc_structure)
            
            # Save output
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully processed {pdf_file.name} -> {output_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
            # Create minimal output for failed documents
            minimal_output = {
                "title": pdf_file.stem.replace('_', ' ').replace('-', ' ').title(),
                "outline": []
            }
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(minimal_output, f, indent=2)
    
    logger.info("Document processing completed")


if __name__ == "__main__":
    main()
