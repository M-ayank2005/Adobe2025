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
import spacy

# Utilities
from pydantic import BaseModel
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
                r'^(?:\d+\.?\s+)?[A-Z][^.!?]*$',  # Numbered sections
                r'^Chapter\s+\d+',  # Chapter headings
                r'^CHAPTER\s+[IVX]+',  # Roman numeral chapters
                r'^[A-Z\s]{3,}$',  # All caps headings
                r'^\s*(?:\d+\.?\s+)?[A-Z][A-Za-z\s,]{10,50}$'  # Title case headings
            ],
            'H2': [
                r'^\s*(?:\d+\.\d+\.?\s+)?[A-Z][A-Za-z\s,]{5,40}$',  # Subsections
                r'^\s*(?:[a-z]\)|\([a-z]\))\s+[A-Z]',  # Lettered subsections
            ],
            'H3': [
                r'^\s*(?:\d+\.\d+\.\d+\.?\s+)?[A-Z][A-Za-z\s,]{3,30}$',  # Sub-subsections
                r'^\s*(?:[ivx]+\.|[IVX]+\.)\s+[A-Z]',  # Roman numeral subsections
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
            
            # Load spaCy model for text processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, downloading...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize NLTK components
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[int, str]]:
        """Extract text from PDF with page-wise mapping"""
        try:
            page_texts = {}
            full_text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text() or ""
                        page_texts[page_num] = page_text
                        full_text += f"\\n[PAGE {page_num}]\\n{page_text}\\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        page_texts[page_num] = ""
            
            return full_text, page_texts
            
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            # Fallback to PyPDF2
            return self._extract_text_pypdf2(pdf_path)
    
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
        """Extract document title from text or filename"""
        lines = text.split('\\n')[:20]  # Check first 20 lines
        
        # Look for title patterns
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 100:
                # Check if line looks like a title
                if (line.isupper() or 
                    (line[0].isupper() and not line.endswith('.')) or
                    'title' in line.lower()):
                    # Clean up the title
                    title = re.sub(r'[^a-zA-Z0-9\\s\\-_]', '', line).strip()
                    if len(title) > 3:
                        return title
        
        # Fallback to filename
        return pdf_path.stem.replace('_', ' ').replace('-', ' ').title()
    
    def extract_outline(self, text: str, page_texts: Dict[int, str]) -> List[OutlineItem]:
        """Extract structured outline from document text"""
        outline_items = []
        
        # Process page by page to maintain page numbers
        for page_num, page_text in page_texts.items():
            if not page_text.strip():
                continue
                
            lines = page_text.split('\\n')
            
            for line in lines:
                line = line.strip()
                if len(line) < 3 or len(line) > 200:
                    continue
                
                # Check against heading patterns
                heading_level = self._classify_heading(line)
                if heading_level:
                    # Clean up the heading text
                    clean_text = self._clean_heading_text(line)
                    if clean_text:
                        outline_items.append(OutlineItem(
                            level=heading_level,
                            text=clean_text,
                            page=page_num,
                            confidence=self._calculate_heading_confidence(line, heading_level)
                        ))
        
        # Post-process and filter outline items
        return self._post_process_outline(outline_items)
    
    def _classify_heading(self, line: str) -> Optional[str]:
        """Classify a line as a heading level"""
        line = line.strip()
        
        # Skip if too short or contains too many special characters
        if len(line) < 3 or len(re.sub(r'[a-zA-Z0-9\\s]', '', line)) > len(line) * 0.3:
            return None
            
        # Check patterns for each level
        for level, patterns in self.heading_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return level
        
        # Additional heuristics
        if self._is_likely_heading(line):
            if re.match(r'^\\d+\\.', line):
                return 'H1'
            elif re.match(r'^\\d+\\.\\d+', line):
                return 'H2'
            elif re.match(r'^\\d+\\.\\d+\\.\\d+', line):
                return 'H3'
            else:
                return 'H2'  # Default for unclassified headings
        
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
        """Clean heading text by removing numbers and extra whitespace"""
        # Remove leading numbers and bullets
        text = re.sub(r'^[\\d\\.\\)\\-\\*]+\\s*', '', text)
        # Remove trailing dots and extra whitespace
        text = re.sub(r'\\.*$', '', text).strip()
        # Normalize whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        return text if len(text) > 2 else ""
    
    def _calculate_heading_confidence(self, line: str, level: str) -> float:
        """Calculate confidence score for heading classification"""
        base_confidence = 0.8
        
        # Boost confidence for numbered headings
        if re.match(r'^\\d+', line.strip()):
            base_confidence += 0.1
        
        # Boost confidence for capitalized headings
        if line.strip()[0].isupper():
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _post_process_outline(self, outline_items: List[OutlineItem]) -> List[OutlineItem]:
        """Post-process outline items to improve quality"""
        if not outline_items:
            return []
        
        # Remove duplicates
        seen = set()
        filtered_items = []
        for item in outline_items:
            key = (item.text.lower(), item.page)
            if key not in seen:
                seen.add(key)
                filtered_items.append(item)
        
        # Sort by page number and confidence
        filtered_items.sort(key=lambda x: (x.page, -x.confidence))
        
        # Limit to reasonable number of outline items
        return filtered_items[:50]
    
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
