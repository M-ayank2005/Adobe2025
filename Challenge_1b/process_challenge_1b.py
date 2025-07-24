#!/usr/bin/env python3
"""
Challenge 1b: Multi-Collection PDF Analysis Processor
Processes multiple document collections and extracts relevant content based on personas and use cases.
"""

import os
import json
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

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
    full_text: str = ""


class SimplePDFProcessor:
    """Simplified PDF processor for Challenge 1b"""
    
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
        full_text = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(page_text)
                    
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
        
        return text_blocks, '\n'.join(full_text)
    
    def extract_text_fallback(self, pdf_path: Path):
        """Fallback extraction using PyPDF2"""
        text_blocks = []
        full_text = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
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
        
        return text_blocks, '\n'.join(full_text)
    
    def is_likely_heading(self, text: str, font_size: float, is_bold: bool, avg_font_size: float) -> bool:
        """Determine if text is likely a heading"""
        text = text.strip()
        
        # Skip if too short or too long
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Skip financial/numeric data
        if re.search(r'^\$[\d,]+|^\d+,\d+|^\d+\.\d{2}$', text):
            return False
        
        # Skip if mostly punctuation
        if len(re.sub(r'[^\w\s]', '', text)) < len(text) * 0.5:
            return False
        
        confidence = 0.0
        
        # Font size analysis
        if font_size > avg_font_size * 1.2:
            confidence += 0.4
        
        # Bold text
        if is_bold:
            confidence += 0.3
        
        # Pattern matching
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                confidence += 0.3
                break
        
        # Position and formatting clues
        if text.isupper() and len(text.split()) <= 6:
            confidence += 0.2
        
        if text.endswith(':') and not text.count(':') > 1:
            confidence += 0.1
        
        if re.match(r'^\d+\.?\s+[A-Z]', text):
            confidence += 0.2
        
        return confidence > 0.4
    
    def process_single_document(self, pdf_path: Path) -> DocumentStructure:
        """Process a single PDF document"""
        print(f"  Processing {pdf_path.name}...")
        
        # Extract text blocks
        text_blocks, full_text = self.extract_text_with_fonts(pdf_path)
        
        if not text_blocks:
            return DocumentStructure(title="Document", outline=[], full_text=full_text)
        
        # Calculate average font size
        font_sizes = [block['font_size'] for block in text_blocks if block['font_size'] > 0]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        
        # Extract title
        title = text_blocks[0]['text'] if text_blocks else "Document"
        
        # Find headings
        outline = []
        seen_headings = set()
        
        for block in text_blocks:
            text = block['text'].strip()
            
            # Skip duplicates
            if text.lower() in seen_headings:
                continue
            
            if self.is_likely_heading(text, block['font_size'], block['is_bold'], avg_font_size):
                # Determine heading level
                level = "H1"
                if block['font_size'] > avg_font_size * 1.5:
                    level = "H1"
                elif block['font_size'] > avg_font_size * 1.2:
                    level = "H2"
                else:
                    level = "H3"
                
                outline_item = OutlineItem(
                    level=level,
                    text=text,
                    page=block['page']
                )
                outline.append(outline_item)
                seen_headings.add(text.lower())
        
        return DocumentStructure(title=title, outline=outline, full_text=full_text)


class SimpleSemanticAnalyzer:
    """Enhanced semantic analyzer for persona-based content extraction"""
    
    def __init__(self):
        # Enhanced persona keywords mapping
        self.persona_keywords = {
            "travel planner": {
                "primary": ["destination", "hotel", "restaurant", "attraction", "transport", "city", "guide", "tour", "accommodation", "booking"],
                "secondary": ["location", "visit", "stay", "eat", "travel", "trip", "journey", "sightseeing", "culture", "local"]
            },
            "hr professional": {
                "primary": ["form", "onboarding", "compliance", "employee", "document", "policy", "procedure", "hiring", "recruitment"],
                "secondary": ["staff", "personnel", "training", "benefits", "payroll", "management", "workflow", "process"]
            },
            "food contractor": {
                "primary": ["recipe", "menu", "ingredient", "cooking", "vegetarian", "buffet", "catering", "food", "preparation"],
                "secondary": ["dish", "meal", "cuisine", "nutrition", "dietary", "kitchen", "service", "gluten-free", "vegan"]
            },
            "phd researcher": {
                "primary": ["methodology", "research", "analysis", "study", "experiment", "data", "results", "conclusion"],
                "secondary": ["literature", "review", "hypothesis", "theory", "model", "dataset", "benchmark", "evaluation"]
            },
            "investment analyst": {
                "primary": ["revenue", "financial", "investment", "market", "analysis", "profit", "growth", "performance"],
                "secondary": ["earnings", "portfolio", "risk", "return", "valuation", "trend", "forecast", "strategy"]
            },
            "chemistry student": {
                "primary": ["reaction", "mechanism", "kinetics", "organic", "molecule", "synthesis", "compound"],
                "secondary": ["chemistry", "chemical", "formula", "structure", "bond", "catalyst", "equation"]
            }
        }
        
        # Task-specific keywords
        self.task_keywords = {
            "literature review": ["methodology", "approach", "study", "research", "analysis", "comparison"],
            "exam preparation": ["concept", "principle", "formula", "example", "practice", "key"],
            "trip planning": ["itinerary", "schedule", "recommendation", "guide", "tips"],
            "menu planning": ["recipe", "ingredient", "preparation", "serving", "dietary"],
            "financial analysis": ["revenue", "cost", "profit", "trend", "performance", "growth"]
        }
    
    def analyze_relevance(self, text: str, persona: str, task: str) -> float:
        """Enhanced relevance analysis with weighted scoring"""
        text_lower = text.lower()
        persona_lower = persona.lower()
        task_lower = task.lower()
        
        score = 0.0
        total_weight = 0.0
        
        # Find matching persona keywords
        for persona_key, keywords in self.persona_keywords.items():
            if persona_key in persona_lower:
                # Primary keywords (higher weight)
                for keyword in keywords["primary"]:
                    if keyword in text_lower:
                        score += 3.0
                        total_weight += 3.0
                    else:
                        total_weight += 3.0
                
                # Secondary keywords (lower weight)
                for keyword in keywords["secondary"]:
                    if keyword in text_lower:
                        score += 1.0
                        total_weight += 1.0
                    else:
                        total_weight += 1.0
                break
        
        # Task-specific analysis
        for task_key, keywords in self.task_keywords.items():
            if any(word in task_lower for word in task_key.split()):
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 2.0
                        total_weight += 2.0
                    else:
                        total_weight += 2.0
                break
        
        # Task words direct matching
        task_words = [word for word in task_lower.split() if len(word) > 3]
        for word in task_words:
            if word in text_lower:
                score += 1.5
            total_weight += 1.5
        
        # Normalize score
        return min(score / max(total_weight, 1), 1.0) if total_weight > 0 else 0.0
    
    def extract_relevant_sections(self, documents: Dict[str, DocumentStructure], persona: str, task: str) -> List[dict]:
        """Extract sections relevant to persona and task"""
        relevant_sections = []
        
        for filename, doc_structure in documents.items():
            for i, item in enumerate(doc_structure.outline):
                relevance = self.analyze_relevance(item.text, persona, task)
                
                if relevance > 0.1:  # Threshold for relevance
                    relevant_sections.append({
                        "document": filename,
                        "section_title": item.text,
                        "importance_rank": 0,  # Will be set later
                        "page_number": item.page,
                        "relevance_score": relevance
                    })
        
        # Sort by relevance and assign ranks
        relevant_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        for i, section in enumerate(relevant_sections[:10]):
            section["importance_rank"] = i + 1
        
        return relevant_sections[:10]
    
    def generate_subsection_analysis(self, extracted_sections: List[dict], documents: Dict[str, DocumentStructure]) -> List[dict]:
        """Generate refined subsection analysis"""
        analyses = []
        
        for section in extracted_sections[:5]:  # Top 5 sections
            doc_name = section["document"]
            if doc_name in documents:
                doc = documents[doc_name]
                
                # Find content around this section
                section_text = section["section_title"]
                
                # Simple content extraction - just use the section title and some context
                refined_text = section_text
                
                # Try to get more context from full text
                if doc.full_text and section_text in doc.full_text:
                    start_idx = doc.full_text.find(section_text)
                    if start_idx != -1:
                        # Get some context around the section
                        context_start = max(0, start_idx - 100)
                        context_end = min(len(doc.full_text), start_idx + len(section_text) + 200)
                        refined_text = doc.full_text[context_start:context_end].strip()
                
                analyses.append({
                    "document": doc_name,
                    "refined_text": refined_text,
                    "page_number": section["page_number"]
                })
        
        return analyses

def analyze_collection(collection_path, processor, semantic_analyzer):
    """Analyze a single collection based on its input configuration"""
    
    # Load input configuration
    input_file = collection_path / "challenge1b_input.json"
    if not input_file.exists():
        print(f"Input file not found in {collection_path}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Extract configuration details
    challenge_info = config.get("challenge_info", {})
    documents = config.get("documents", [])
    persona = config.get("persona", {})
    job_to_be_done = config.get("job_to_be_done", {})
    
    print(f"Processing collection: {challenge_info.get('challenge_id', 'Unknown')}")
    print(f"Persona: {persona.get('role', 'Unknown')}")
    print(f"Task: {job_to_be_done.get('task', 'Unknown')}")
    
    # Process each document in the collection
    processed_documents = {}
    pdfs_dir = collection_path / "PDFs"
    
    for doc_info in documents:
        filename = doc_info.get("filename", "")
        title = doc_info.get("title", filename)
        pdf_path = pdfs_dir / filename
        
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            continue
        
        try:
            print(f"  Processing: {filename}")
            
            # Process the PDF document
            result = processor.process_single_document(pdf_path)
            processed_documents[filename] = result
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Use the semantic analyzer to extract relevant sections
    persona_role = persona.get("role", "")
    task_description = job_to_be_done.get("task", "")
    
    try:
        # Extract relevant sections using the semantic analyzer
        extracted_sections = semantic_analyzer.extract_relevant_sections(
            processed_documents, persona_role, task_description
        )
        
        # Generate subsection analysis
        subsection_analysis = semantic_analyzer.generate_subsection_analysis(
            extracted_sections, processed_documents
        )
        
    except Exception as e:
        print(f"Error in semantic analysis: {str(e)}")
        # Fallback to simple extraction
        extracted_sections = []
        subsection_analysis = []
        
        for filename, doc_structure in processed_documents.items():
            for i, item in enumerate(doc_structure.outline[:10]):
                extracted_sections.append({
                    "document": filename,
                    "section_title": item.text,
                    "importance_rank": i + 1,
                    "page_number": item.page
                })
            
            for item in doc_structure.outline[:5]:
                subsection_analysis.append({
                    "document": filename,
                    "refined_text": item.text,
                    "page_number": item.page
                })
    
    # Create output JSON
    output_data = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in documents],
            "persona": persona.get("role", ""),
            "job_to_be_done": job_to_be_done.get("task", "")
        },
        "extracted_sections": [
            {
                "document": section.get("document", ""),
                "section_title": section.get("section_title", ""),
                "importance_rank": section.get("importance_rank", 1),
                "page_number": section.get("page_number", 1)
            }
            for section in extracted_sections
        ],
        "subsection_analysis": [
            {
                "document": analysis.get("document", ""),
                "refined_text": analysis.get("refined_text", ""),
                "page_number": analysis.get("page_number", 1)
            }
            for analysis in subsection_analysis
        ]
    }
    
    # Save output
    output_file = collection_path / "challenge1b_output_generated.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Generated output: {output_file}")

def process_challenge_1b():
    """Main processing function for Challenge 1b"""
    
    # Get the Challenge 1b directory (current working directory in Docker)
    challenge_dir = Path(".")
    
    # Initialize processors
    try:
        processor = SimplePDFProcessor()
        semantic_analyzer = SimpleSemanticAnalyzer()
        print("Processors initialized successfully")
    except Exception as e:
        print(f"Error initializing processors: {str(e)}")
        return
    
    # Find all collection directories
    collections = [d for d in challenge_dir.iterdir() if d.is_dir() and d.name.startswith("Collection")]
    
    if not collections:
        print("No collection directories found")
        return
    
    print(f"Found {len(collections)} collections to process")
    
    # Process each collection
    for collection in sorted(collections):
        print(f"\n--- Processing {collection.name} ---")
        analyze_collection(collection, processor, semantic_analyzer)
    
    print("\nChallenge 1b processing completed!")

if __name__ == "__main__":
    process_challenge_1b()
