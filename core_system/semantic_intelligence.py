"""
Advanced Semantic Linking and Multi-Document Analysis
Handles Challenge 1b requirements for persona-based content analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from collections import defaultdict

from intelligent_document_processor import (
    IntelligentDocumentProcessor, DocumentStructure, OutlineItem, SemanticLink
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSection:
    """Represents an extracted section relevant to the persona/task"""
    document: str
    section_title: str
    importance_rank: int
    page_number: int
    content: str = ""
    relevance_score: float = 0.0


@dataclass
class SubsectionAnalysis:
    """Detailed analysis of a subsection"""
    document: str
    refined_text: str
    page_number: int
    semantic_category: str = ""
    key_concepts: List[str] = None


@dataclass
class PersonaBasedAnalysis:
    """Complete analysis for a specific persona and task"""
    metadata: Dict[str, Any]
    extracted_sections: List[ExtractedSection]
    subsection_analysis: List[SubsectionAnalysis]
    semantic_links: List[SemanticLink] = None
    concept_map: Dict[str, List[str]] = None


class PersonaIntelligenceEngine:
    """
    Advanced engine for persona-based document analysis and content extraction
    """
    
    def __init__(self, base_processor: IntelligentDocumentProcessor):
        self.processor = base_processor
        
        # Persona-specific keywords and concepts
        self.persona_keywords = {
            "Travel Planner": {
                "primary": ["travel", "trip", "vacation", "destination", "itinerary", "hotel", "restaurant", 
                           "attraction", "activity", "tour", "booking", "transportation", "flight"],
                "secondary": ["culture", "history", "food", "entertainment", "nightlife", "shopping",
                             "budget", "cost", "price", "recommendation", "guide", "tips"]
            },
            "HR Professional": {
                "primary": ["form", "document", "employee", "onboarding", "compliance", "fillable",
                           "signature", "workflow", "process", "automation", "template"],
                "secondary": ["digital", "electronic", "pdf", "acrobat", "training", "policy",
                             "procedure", "management", "administration", "legal"]
            },
            "Food Contractor": {
                "primary": ["recipe", "cooking", "food", "meal", "ingredient", "preparation", "kitchen",
                           "catering", "menu", "buffet", "vegetarian", "dietary"],
                "secondary": ["nutrition", "allergy", "serving", "portion", "cost", "planning",
                             "equipment", "safety", "hygiene", "presentation"]
            }
        }
        
        # Task-specific concepts
        self.task_concepts = {
            "planning": ["plan", "schedule", "organize", "arrange", "coordinate", "prepare"],
            "group": ["group", "team", "multiple", "friends", "colleagues", "people"],
            "corporate": ["corporate", "business", "professional", "office", "company"],
            "duration": ["days", "time", "duration", "period", "schedule"],
            "dietary": ["vegetarian", "vegan", "dietary", "restriction", "allergy", "preference"]
        }
        
    def analyze_persona_relevance(self, text: str, persona: str, task: str) -> float:
        """Calculate relevance score based on persona and task"""
        text_lower = text.lower()
        score = 0.0
        
        # Get persona keywords
        persona_kw = self.persona_keywords.get(persona, {"primary": [], "secondary": []})
        
        # Score based on primary keywords (higher weight)
        primary_matches = sum(1 for kw in persona_kw["primary"] if kw in text_lower)
        score += primary_matches * 2.0
        
        # Score based on secondary keywords
        secondary_matches = sum(1 for kw in persona_kw["secondary"] if kw in text_lower)
        score += secondary_matches * 1.0
        
        # Score based on task-related concepts
        task_lower = task.lower()
        for concept, keywords in self.task_concepts.items():
            if any(kw in task_lower for kw in keywords):
                task_matches = sum(1 for kw in keywords if kw in text_lower)
                score += task_matches * 1.5
        
        # Normalize score
        max_possible = len(persona_kw["primary"]) * 2 + len(persona_kw["secondary"]) * 1
        return min(score / max(max_possible, 1), 1.0) if max_possible > 0 else 0.0
    
    def extract_relevant_sections(self, 
                                documents: Dict[str, DocumentStructure],
                                persona: str,
                                task: str,
                                max_sections: int = 10) -> List[ExtractedSection]:
        """Extract sections most relevant to the persona and task"""
        all_sections = []
        
        for doc_name, doc_structure in documents.items():
            for item in doc_structure.outline:
                # Calculate relevance score
                relevance = self.analyze_persona_relevance(item.text, persona, task)
                
                if relevance > 0.1:  # Only include reasonably relevant sections
                    section = ExtractedSection(
                        document=doc_name,
                        section_title=item.text,
                        importance_rank=0,  # Will be set later
                        page_number=item.page,
                        relevance_score=relevance
                    )
                    all_sections.append(section)
        
        # Sort by relevance score
        all_sections.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Assign importance ranks
        for i, section in enumerate(all_sections[:max_sections]):
            section.importance_rank = i + 1
        
        return all_sections[:max_sections]
    
    def generate_subsection_analysis(self,
                                   extracted_sections: List[ExtractedSection],
                                   documents: Dict[str, DocumentStructure],
                                   pdf_texts: Dict[str, Dict[int, str]],
                                   max_analyses: int = 5) -> List[SubsectionAnalysis]:
        """Generate detailed analysis for top sections"""
        analyses = []
        
        # Select top sections for detailed analysis
        top_sections = extracted_sections[:max_analyses]
        
        for section in top_sections:
            try:
                # Get the page text for this section
                doc_name = section.document
                page_num = section.page_number
                
                if doc_name in pdf_texts and page_num in pdf_texts[doc_name]:
                    page_text = pdf_texts[doc_name][page_num]
                    
                    # Extract relevant content around the section
                    refined_text = self._extract_section_content(
                        page_text, section.section_title
                    )
                    
                    if refined_text:
                        analysis = SubsectionAnalysis(
                            document=doc_name,
                            refined_text=refined_text,
                            page_number=page_num,
                            semantic_category=self._categorize_content(refined_text)
                        )
                        analyses.append(analysis)
                        
            except Exception as e:
                logger.warning(f"Error analyzing section {section.section_title}: {e}")
        
        return analyses
    
    def _extract_section_content(self, page_text: str, section_title: str) -> str:
        """Extract content related to a specific section from page text"""
        lines = page_text.split('\\n')
        content_lines = []
        collecting = False
        
        for line in lines:
            line = line.strip()
            
            # Check if this line contains or starts the section
            if section_title.lower() in line.lower() or collecting:
                collecting = True
                content_lines.append(line)
                
                # Stop collecting if we hit another major section
                if (len(content_lines) > 1 and 
                    len(line) > 0 and 
                    (line[0].isupper() or line.startswith('•') or 
                     any(marker in line for marker in ['Chapter', 'Section', '1.', '2.', '3.']))):
                    if not section_title.lower() in line.lower():
                        break
            
            # Limit content length
            if len(content_lines) > 15:
                break
        
        # Join and clean the content
        content = ' '.join(content_lines)
        content = ' '.join(content.split())  # Normalize whitespace
        
        # Ensure reasonable length (100-500 words)
        words = content.split()
        if len(words) > 500:
            content = ' '.join(words[:500])
        elif len(words) < 20:
            # If too short, try to get more context
            content = self._get_extended_context(page_text, section_title)
        
        return content
    
    def _get_extended_context(self, page_text: str, section_title: str) -> str:
        """Get extended context when initial extraction is too short"""
        # Simple approach: take text around the section title
        text_lower = page_text.lower()
        title_lower = section_title.lower()
        
        # Find the position of the section title
        pos = text_lower.find(title_lower)
        if pos != -1:
            # Get surrounding context (±200 characters)
            start = max(0, pos - 200)
            end = min(len(page_text), pos + len(section_title) + 400)
            context = page_text[start:end]
            
            # Clean up
            context = ' '.join(context.split())
            return context
        
        return section_title  # Fallback to just the title
    
    def _categorize_content(self, text: str) -> str:
        """Categorize content into semantic categories"""
        text_lower = text.lower()
        
        # Define category keywords
        categories = {
            "travel_planning": ["destination", "travel", "trip", "vacation", "itinerary"],
            "accommodation": ["hotel", "stay", "accommodation", "booking", "room"],
            "dining": ["restaurant", "food", "cuisine", "dining", "meal"],
            "activities": ["activity", "attraction", "tour", "visit", "entertainment"],
            "transportation": ["transport", "flight", "train", "car", "bus"],
            "cultural": ["culture", "history", "tradition", "museum", "heritage"],
            "practical": ["tips", "guide", "advice", "practical", "useful"],
            "forms_management": ["form", "document", "template", "fillable"],
            "compliance": ["compliance", "legal", "policy", "requirement"],
            "cooking": ["recipe", "cooking", "ingredient", "preparation"],
            "menu_planning": ["menu", "meal", "buffet", "catering", "planning"]
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return the highest scoring category
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def create_concept_map(self, analyses: List[SubsectionAnalysis]) -> Dict[str, List[str]]:
        """Create a concept map linking related ideas across documents"""
        concept_map = defaultdict(list)
        
        # Extract key concepts from each analysis
        for analysis in analyses:
            concepts = self._extract_key_concepts(analysis.refined_text)
            
            for concept in concepts:
                if concept not in concept_map[analysis.semantic_category]:
                    concept_map[analysis.semantic_category].append(concept)
        
        return dict(concept_map)
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using NLP"""
        try:
            # Use spaCy to extract named entities and noun phrases
            doc = self.processor.nlp(text)
            
            concepts = []
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'GPE', 'PERSON', 'PRODUCT', 'EVENT']:
                    concepts.append(ent.text.strip())
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if 2 <= len(chunk.text.split()) <= 4:  # 2-4 word phrases
                    concepts.append(chunk.text.strip())
            
            # Deduplicate and filter
            unique_concepts = list(set(concepts))
            return [c for c in unique_concepts if len(c) > 3 and len(c) < 50][:10]
            
        except Exception as e:
            logger.warning(f"Error extracting concepts: {e}")
            return []
    
    def process_challenge_1b(self, 
                           input_file: Path,
                           pdf_directory: Path) -> PersonaBasedAnalysis:
        """Process Challenge 1b input and generate analysis"""
        
        # Load input configuration
        with open(input_file, 'r') as f:
            config = json.load(f)
        
        # Extract configuration details
        challenge_info = config.get("challenge_info", {})
        documents_config = config.get("documents", [])
        persona = config.get("persona", {}).get("role", "Unknown")
        task = config.get("job_to_be_done", {}).get("task", "")
        
        logger.info(f"Processing Challenge 1b for persona: {persona}")
        logger.info(f"Task: {task}")
        
        # Find PDF files
        pdf_files = []
        pdf_texts = {}
        
        for doc_config in documents_config:
            filename = doc_config.get("filename", "")
            pdf_path = pdf_directory / filename
            
            if pdf_path.exists():
                pdf_files.append(pdf_path)
                # Extract text for later use
                try:
                    _, page_texts = self.processor.extract_text_from_pdf(pdf_path)
                    pdf_texts[filename] = page_texts
                except Exception as e:
                    logger.warning(f"Error extracting text from {filename}: {e}")
                    pdf_texts[filename] = {}
        
        # Process documents
        documents = self.processor.process_document_collection(pdf_files)
        
        # Extract relevant sections
        extracted_sections = self.extract_relevant_sections(documents, persona, task)
        
        # Generate subsection analysis
        subsection_analysis = self.generate_subsection_analysis(
            extracted_sections, documents, pdf_texts
        )
        
        # Create concept map
        concept_map = self.create_concept_map(subsection_analysis)
        
        # Create metadata
        metadata = {
            "input_documents": [doc["filename"] for doc in documents_config],
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return PersonaBasedAnalysis(
            metadata=metadata,
            extracted_sections=extracted_sections,
            subsection_analysis=subsection_analysis,
            concept_map=concept_map
        )
    
    def generate_challenge_1b_output(self, analysis: PersonaBasedAnalysis) -> Dict:
        """Generate output JSON for Challenge 1b"""
        
        # Convert extracted sections
        extracted_sections_json = []
        for section in analysis.extracted_sections:
            extracted_sections_json.append({
                "document": section.document,
                "section_title": section.section_title,
                "importance_rank": section.importance_rank,
                "page_number": section.page_number
            })
        
        # Convert subsection analysis
        subsection_analysis_json = []
        for analysis_item in analysis.subsection_analysis:
            subsection_analysis_json.append({
                "document": analysis_item.document,
                "refined_text": analysis_item.refined_text,
                "page_number": analysis_item.page_number
            })
        
        return {
            "metadata": analysis.metadata,
            "extracted_sections": extracted_sections_json,
            "subsection_analysis": subsection_analysis_json
        }


def process_challenge_1b_collection(collection_path: Path):
    """Process a Challenge 1b collection"""
    logger.info(f"Processing Challenge 1b collection: {collection_path.name}")
    
    # Initialize processors
    base_processor = IntelligentDocumentProcessor()
    persona_engine = PersonaIntelligenceEngine(base_processor)
    
    # Find input file
    input_file = collection_path / "challenge1b_input.json"
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Find PDF directory
    pdf_directory = collection_path / "PDFs"
    if not pdf_directory.exists():
        logger.error(f"PDF directory not found: {pdf_directory}")
        return
    
    try:
        # Process the collection
        analysis = persona_engine.process_challenge_1b(input_file, pdf_directory)
        
        # Generate output
        output_data = persona_engine.generate_challenge_1b_output(analysis)
        
        # Save output
        output_file = collection_path / "challenge1b_output_generated.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Generated output: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing collection {collection_path.name}: {e}")


def main():
    """Main function for Challenge 1b processing"""
    base_path = Path(".")
    
    # Process all collections
    for collection_dir in ["Collection 1", "Collection 2", "Collection 3"]:
        collection_path = base_path / collection_dir
        if collection_path.exists():
            process_challenge_1b_collection(collection_path)


if __name__ == "__main__":
    main()
