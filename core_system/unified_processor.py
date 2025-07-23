"""
Unified Document Processing System
Handles both Challenge 1a (basic outline extraction) and Challenge 1b (semantic analysis)
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Import our core modules
from intelligent_document_processor import IntelligentDocumentProcessor
from semantic_intelligence import PersonaIntelligenceEngine, process_challenge_1b_collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedDocumentSystem:
    """
    Unified system that can handle both types of challenges
    """
    
    def __init__(self):
        """Initialize the unified system"""
        self.base_processor = IntelligentDocumentProcessor()
        self.persona_engine = PersonaIntelligenceEngine(self.base_processor)
    
    def process_challenge_1a(self, input_dir: Path, output_dir: Path):
        """Process Challenge 1a: Basic outline extraction"""
        logger.info("Processing Challenge 1a: Basic outline extraction")
        
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
                doc_structure = self.base_processor.process_single_document(pdf_file)
                
                # Generate output JSON
                output_data = self.base_processor.generate_output_json(doc_structure)
                
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
    
    def process_challenge_1b(self, base_dir: Path):
        """Process Challenge 1b: Semantic analysis"""
        logger.info("Processing Challenge 1b: Semantic analysis")
        
        # Process all collections
        collections = ["Collection 1", "Collection 2", "Collection 3"]
        
        for collection_name in collections:
            collection_path = base_dir / collection_name
            if collection_path.exists():
                logger.info(f"Processing {collection_name}")
                
                # Find input file
                input_file = collection_path / "challenge1b_input.json"
                pdf_directory = collection_path / "PDFs"
                
                if input_file.exists() and pdf_directory.exists():
                    try:
                        # Process the collection
                        analysis = self.persona_engine.process_challenge_1b(input_file, pdf_directory)
                        
                        # Generate output
                        output_data = self.persona_engine.generate_challenge_1b_output(analysis)
                        
                        # Save output
                        output_file = collection_path / "challenge1b_output_generated.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, indent=4, ensure_ascii=False)
                        
                        logger.info(f"Generated output for {collection_name}: {output_file}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {collection_name}: {e}")
                else:
                    logger.warning(f"Missing files for {collection_name}")
            else:
                logger.warning(f"Collection not found: {collection_name}")
    
    def auto_detect_and_process(self, work_dir: Path):
        """Auto-detect challenge type and process accordingly"""
        logger.info(f"Auto-detecting challenge type in: {work_dir}")
        
        # Check for Challenge 1a structure
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        
        if input_dir.exists():
            logger.info("Detected Challenge 1a structure")
            self.process_challenge_1a(input_dir, output_dir)
            return
        
        # Check for Challenge 1b structure
        collection_dirs = [d for d in work_dir.iterdir() if d.is_dir() and "Collection" in d.name]
        
        if collection_dirs:
            logger.info("Detected Challenge 1b structure")
            self.process_challenge_1b(work_dir)
            return
        
        # Check for individual PDF files (Challenge 1a alternative)
        pdf_files = list(work_dir.glob("*.pdf"))
        if pdf_files:
            logger.info("Detected individual PDF files, processing as Challenge 1a")
            output_dir = work_dir / "output"
            self.process_challenge_1a(work_dir, output_dir)
            return
        
        logger.warning("Could not detect challenge type. No processing performed.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unified Document Processing System")
    parser.add_argument("--challenge", choices=["1a", "1b", "auto"], default="auto",
                       help="Challenge type to process")
    parser.add_argument("--input-dir", type=Path, help="Input directory for Challenge 1a")
    parser.add_argument("--output-dir", type=Path, help="Output directory for Challenge 1a")
    parser.add_argument("--work-dir", type=Path, default=Path("."),
                       help="Working directory (for auto-detection)")
    
    args = parser.parse_args()
    
    # Initialize system
    system = UnifiedDocumentSystem()
    
    if args.challenge == "1a":
        if not args.input_dir or not args.output_dir:
            logger.error("For Challenge 1a, both --input-dir and --output-dir are required")
            sys.exit(1)
        system.process_challenge_1a(args.input_dir, args.output_dir)
    
    elif args.challenge == "1b":
        system.process_challenge_1b(args.work_dir)
    
    else:  # auto
        system.auto_detect_and_process(args.work_dir)


# Docker-compatible entry point
def docker_main():
    """Entry point for Docker containers (Challenge 1a)"""
    logger.info("Starting Docker-compatible processing")
    
    # Use standard Docker paths
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Initialize and run
    system = UnifiedDocumentSystem()
    system.process_challenge_1a(input_dir, output_dir)
    
    logger.info("Docker processing completed")


if __name__ == "__main__":
    # Check if running in Docker environment
    if os.path.exists("/app/input"):
        docker_main()
    else:
        main()
