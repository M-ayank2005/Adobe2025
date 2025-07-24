#!/usr/bin/env python3
"""
Main processor for Adobe2025 PDF Processing Solution
Processes all PDFs from /app/input directory and generates JSON outputs in /app/output
"""

import os
import json
import sys
from pathlib import Path
import logging

# Add current directory to Python path
sys.path.append('/app')

from intelligent_document_processor import IntelligentDocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main processing function for Docker container"""
    
    # Define input and output directories as per Docker requirements
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure directories exist
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files from input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Initialize the intelligent document processor
    try:
        processor = IntelligentDocumentProcessor()
        logger.info("Document processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing {pdf_file.name}...")
            
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
            
            # Create output JSON file with same name as PDF but .json extension
            output_file = output_dir / f"{pdf_file.stem}.json"
            
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Successfully processed {pdf_file.name} -> {output_file.name} ({len(json_data['outline'])} headings)")
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")
            # Continue processing other files even if one fails
            continue
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
