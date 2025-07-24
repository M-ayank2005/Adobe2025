import os
import json
import sys
from pathlib import Path

# Add core_system to path for imports
sys.path.append(str(Path(__file__).parent.parent / "core_system"))

from intelligent_document_processor import IntelligentDocumentProcessor

def process_pdfs():
    # Get input and output directories
    current_dir = Path(__file__).parent
    input_dir = current_dir / "sample_dataset" / "pdfs"
    output_dir = current_dir / "sample_dataset" / "outputs"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the intelligent processor
    processor = IntelligentDocumentProcessor()
    
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