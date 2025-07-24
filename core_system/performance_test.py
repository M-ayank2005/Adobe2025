#!/usr/bin/env python3
"""
Performance test for PDF processing
"""

import time
import sys
from pathlib import Path
sys.path.append('.')

from intelligent_document_processor import IntelligentDocumentProcessor

def test_performance():
    """Test processing performance"""
    print("🚀 Testing PDF Processing Performance...\n")
    
    processor = IntelligentDocumentProcessor()
    pdf_dir = Path("../Challenge_1a/sample_dataset/pdfs")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    total_start_time = time.time()
    
    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}")
        
        start_time = time.time()
        
        try:
            # Process the document
            doc_structure = processor.process_single_document(pdf_file)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"   ⏱️  Time: {processing_time:.3f}s")
            print(f"   📋 Title: {doc_structure.title[:50]}...")
            print(f"   📖 Outline items: {len(doc_structure.outline)}")
            
            # Check if within performance requirements
            if processing_time > 10.0:  # Should process in under 10 seconds
                print(f"   ⚠️  Warning: Processing took {processing_time:.3f}s (>10s limit)")
            else:
                print(f"   ✅ Performance: OK")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"📊 Summary:")
    print(f"   📁 Files processed: {len(pdf_files)}")
    print(f"   ⏱️  Total time: {total_time:.3f}s")
    print(f"   📈 Average time per file: {total_time/len(pdf_files):.3f}s")
    
    if total_time/len(pdf_files) < 10.0:
        print(f"   ✅ Performance requirement met!")
    else:
        print(f"   ❌ Performance requirement not met")

if __name__ == "__main__":
    test_performance()
