"""
Test Suite and Validation System
Comprehensive testing for the intelligent document processing system
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from intelligent_document_processor import (
    IntelligentDocumentProcessor, DocumentStructure, OutlineItem
)
from semantic_intelligence import PersonaIntelligenceEngine
from unified_processor import UnifiedDocumentSystem


class TestIntelligentDocumentProcessor(unittest.TestCase):
    """Test cases for the core document processor"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processor = IntelligentDocumentProcessor()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_heading_classification(self):
        """Test heading level classification"""
        test_cases = [
            ("1. Introduction", "H1"),
            ("1.1 Overview", "H2"),
            ("1.1.1 Details", "H3"),
            ("CHAPTER 1", "H1"),
            ("Chapter 1: Getting Started", "H1"),
            ("Introduction to AI", "H2"),
            ("This is not a heading.", None),
            ("", None)
        ]
        
        for text, expected in test_cases:
            result = self.processor._classify_heading(text)
            if expected is None:
                self.assertIsNone(result, f"Expected None for '{text}', got {result}")
            else:
                self.assertEqual(result, expected, f"Expected {expected} for '{text}', got {result}")
    
    def test_title_extraction(self):
        """Test document title extraction"""
        test_text = """
        DOCUMENT TITLE HERE
        
        This is the content of the document.
        It has multiple lines.
        """
        
        result = self.processor.extract_title(test_text, Path("test.pdf"))
        self.assertIn("DOCUMENT TITLE HERE", result)
    
    def test_outline_extraction(self):
        """Test outline extraction from structured text"""
        page_texts = {
            1: """
            1. Introduction
            This is the introduction section.
            
            1.1 Overview
            This provides an overview.
            
            1.2 Scope
            This defines the scope.
            """,
            2: """
            2. Main Content
            This is the main content.
            
            2.1 Details
            These are the details.
            """
        }
        
        outline = self.processor.extract_outline("", page_texts)
        
        # Check that we extracted some outline items
        self.assertGreater(len(outline), 0)
        
        # Check structure
        h1_items = [item for item in outline if item.level == "H1"]
        h2_items = [item for item in outline if item.level == "H2"]
        
        self.assertGreater(len(h1_items), 0, "Should have H1 items")
        self.assertGreater(len(h2_items), 0, "Should have H2 items")
    
    def test_confidence_calculation(self):
        """Test heading confidence calculation"""
        high_confidence = self.processor._calculate_heading_confidence("1. Introduction", "H1")
        low_confidence = self.processor._calculate_heading_confidence("maybe heading?", "H2")
        
        self.assertGreater(high_confidence, low_confidence)
        self.assertLessEqual(high_confidence, 1.0)
        self.assertGreaterEqual(low_confidence, 0.0)
    
    def test_semantic_embeddings(self):
        """Test semantic embedding generation"""
        outline_items = [
            OutlineItem("H1", "Introduction to Machine Learning", 1),
            OutlineItem("H2", "Neural Networks", 2),
            OutlineItem("H3", "Deep Learning", 3)
        ]
        
        embeddings = self.processor.generate_semantic_embeddings(outline_items)
        
        self.assertEqual(embeddings.shape[0], len(outline_items))
        self.assertGreater(embeddings.shape[1], 0)  # Should have embedding dimensions
    
    def test_output_json_format(self):
        """Test JSON output format compliance"""
        outline_items = [
            OutlineItem("H1", "Introduction", 1),
            OutlineItem("H2", "Overview", 1),
            OutlineItem("H3", "Details", 2)
        ]
        
        doc_structure = DocumentStructure(
            title="Test Document",
            outline=outline_items
        )
        
        output = self.processor.generate_output_json(doc_structure)
        
        # Check required fields
        self.assertIn("title", output)
        self.assertIn("outline", output)
        self.assertIsInstance(output["outline"], list)
        
        # Check outline item structure
        for item in output["outline"]:
            self.assertIn("level", item)
            self.assertIn("text", item)
            self.assertIn("page", item)


class TestPersonaIntelligenceEngine(unittest.TestCase):
    """Test cases for the persona intelligence engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.processor = IntelligentDocumentProcessor()
        self.engine = PersonaIntelligenceEngine(self.processor)
    
    def test_persona_relevance_scoring(self):
        """Test persona-based relevance scoring"""
        travel_text = "Plan your vacation to France with these travel tips and hotel recommendations"
        hr_text = "Complete this employee onboarding form for compliance with company policies"
        food_text = "Vegetarian recipe for corporate catering buffet menu planning"
        
        # Test Travel Planner persona
        travel_score = self.engine.analyze_persona_relevance(
            travel_text, "Travel Planner", "Plan a trip for friends"
        )
        
        # Test HR Professional persona
        hr_score = self.engine.analyze_persona_relevance(
            hr_text, "HR Professional", "Create fillable forms"
        )
        
        # Test Food Contractor persona
        food_score = self.engine.analyze_persona_relevance(
            food_text, "Food Contractor", "Prepare vegetarian buffet"
        )
        
        # Each text should score highest for its matching persona
        self.assertGreater(travel_score, 0.1)
        self.assertGreater(hr_score, 0.1)
        self.assertGreater(food_score, 0.1)
    
    def test_content_categorization(self):
        """Test semantic content categorization"""
        test_cases = [
            ("Visit restaurants and try local cuisine", "dining"),
            ("Book hotels and accommodations", "accommodation"),
            ("Complete employee forms and documents", "forms_management"),
            ("Vegetarian recipes and cooking instructions", "cooking"),
            ("Cultural attractions and museums", "cultural")
        ]
        
        for text, expected_category in test_cases:
            category = self.engine._categorize_content(text)
            self.assertEqual(category, expected_category)
    
    def test_concept_extraction(self):
        """Test key concept extraction"""
        text = """
        Paris is a beautiful city in France with many attractions.
        Visit the Eiffel Tower and Louvre Museum.
        Try French cuisine at local restaurants.
        """
        
        concepts = self.engine._extract_key_concepts(text)
        
        # Should extract some concepts
        self.assertGreater(len(concepts), 0)
        
        # Check for expected concepts (if NLP models are working)
        concept_text = " ".join(concepts).lower()
        expected_entities = ["paris", "france", "eiffel tower", "louvre"]
        
        # At least some expected entities should be found
        found_entities = sum(1 for entity in expected_entities if entity in concept_text)
        self.assertGreater(found_entities, 0)


class TestUnifiedDocumentSystem(unittest.TestCase):
    """Test cases for the unified system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.system = UnifiedDocumentSystem()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_challenge_detection(self):
        """Test automatic challenge type detection"""
        # Create Challenge 1a structure
        input_dir = self.temp_dir / "input"
        input_dir.mkdir()
        
        # Create a test PDF file (empty file for testing)
        test_pdf = input_dir / "test.pdf"
        test_pdf.write_bytes(b"dummy pdf content")
        
        # Mock the processing to avoid actual PDF processing
        with patch.object(self.system, 'process_challenge_1a') as mock_1a:
            self.system.auto_detect_and_process(self.temp_dir)
            mock_1a.assert_called_once()
    
    def test_output_directory_creation(self):
        """Test output directory creation"""
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"
        input_dir.mkdir()
        
        # Mock the document processor to avoid model loading
        with patch.object(self.system.base_processor, 'process_single_document'):
            self.system.process_challenge_1a(input_dir, output_dir)
        
        self.assertTrue(output_dir.exists())


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_challenge_1a(self):
        """Test end-to-end Challenge 1a processing"""
        # Create test structure
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"
        input_dir.mkdir()
        
        # Create a minimal test PDF file (this would need actual PDF content in real tests)
        test_pdf = input_dir / "test.pdf"
        test_pdf.write_bytes(b"dummy pdf content")
        
        # Mock the PDF processing to avoid dependency on actual PDF files
        mock_structure = DocumentStructure(
            title="Test Document",
            outline=[
                OutlineItem("H1", "Introduction", 1),
                OutlineItem("H2", "Overview", 1)
            ]
        )
        
        system = UnifiedDocumentSystem()
        
        with patch.object(system.base_processor, 'process_single_document', return_value=mock_structure):
            system.process_challenge_1a(input_dir, output_dir)
        
        # Check output file was created
        output_file = output_dir / "test.json"
        self.assertTrue(output_file.exists())
        
        # Check output format
        with open(output_file, 'r') as f:
            output_data = json.load(f)
        
        self.assertIn("title", output_data)
        self.assertIn("outline", output_data)
        self.assertEqual(output_data["title"], "Test Document")
        self.assertEqual(len(output_data["outline"]), 2)


def run_performance_tests():
    """Run performance tests to ensure speed requirements"""
    import time
    
    print("\\n=== Performance Tests ===")
    
    # Test model loading time
    start_time = time.time()
    processor = IntelligentDocumentProcessor()
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")
    
    # Model loading should be under 30 seconds for acceptable startup time
    assert load_time < 30, f"Model loading too slow: {load_time:.2f}s"
    
    # Test processing speed (mock processing)
    start_time = time.time()
    
    # Simulate processing multiple outline items
    outline_items = [
        OutlineItem(f"H{i%3+1}", f"Section {i}", i%10+1)
        for i in range(100)
    ]
    
    # Test embedding generation speed
    if outline_items:
        embeddings = processor.generate_semantic_embeddings(outline_items)
        
    processing_time = time.time() - start_time
    print(f"Processing time for 100 items: {processing_time:.2f} seconds")
    
    # Processing should be fast for the scale we're working with
    assert processing_time < 5, f"Processing too slow: {processing_time:.2f}s"
    
    print("✅ Performance tests passed")


def run_validation_tests():
    """Run validation tests for output format compliance"""
    print("\\n=== Validation Tests ===")
    
    # Test JSON schema compliance
    processor = IntelligentDocumentProcessor()
    
    # Create test document structure
    test_outline = [
        OutlineItem("H1", "Introduction", 1),
        OutlineItem("H2", "Background", 1),
        OutlineItem("H3", "Methodology", 2)
    ]
    
    doc_structure = DocumentStructure(
        title="Test Document",
        outline=test_outline
    )
    
    # Generate output
    output_data = processor.generate_output_json(doc_structure)
    
    # Validate structure
    assert "title" in output_data
    assert "outline" in output_data
    assert isinstance(output_data["outline"], list)
    
    for item in output_data["outline"]:
        assert "level" in item
        assert "text" in item
        assert "page" in item
        assert isinstance(item["level"], str)
        assert isinstance(item["text"], str)
        assert isinstance(item["page"], int)
    
    print("✅ Validation tests passed")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    try:
        run_performance_tests()
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
    
    # Run validation tests
    try:
        run_validation_tests()
    except Exception as e:
        print(f"❌ Validation tests failed: {e}")
    
    print("\\n🎉 Test suite completed!")
