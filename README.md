# Adobe India Hackathon 2025 - PDF Processing Solution

## Intelligent Document Structure Extraction

A high-performance, AI-powered PDF processing system that extracts structured outlines from documents with blazing speed and pinpoint accuracy.

## Our Approach

### Multi-Method Heading Detection
Our solution employs a sophisticated multi-layered approach to accurately identify headings and document structure:

1. **Font-Based Analysis**: Analyzes character-level font properties (size, weight, family) to identify visual heading patterns
2. **Semantic Intelligence**: Uses transformer models (SentenceTransformer) to understand heading-like content semantically
3. **Pattern Recognition**: Employs regex patterns to identify common heading structures (numbered sections, chapter titles, etc.)
4. **Position Analysis**: Considers document layout and positioning to improve detection accuracy
5. **Multi-Method Validation**: Combines results from multiple detection methods with confidence scoring

### Advanced Content Filtering
- **False Positive Elimination**: Strict validation rules to filter out financial data, table content, and sentence fragments
- **OCR Artifact Cleaning**: Removes scanning artifacts and duplicate characters
- **Context-Aware Processing**: Understands document context to improve heading classification

## Models and Libraries Used

### Core Dependencies
- **PyPDF2** & **pdfplumber**: PDF text extraction and character-level font analysis
- **sentence-transformers**: Semantic understanding using `all-MiniLM-L6-v2` model
- **spaCy**: Natural language processing and text analysis
- **NLTK**: Text preprocessing and tokenization
- **scikit-learn**: Feature extraction and similarity calculations

### AI Models
- **all-MiniLM-L6-v2**: Lightweight transformer model for semantic similarity
- **en_core_web_sm**: English language model for NLP tasks

## How to Build and Run

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
   ```

2. **Run the solution:**
   ```bash
   docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
   ```

### Expected Behavior
- Automatically processes all PDFs from `/app/input` directory
- Generates corresponding `filename.json` files in `/app/output` for each `filename.pdf`
- Each JSON contains structured outline with headings, levels, and page numbers

### Output Format
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Subheading",
      "page": 2
    }
  ]
}
```

## Performance Characteristics
- **Speed**: Sub-second processing for most documents
- **Accuracy**: Multi-method validation ensures high precision
- **Scalability**: Efficient memory usage and batch processing capabilities
- **Robustness**: Handles irregular PDFs with fallback detection methods

## Architecture Highlights
- **Dockerized Deployment**: Platform-independent containerized solution
- **Model Caching**: Pre-downloads AI models during build time for faster runtime
- **Error Handling**: Graceful failure recovery and comprehensive logging
- **Memory Optimization**: Efficient processing of large document collections

---

## Challenge Solutions

### [Challenge 1a: PDF Processing Solution](./Challenge_1a/README.md)
Basic PDF processing with Docker containerization and structured data extraction.

### [Challenge 1b: Multi-Collection PDF Analysis](./Challenge_1b/README.md)
Advanced persona-based content analysis across multiple document collections.

---

**Note**: Each challenge directory contains detailed documentation and implementation details. Please refer to the individual README files for comprehensive information about each solution.