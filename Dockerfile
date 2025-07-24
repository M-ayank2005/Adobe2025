# Adobe2025 PDF Processing Solution
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY core_system/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache models during build time to reduce runtime overhead
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY core_system/intelligent_document_processor.py .
COPY core_system/semantic_intelligence.py .

# Copy the main processing script
COPY main_processor.py .

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Run the main processor that processes all PDFs from /app/input to /app/output
CMD ["python", "main_processor.py"]
