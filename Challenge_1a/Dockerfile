# Challenge 1a: PDF Processing Solution
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the processing script
COPY process_pdfs.py .

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONPATH=/app

# Run the processor that processes all PDFs from /app/input to /app/output
CMD ["python", "process_pdfs.py"]