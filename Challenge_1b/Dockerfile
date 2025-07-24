# Challenge 1b: Multi-Collection PDF Analysis
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
COPY process_challenge_1b.py .

# Copy all collection data
COPY Collection* ./

# Set environment variables
ENV PYTHONPATH=/app

# Run the processor that analyzes all collections
CMD ["python", "process_challenge_1b.py"]
