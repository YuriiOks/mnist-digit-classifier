# MNIST Digit Classifier - web/Dockerfile
# Corrected for production deployment

FROM python:3.11-slim

# Install git (if still needed by any dependency)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* # Clean up apt cache

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
# Install web app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all web application files from build context (./web)
COPY . .

# Make directories if they weren't copied or are needed
# Note: These paths might differ based on your actual web app structure
RUN mkdir -p /app/static /app/ui /app/core /app/services

# Expose Streamlit port
EXPOSE 8501

# Environment variables
ENV PYTHONUNBUFFERED=1

# Entrypoint using Streamlit run
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]