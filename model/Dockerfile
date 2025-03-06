FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for saved models
RUN mkdir -p saved_models

# Volume for persistent model storage
VOLUME /app/saved_models

# Expose the port for API
EXPOSE 5000

# Start the API server
CMD ["python", "app.py"]
