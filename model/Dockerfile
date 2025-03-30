FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Install dependencies including gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for saved models
RUN mkdir -p saved_models 

# Volume for persistent model storage
VOLUME /app/model/saved_models

# Expose the port for API
EXPOSE 5000

# Start the API server
CMD ["python", "app.py"]