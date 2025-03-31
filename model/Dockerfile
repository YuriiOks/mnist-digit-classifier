# MNIST Digit Classifier - model/Dockerfile
# Corrected for production deployment

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
# Install dependencies (MAKE SURE gunicorn is in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL application code and model files from build context (./model)
COPY . .

# Create necessary directories INSIDE /app
# Note: saved_models and utils likely created by COPY, but mkdir -p is safe.
# Add outputs/debug_images IF you keep debug saving enabled in app.py
RUN mkdir -p /app/models/saved_models /app/utils /app/outputs/debug_images

# Expose the port Gunicorn will listen on
EXPOSE 5000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app

# CMD using Gunicorn for production
# Runs the 'app' instance from the 'app.py' file
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]