FROM python:3.9-slim

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Make directories if they don't exist
RUN mkdir -p /app/static
RUN mkdir -p /app/components
RUN mkdir -p /app/utils

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set up entry point that allows container restart
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]