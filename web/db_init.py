import os
import logging
import psycopg2
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration from environment variables
db_config = {
    "host": os.environ.get("DB_HOST", "db"),
    "database": os.environ.get("DB_NAME", "digit_predictions"),
    "user": os.environ.get("DB_USER", "mnist_app"),
    "password": os.environ.get("DB_PASSWORD", "secure_password"),
    "port": os.environ.get("DB_PORT", "5432")
}

# SQL to create table
CREATE_TABLE_SQL = """
-- Create prediction_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction INTEGER NOT NULL,
    true_label INTEGER,
    confidence FLOAT NOT NULL,
    input_type VARCHAR(20),
    image_data TEXT
);

-- Create index for faster querying
CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_logs (timestamp DESC);
"""

def initialize_database():
    """Initialize the database tables."""
    max_attempts = 5
    attempt = 0
    
    # Try to connect with retry mechanism
    while attempt < max_attempts:
        attempt += 1
        try:
            logger.info(f"Attempting to connect to database (attempt {attempt}/{max_attempts})...")
            conn = psycopg2.connect(**db_config)
            conn.autocommit = True
            
            with conn.cursor() as cursor:
                logger.info("Creating tables...")
                cursor.execute(CREATE_TABLE_SQL)
            
            conn.close()
            logger.info("Database initialization completed successfully")
            return True
        except psycopg2.OperationalError as e:
            logger.warning(f"Database connection failed: {str(e)}")
            if attempt < max_attempts:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Maximum connection attempts reached. Database initialization failed.")
                return False
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False

if __name__ == "__main__":
    initialize_database()