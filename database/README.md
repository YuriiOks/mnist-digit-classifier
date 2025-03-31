Okay, here is the draft for the database/README.md.

🎯 Summary:
This README explains the purpose and contents of the database/ directory, focusing on how the init.sql script defines the schema for the PostgreSQL database used by the application and how it's integrated via Docker Compose.

📝 database/README.md Content:

# Database Initialization 💾

This directory contains the necessary SQL script(s) to initialize the schema for the PostgreSQL database used by the MNIST Digit Classifier application.

## 📄 Contents

*   `init.sql`: The primary SQL script executed by the PostgreSQL container on its first run. It defines the structure of the `prediction_logs` table.
*   `utils/`: (Optional) May contain database utility scripts (e.g., for migrations, backups - *currently empty in the project structure*).
*   `tests/`: (Optional) May contain database-related tests (e.g., schema validation - *currently empty in the project structure*).

## 📊 Schema Definition (`init.sql`)

The `init.sql` script ensures the `prediction_logs` table exists with the correct columns. This table stores information about each prediction made through the web application.

```sql
-- Create prediction_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,                     -- Auto-incrementing integer ID
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Time the prediction was logged
    prediction INTEGER NOT NULL,              -- The digit predicted by the model (0-9)
    true_label INTEGER,                       -- The user-provided correct digit (optional)
    confidence FLOAT NOT NULL,                -- The model's confidence score (0.0-1.0)
    input_type VARCHAR(20),                   -- How the digit was input ('canvas', 'upload', 'url')
    image_data TEXT                           -- Base64 encoded representation of the input image (optional)
);

-- Create index for faster querying by timestamp
CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_logs (timestamp DESC);
```

🐳 Integration with Docker Compose

The database initialization is handled automatically by Docker Compose when the db service (PostgreSQL container) starts for the first time.

In the main project docker-compose.yml file, the database/init.sql script is mounted into the /docker-entrypoint-initdb.d/ directory inside the container:

services:
  db:
    # ... other db config ...
    volumes:
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql # <-- This line mounts the script
    # ... rest of db config ...

> **Note**: The `init.sql` script will only be run when the container is first created, ensuring your database table is properly set up from the start. Subsequent restarts will not re-run the script unless you remove the existing container/volume.

---

## 📚 Related Documentation

- [Main Project README](../README.md)
- [Model README](../model/README.md)
- [Web README](../web/README.md)
