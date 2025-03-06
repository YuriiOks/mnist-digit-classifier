-- Create prediction_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction INTEGER NOT NULL,
    true_label INTEGER,
    confidence FLOAT NOT NULL
);

-- Create index for faster querying
CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_logs (timestamp DESC);
