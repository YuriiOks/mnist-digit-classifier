# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/data_loader.py
# Description: Utility for loading data from JSON files
# Created: 2024-05-01

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class DataLoader:
    @staticmethod
    def load_json(data_path: str) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """Load JSON data from a file.
        
        Args:
            data_path: Path to the JSON file, relative to assets/data directory
            
        Returns:
            The parsed JSON data, or None if the file couldn't be loaded
        """

        try:
            # Get the project root directory (app directory in Docker)
            project_root = Path(__file__).resolve().parents[1]  # Up to the web/ directory
            
            # Strip leading slash if present to ensure relative path
            data_path = data_path.lstrip("/")

            # Construct the full path to the data file
            full_data_path = project_root / "data" / data_path
            
            logger.info(f"Project root: {full_data_path}")
            # logger.info(f"Looking for JSON data at: {full_data_path}")

            if not full_data_path.exists():
                # Try alternative path structure
                alt_data_path = project_root / "assets" / "data" / data_path
                logger.info(f"Trying alternative path: {alt_data_path}")
                
                if not alt_data_path.exists():
                    logger.warning(f"Data file not found at either location: {full_data_path} or {alt_data_path}")
                    return None
                    
                full_data_path = alt_data_path

            # Load and parse the JSON data
            with open(full_data_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading JSON data '{data_path}': {str(e)}", exc_info=True)
            return None