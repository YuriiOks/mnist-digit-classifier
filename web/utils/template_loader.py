# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/template_loader.py
# Description: Utility for loading HTML templates
# Created: 2024-05-01

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TemplateLoader:
    @staticmethod
    def load_template(template_path: str) -> str:
        """Load an HTML template from a file.

        Args:
            template_path: Path to the template file, relative to assets/templates directory

        Returns:
            str: The template content as a string
        """

        try:
            # Get the project root directory (app directory in Docker)
            project_root = Path(__file__).resolve().parents[1]

            # Strip leading slash if present to ensure relative path
            template_path = template_path.lstrip("/")

            # Construct the full path to the template file
            full_template_path = project_root / "templates" / template_path

            if not full_template_path.exists():
                # Try alternative path structure
                alt_template_path = project_root / "assets" / "templates" / template_path
                logger.info(f"Trying alternative path: {alt_template_path}")

                if not alt_template_path.exists():
                    logger.warning(f"Template file not found at either location: {full_template_path} or {alt_template_path}")
                    return ""

                full_template_path = alt_template_path

            return full_template_path.read_text()
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {e}")
            return ""

    @staticmethod
    def render_template(template_path: str, context: Dict[str, Any]) -> str:
        template_content = TemplateLoader.load_template(template_path)
        if not template_content:
            return ""

        try:
            return template_content.format(**context)
        except Exception as e:
            logger.error(f"Error rendering template {template_path}: {e}")
            return template_content

