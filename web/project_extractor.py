#!/usr/bin/env python3
"""
Refined Project Structure and Content Extractor

This script traverses a directory, extracts its structure, and saves the content
of specific files to a single markdown file.
"""

import os
import sys
import argparse
import datetime
from pathlib import Path


def is_binary_file(file_path):
    """
    Check if a file is binary.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if the file is binary, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)  # Try to read some of the file as text
        return False
    except UnicodeDecodeError:
        return True


def get_file_extension(file_path):
    """
    Get a file's extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: The file extension
    """
    return os.path.splitext(file_path)[1][1:].lower()


def get_language_for_markdown(file_path):
    """
    Get the appropriate language identifier for markdown code blocks.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Language identifier for markdown
    """
    ext = get_file_extension(file_path)
    language_map = {
        'py': 'python',
        'js': 'javascript',
        'html': 'html',
        'css': 'css',
        'json': 'json',
        'md': 'markdown',
        'yml': 'yaml',
        'yaml': 'yaml',
        'sh': 'bash',
        'bash': 'bash',
        'sql': 'sql',
        'java': 'java',
        'c': 'c',
        'cpp': 'cpp',
        'cs': 'csharp',
        'go': 'go',
        'rs': 'rust',
        'php': 'php',
        'rb': 'ruby',
        'ts': 'typescript',
        'toml': 'toml',
        'dockerfile': 'dockerfile',
    }
    
    # Special case for Dockerfile without extension
    if os.path.basename(file_path).lower() == 'dockerfile':
        return 'dockerfile'
    
    return language_map.get(ext, '')


def should_skip_dir(dir_path, ignore_dirs):
    """
    Check if a directory should be skipped when generating the directory structure.
    
    Args:
        dir_path: Path to the directory
        ignore_dirs: List of directories to ignore
        
    Returns:
        bool: True if the directory should be skipped, False otherwise
    """
    dir_name = os.path.basename(dir_path)
    
    # Skip hidden directories except .streamlit
    if dir_name.startswith('.') and not dir_name == '.streamlit':
        return True
    
    # Skip exact directory name matches
    if dir_name in ignore_dirs:
        return True
    
    # Skip anything with "debug" in the name (case-insensitive)
    if "debug" in dir_name.lower():
        return True
        
    # Skip if the directory path contains any of the ignore directories
    for ignore_dir in ignore_dirs:
        if isinstance(ignore_dir, str) and ignore_dir.strip() in dir_path:
            return True
    
    return False


def extract_project(root_dir, output_file, include_files=None, ignore_dirs=None, max_file_size_kb=5000):
    """
    Extract project structure and file contents to a markdown file.

    Args:
        root_dir: Root directory of the project
        output_file: Output markdown file path
        include_files: List of files to include with full content (relative paths starting with /)
        ignore_dirs: List of directories to ignore in structure
        max_file_size_kb: Maximum file size to include content for (in KB)
    """
    # --- Refined List of Files to Include ---
    if include_files is None:
        include_files = [
            # === Core ML / Model Definition ===
            '/model/model.py',             # The NN architecture itself

            # === Model Training & Evaluation ===
            '/model/train.py',             # How the model is trained
            '/model/utils/evaluation.py',        # Evaluation functions (accuracy, plotting)
            '/model/utils/augmentation.py',      # Data augmentation strategies used in training

            # === Preprocessing (Crucial for Consistency) ===
            '/utils/preprocessing.py',     # Image preprocessing logic (used by train & inference)

            # === Model Inference Service (Flask API) ===
            '/model/app.py',               # Flask API endpoint code (/predict)
            '/model/inference.py',         # Predictor class (loads model, uses preprocessing)
            '/model/requirements.txt',     # Dependencies for the model service container
            '/model/Dockerfile',           # How the model service container is built

            # === Web Application (Streamlit Frontend) ===
            '/web/app.py',                 # Main Streamlit app entry point
            '/web/model/digit_classifier.py', # Client class calling the model API
            '/web/ui/views/draw_view.py',  # Example view using the canvas/prediction (or relevant view)
            '/web/ui/views/history_view.py', # Example view for showing prediction history
            '/web/services/prediction/prediction_service.py', # Handles DB logging for web app
            '/web/requirements.txt',       # Dependencies for the web service container
            '/web/Dockerfile',             # How the web service container is built
            '/web/.streamlit/config.toml', # Streamlit configuration (if relevant)

            # === Database ===
            '/database/init.sql',          # Database schema initialization

            # === Orchestration & Deployment ===
            '/docker-compose.yml',         # Defines services, network, volumes

            # === Environment & Utilities ===
            '/utils/environment_setup.py', # Environment setup logic (PyTorch, MPS, seeds)
            '/utils/mps_verification.py',  # Script to check MPS performance (context)

            # === Testing & Benchmarking ===
            '/scripts/benchmark_model.py', # Script for performance benchmarking
            '/model/tests/test_model.py',        # Unit test for model architecture
            '/model/tests/test_inference.py',    # Unit test for predictor class
            '/model/tests/test_preprocessing.py', # Unit test for preprocessing (important!)
            '/tests/test_model_integration.py',  # Integration test (e.g., web calls model)

            # === Project Setup & Root Config ===
            '/README.md',                  # Project overview
            '/.gitignore',                 # Shows what's intentionally excluded
            '/requirements.txt',           # Top-level dependencies (dev/utility tools)
            '/scripts/check_environment.sh', # Maybe useful for seeing env checks performed
            '/scripts/run_mps_verification.sh', # Script that runs the mps_verification.py
        ]

    # Default directories to ignore
    if ignore_dirs is None:
        ignore_dirs = [
            '.git', '__pycache__', 'venv', 'mnist_env', '.vscode', '.idea',
            'data', # Ignore potentially large raw data folder
            'outputs', # Ignore generated outputs
            'model/saved_models', # Ignore potentially large model files
            'model/logs', # Ignore training logs in structure
            # Add any other non-essential / generated directories
            'web/assets/images/icons', # Example: Ignore potentially many small icon files
        ]
    
    # Normalize paths
    normalized_include_files = []
    for f in include_files:
        if f.startswith('/'):
            normalized_include_files.append(f[1:])  # Remove leading slash
        else:
            normalized_include_files.append(f)
    
    if ignore_dirs is None:
        ignore_dirs = ['venv', 'env', '__pycache__', 'node_modules', '.git', '.idea', '.vscode']
    
    # Normalize the directory paths
    ignore_dirs = [d.strip() if isinstance(d, str) else d for d in ignore_dirs]
    
    root_dir = os.path.abspath(root_dir)
    root_dir_name = os.path.basename(root_dir)
    
    # Map of file paths to find
    file_paths_found = {}
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # Write header
        out_file.write(f"# Project Structure and Content: {root_dir_name}\n\n")
        out_file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write table of contents
        out_file.write("## Table of Contents\n\n")
        out_file.write("1. [Directory Structure](#directory-structure)\n")
        out_file.write("2. [File Contents](#file-contents)\n\n")
        
        # Write directory structure
        out_file.write("## Directory Structure\n\n")
        out_file.write("```\n")
        
        # First pass: generate directory structure and collect absolute paths of included files
        for root, dirs, files in os.walk(root_dir):
            # Skip directories that match ignore patterns
            dirs[:] = [d for d in dirs if not should_skip_dir(os.path.join(root, d), ignore_dirs)]
            
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            rel_path = os.path.relpath(root, root_dir) if root != root_dir else ''
            dir_name = os.path.basename(root) if rel_path else root_dir_name
            
            out_file.write(f"{indent}{dir_name}/\n")
            
            # List all files in this directory
            for f in sorted(files):
                file_path = os.path.join(root, f)
                rel_file_path = os.path.relpath(file_path, root_dir).replace('\\', '/')  # Normalize path separators
                
                # Check if this file should be included with full content
                is_included = rel_file_path in normalized_include_files
                
                # Mark files that will have their content included
                if is_included:
                    file_paths_found[rel_file_path] = file_path
                    file_info = " [INCLUDED]"
                else:
                    file_info = ""
                    
                # Add file size info for larger files
                file_size_kb = os.path.getsize(file_path) / 1024
                if file_size_kb > 1000:  # Only show size for files > 1MB
                    file_info += f" [{file_size_kb/1000:.1f} MB]"
                
                out_file.write(f"{indent}    {f}{file_info}\n")
        
        out_file.write("```\n\n")
        
        # Write file contents for included files
        out_file.write("## File Contents\n\n")
        
        # Check for missing files
        missing_files = set(normalized_include_files) - set(file_paths_found.keys())
        if missing_files:
            out_file.write("### Missing Files\n\n")
            out_file.write("The following files were specified for inclusion but not found in the project:\n\n")
            for missing in sorted(missing_files):
                out_file.write(f"- `{missing}`\n")
            out_file.write("\n")
        
        # Write contents of found files
        for rel_path, abs_path in sorted(file_paths_found.items()):
            out_file.write(f"### {rel_path}\n\n")
            
            file_size_kb = os.path.getsize(abs_path) / 1024
            
            if file_size_kb > max_file_size_kb:
                out_file.write(f"*Large file: {file_size_kb:.1f} KB - content not included*\n\n")
                continue
            
            if is_binary_file(abs_path):
                out_file.write("*Binary file - content not included*\n\n")
                continue
            
            language = get_language_for_markdown(abs_path)
            out_file.write(f"```{language}\n")
            
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    out_file.write(content)
                    if not content.endswith('\n'):
                        out_file.write('\n')
            except Exception as e:
                out_file.write(f"Error reading file: {str(e)}\n")
            
            out_file.write("```\n\n")


def main():
    parser = argparse.ArgumentParser(description='Extract project structure and contents to a markdown file.')
    parser.add_argument('--dir', '-d', default='.', help='Project root directory (default: current directory)')
    parser.add_argument('--output', '-o', default='project_structure.md', help='Output markdown file (default: project_structure.md)')
    parser.add_argument('--max-size', '-s', type=int, default=5000, help='Max file size in KB to include content (default: 5000)')
    parser.add_argument('--ignore-dirs', nargs='*', help='Additional directories to ignore')
    parser.add_argument('--include-files', nargs='*', help='Specific files to include with content (relative paths)')
    
    args = parser.parse_args()
    
    ignore_dirs = ['venv', 'env', '__pycache__', 'node_modules', '.git', '.idea', '.vscode']
    if args.ignore_dirs:
        ignore_dirs.extend(args.ignore_dirs)
    
    include_files = None
    if args.include_files:
        include_files = args.include_files
    
    print(f"Extracting project structure from {args.dir}")
    print(f"Writing to {args.output}")
    
    extract_project(
        args.dir, 
        args.output, 
        include_files=include_files,
        ignore_dirs=ignore_dirs,
        max_file_size_kb=args.max_size
    )
    
    print(f"Done! Structure and contents extracted to {args.output}")


if __name__ == "__main__":
    main()