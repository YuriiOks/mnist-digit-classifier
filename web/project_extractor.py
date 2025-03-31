#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Structure and Content Extractor

This script traverses a project directory, extracts its structure,
and saves the structure along with the content of specified file types
(Python, SQL, Dockerfiles, docker-compose) to a single markdown file.

Author: Your Name / Team Name
Date: 2023-10-27
Version: 1.1.0
"""

import os
import argparse
import datetime
from pathlib import Path
from typing import List, Optional, Set

# Default directories to ignore during traversal
DEFAULT_IGNORE_DIRS: List[str] = [
    '.git', '__pycache__', 'venv', 'env', 'node_modules', '.vscode', '.idea',
    'dist', 'build', '*.egg-info',
    'data',             # Ignore potentially large raw data folders
    'outputs',          # Ignore generated outputs/results
    'model/saved_models', # Ignore potentially large model files
    'model/logs',       # Ignore training logs
    '.pytest_cache',
    '.mypy_cache',
    # Add any other non-essential / generated directories
]

# File types/names to automatically include content for
DEFAULT_INCLUDE_PATTERNS: Set[str] = {
    '.py',          # Python files
    '.sql',         # SQL files
    '.js',         # JavaScript files
    '.html',       # HTML files
    '.css',        # CSS files
    'dockerfile',   # Dockerfiles (exact filename match, case-insensitive)
    '.dockerignore',# Docker ignore files
    'docker-compose.yml', # Docker compose files
    'docker-compose.yaml',# Docker compose files (alternative extension)
    'requirements.txt', # Python requirements
    '.env.example', # Example environment files
    'config.toml',  # Common config like Streamlit
    'pyproject.toml', # Project metadata and dependencies
    'setup.py',     # Legacy setup files
    'setup.cfg',    # Setup configuration
    '.sh',          # Shell scripts
    '.md',          # Markdown files (like READMEs)
    '.json',        # JSON files (often configs)
    '.yaml',        # YAML files (often configs)
    '.yml',         # YAML files (alternative extension)
    'Makefile',     # Makefiles
}

def is_binary_file(file_path: str) -> bool:
    """
    Check if a file seems to be binary by trying to read it as UTF-8.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if the file is likely binary, False otherwise.
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            chunk.decode('utf-8')
        return False
    except UnicodeDecodeError:
        return True
    except Exception:
        # Handle other potential read errors if necessary, assume binary for safety
        return True


def get_file_extension(file_path: str) -> str:
    """
    Get a file's extension in lowercase, without the leading dot.

    Args:
        file_path: Path to the file.

    Returns:
        The file extension (e.g., 'py', 'txt') or an empty string if no extension.
    """
    return os.path.splitext(file_path)[1][1:].lower()


def get_language_for_markdown(file_path: str) -> str:
    """
    Get the appropriate language identifier for markdown code blocks based on file extension or name.

    Args:
        file_path: Path to the file.

    Returns:
        Language identifier string for markdown (e.g., 'python', 'yaml').
    """
    filename = os.path.basename(file_path).lower()
    ext = get_file_extension(file_path)

    # Handle specific filenames first
    if filename == 'dockerfile':
        return 'dockerfile'
    if filename == 'makefile':
        return 'makefile'
    if 'docker-compose' in filename and (ext == 'yml' or ext == 'yaml'):
        return 'yaml'

    # Map extensions to language identifiers
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
        'txt': 'text', # Default for requirements.txt etc.
        'cfg': 'ini', # For setup.cfg
    }
    return language_map.get(ext, '') # Return empty string if no match


def should_skip_dir(dir_path: str, ignore_dirs: List[str]) -> bool:
    """
    Check if a directory should be skipped based on ignore patterns.

    Args:
        dir_path: Absolute path to the directory.
        ignore_dirs: List of directory names or partial paths to ignore.

    Returns:
        True if the directory should be skipped, False otherwise.
    """
    dir_name = os.path.basename(dir_path)

    # Skip hidden directories (except specific ones if needed, e.g., .streamlit)
    if dir_name.startswith('.') and dir_name.lower() != '.streamlit':
        return True

    # Check against exact directory name matches provided in ignore_dirs
    if dir_name in ignore_dirs:
        return True

    # Check if the directory name contains common temporary/build patterns
    if "debug" in dir_name.lower() or "build" in dir_name.lower() or dir_name.endswith('.egg-info'):
         return True

    # Check if the *full path* contains any of the ignore_dirs patterns
    # This helps catch nested ignored dirs like 'project/src/generated/__pycache__'
    normalized_dir_path = dir_path.replace('\\', '/')
    for ignore_pattern in ignore_dirs:
        if f'/{ignore_pattern}/' in f'/{normalized_dir_path}/': # Check as path components
             return True

    return False

def should_include_content(file_path: str, include_patterns: Set[str]) -> bool:
    """
    Check if the content of a file should be included based on its name or extension.

    Args:
        file_path: Absolute path to the file.
        include_patterns: Set of patterns (extensions like '.py' or filenames like 'dockerfile') to include.

    Returns:
        True if the file content should be included, False otherwise.
    """
    filename = os.path.basename(file_path).lower()
    ext = f".{get_file_extension(file_path)}" # Add dot back for pattern matching

    if filename in include_patterns:
        return True
    if ext in include_patterns and ext != ".": # Check extension if it exists
        return True

    return False


def extract_project(root_dir: str, output_file: str, ignore_dirs: List[str], max_file_size_kb: int):
    """
    Extract project structure and specified file contents to a markdown file.

    Args:
        root_dir: Root directory of the project.
        output_file: Output markdown file path.
        ignore_dirs: List of directory names or paths to ignore.
        max_file_size_kb: Maximum file size in KB to include content for.
    """
    root_dir = os.path.abspath(root_dir)
    root_dir_name = os.path.basename(root_dir)
    max_bytes = max_file_size_kb * 1024

    files_to_include_content: List[tuple[str, str]] = [] # List of (relative_path, absolute_path)

    print("Scanning project structure...")
    structure_lines = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Filter directories in place using should_skip_dir
        dirs[:] = [d for d in dirs if not should_skip_dir(os.path.join(root, d), ignore_dirs)]

        level = root.replace(root_dir, '').count(os.sep)
        indent = '    ' * level # Use 4 spaces for indentation
        rel_path_dir = os.path.relpath(root, root_dir).replace('\\', '/')
        dir_name = os.path.basename(root) if rel_path_dir != '.' else root_dir_name

        # Add directory to structure
        structure_lines.append(f"{indent}{dir_name}/")

        sub_indent = '    ' * (level + 1)
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            rel_file_path = os.path.relpath(file_path, root_dir).replace('\\', '/')

            # Check if this file's content should be included
            include_content_flag = should_include_content(file_path, DEFAULT_INCLUDE_PATTERNS)
            file_info = ""

            if include_content_flag:
                 # Check size limit *before* deciding to include
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > max_bytes:
                        file_info += f" [SIZE LIMIT EXCEEDED > {max_file_size_kb}KB]"
                        include_content_flag = False # Don't include content if too large
                    elif is_binary_file(file_path):
                         file_info += " [BINARY]"
                         include_content_flag = False # Don't include content if binary
                    else:
                         files_to_include_content.append((rel_file_path, file_path))
                         file_info += " [CONTENT INCLUDED]"
                except OSError:
                    file_info += " [ERROR ACCESSING]"
                    include_content_flag = False # Cannot include if error

            # Optionally add size info for large files even if not included
            elif not file_info: # Only if no other info added yet
                 try:
                      file_size_kb = os.path.getsize(file_path) / 1024
                      if file_size_kb > 1000: # Show size if > 1MB
                          file_info += f" [{file_size_kb/1000:.1f} MB]"
                 except OSError:
                     file_info += " [SIZE UNKNOWN]"


            structure_lines.append(f"{sub_indent}{filename}{file_info}")

    print(f"Found {len(files_to_include_content)} files to include content for.")

    # --- Writing Output File ---
    print(f"Writing output to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            # Write Header
            out_file.write(f"# Project Structure and Content: {root_dir_name}\n\n")
            out_file.write(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
            out_file.write(f"*Max file size for content inclusion: {max_file_size_kb} KB*\n\n")

            # Write Table of Contents
            out_file.write("## Table of Contents\n\n")
            out_file.write("1. [Directory Structure](#directory-structure)\n")
            out_file.write("2. [Included File Contents](#included-file-contents)\n\n")

            # Write Directory Structure
            out_file.write("## Directory Structure\n\n")
            out_file.write("```plaintext\n")
            out_file.write(f"{root_dir_name}/\n") # Add root dir explicitly
            out_file.write("\n".join(structure_lines))
            out_file.write("\n```\n\n")

            # Write Included File Contents
            out_file.write("## Included File Contents\n\n")
            if not files_to_include_content:
                out_file.write("*No files matched the inclusion criteria or size/binary limits.*\n\n")
            else:
                 # Sort by relative path for consistent order
                files_to_include_content.sort(key=lambda item: item[0])

                for rel_path, abs_path in files_to_include_content:
                    out_file.write(f"### `{rel_path}`\n\n")
                    language = get_language_for_markdown(abs_path)
                    out_file.write(f"```{language}\n")
                    try:
                        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f_content:
                            content = f_content.read()
                            # Ensure content ends with a newline for clean markdown formatting
                            if content and not content.endswith('\n'):
                                content += '\n'
                            out_file.write(content if content else "[EMPTY FILE]\n")
                    except Exception as e:
                        out_file.write(f"```\n\n[Error reading file: {e}]\n\n")
                        print(f"⚠️ Error reading content from {abs_path}: {e}")
                        continue # Skip the closing backticks if read failed severely

                    out_file.write("```\n\n")

    except IOError as e:
        print(f"❌ Error writing to output file {output_file}: {e}")
        return # Stop execution if cannot write output file
    except Exception as e:
        print(f"❌ An unexpected error occurred during output generation: {e}")
        # Optionally write error to file if it's open, or just exit
        return

    print(f"✅ Successfully extracted structure and contents to {output_file}")


def main():
    """Main function to parse arguments and run the extraction."""
    parser = argparse.ArgumentParser(
        description='Extract project structure and specified file contents to a markdown file.',
        formatter_class=argparse.RawTextHelpFormatter # Preserve newline formatting in help
        )
    parser.add_argument(
        '--dir', '-d', default='.',
        help='Project root directory (default: current directory)'
        )
    parser.add_argument(
        '--output', '-o', default='project_summary.md',
        help='Output markdown file name (default: project_summary.md)'
        )
    parser.add_argument(
        '--max-size', '-s', type=int, default=1000,
        help='Max file size in KB to include content (default: 1000 KB)'
        )
    parser.add_argument(
        '--ignore-dirs', nargs='*', default=[],
        help=f"Space-separated list of additional directory names to ignore.\nDefaults include: {', '.join(DEFAULT_IGNORE_DIRS)}"
        )

    args = parser.parse_args()

    # Combine default ignore list with user-provided list
    ignore_dirs = list(set(DEFAULT_IGNORE_DIRS + args.ignore_dirs)) # Use set to avoid duplicates

    print(f"⚙️  Starting project extraction...")
    print(f"    Root Directory: {os.path.abspath(args.dir)}")
    print(f"    Output File: {args.output}")
    print(f"    Max Content Size: {args.max_size} KB")
    print(f"    Ignoring Directories: {', '.join(ignore_dirs)}")
    print(f"    Including Content For Types: {', '.join(sorted(DEFAULT_INCLUDE_PATTERNS))}")

    extract_project(
        root_dir=args.dir,
        output_file=args.output,
        ignore_dirs=ignore_dirs,
        max_file_size_kb=args.max_size
    )

if __name__ == "__main__":
    main()