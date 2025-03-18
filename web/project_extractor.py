#!/usr/bin/env python3
"""
Project Structure and Content Extractor

This script traverses a directory, extracts its structure, and saves the content
of all files to a single markdown file.
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
import mimetypes


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


def should_skip_file(file_path, ignore_patterns):
    """
    Check if a file should be skipped based on patterns.
    
    Args:
        file_path: Path to the file
        ignore_patterns: List of patterns to ignore
        
    Returns:
        bool: True if the file should be skipped, False otherwise
    """
    file_name = os.path.basename(file_path)
    
    # Skip hidden files and directories (except .streamlit)
    if file_name.startswith('.') and not file_name == '.streamlit':
        return True
    
    # Skip if file extension is in ignore patterns
    ext = os.path.splitext(file_name)[1]
    if ext in ignore_patterns:
        return True
    
    # Skip exact filename matches
    if file_name in ignore_patterns:
        return True
    
    # Skip anything with "debug" in the name (case-insensitive)
    if "debug" in file_name.lower():
        return True
        
    # Skip if the file path contains any of the ignore patterns
    for pattern in ignore_patterns:
        # Make sure pattern is a string and strip any whitespace
        if isinstance(pattern, str) and pattern.strip() in file_path:
            return True
    
    return False


def should_skip_dir(dir_path, ignore_dirs):
    """
    Check if a directory should be skipped.
    
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


def extract_project(root_dir, output_file, ignore_dirs=None, ignore_patterns=None, max_file_size_kb=500):
    """
    Extract project structure and file contents to a markdown file.
    
    Args:
        root_dir: Root directory of the project
        output_file: Output markdown file path
        ignore_dirs: List of directories to ignore
        ignore_patterns: List of file patterns to ignore
        max_file_size_kb: Maximum file size to include (in KB)
    """
    if ignore_dirs is None:
        ignore_dirs = ['venv', 'env', '__pycache__', 'node_modules', '.git', '.idea', '.vscode', 'static_folder']
    
    if ignore_patterns is None:
        ignore_patterns = ['.pyc', '.pyo', '.so', '.o', '.a', '.dylib', '.exe', '.dll', '.obj', 
                          '.class', '.pdb', '.zip', '.tar', '.gz', '.jar', '.war', '.ear', '.log',
                          'project_extractor.py', 'project_structure.md', 'setup_project_structure.sh', 
                          'structure.md', 'webdev_description.md', 'prompts.md', 'instructions.md', 'theme_implementation_guide.md',
                          'mnist_app.log', 'path_test.py']
    
    # Clean up patterns - remove any trailing/leading whitespace
    ignore_patterns = [p.strip() if isinstance(p, str) else p for p in ignore_patterns]
    ignore_dirs = [d.strip() if isinstance(d, str) else d for d in ignore_dirs]
    
    root_dir = os.path.abspath(root_dir)
    root_dir_name = os.path.basename(root_dir)
    
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
        
        # First pass: generate directory structure
        for root, dirs, files in os.walk(root_dir):
            # Skip directories that match ignore patterns
            dirs[:] = [d for d in dirs if not should_skip_dir(os.path.join(root, d), ignore_dirs)]
            
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            rel_path = os.path.relpath(root, root_dir) if root != root_dir else ''
            dir_name = os.path.basename(root) if rel_path else root_dir_name
            
            out_file.write(f"{indent}{dir_name}/\n")
            
            # Filter files based on ignore patterns
            for f in sorted(files):
                file_path = os.path.join(root, f)
                if should_skip_file(file_path, ignore_patterns):
                    continue
                
                file_size_kb = os.path.getsize(file_path) / 1024
                if file_size_kb > max_file_size_kb:
                    file_info = f" [Large file: {file_size_kb:.1f} KB]"
                else:
                    file_info = ""
                
                out_file.write(f"{indent}    {f}{file_info}\n")
        
        out_file.write("```\n\n")
        
        # Write file contents
        out_file.write("## File Contents\n\n")
        
        # Second pass: write file contents
        for root, dirs, files in os.walk(root_dir):
            # Skip directories that match ignore patterns
            dirs[:] = [d for d in dirs if not should_skip_dir(os.path.join(root, d), ignore_dirs)]
            
            for f in sorted(files):
                file_path = os.path.join(root, f)
                if should_skip_file(file_path, ignore_patterns):
                    continue
                
                rel_path = os.path.relpath(file_path, root_dir)
                file_size_kb = os.path.getsize(file_path) / 1024
                
                out_file.write(f"### {rel_path}\n\n")
                
                if file_size_kb > max_file_size_kb:
                    out_file.write(f"*Large file: {file_size_kb:.1f} KB - content not included*\n\n")
                    continue
                
                if is_binary_file(file_path):
                    out_file.write("*Binary file - content not included*\n\n")
                    continue
                
                language = get_language_for_markdown(file_path)
                out_file.write(f"```{language}\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Special case for .html , .css, .js, .md
                        # files - show only the first few lines
                        if get_file_extension(file_path) in ['html', 'css', 'js', 'md']:
                            lines = []
                            for i, line in enumerate(f):
                                if i < 6:
                                    lines.append(line)
                                else:
                                    break
                            content = ''.join(lines)
                            out_file.write(content)
                            if not content.endswith('\n'):
                                out_file.write('\n')
                            out_file.write("... (remaining content omitted)\n")
                        else:
                            # Normal case - show all content
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
    parser.add_argument('--max-size', '-s', type=int, default=500, help='Max file size in KB to include content (default: 500)')
    parser.add_argument('--ignore-dirs', nargs='*', help='Additional directories to ignore')
    parser.add_argument('--ignore-patterns', nargs='*', help='Additional file patterns to ignore')
    
    args = parser.parse_args()
    
    ignore_dirs = ['venv', 'env', '__pycache__', 'node_modules', '.git', '.idea', '.vscode', 'static_folder']
    if args.ignore_dirs:
        ignore_dirs.extend(args.ignore_dirs)
    
    ignore_patterns = ['.pyc', '.pyo', '.so', '.o', '.a', '.dylib', '.exe', '.dll', '.obj', 
                      '.class', '.pdb', '.zip', '.tar', '.gz', '.jar', '.war', '.ear', '.log',
                      'project_extractor.py', 'project_structure.md', 'setup_project_structure.sh', 
                      'structure.md', 'webdev_description.md', 'prompts.md']
    if args.ignore_patterns:
        ignore_patterns.extend(args.ignore_patterns)
    
    print(f"Extracting project structure from {args.dir}")
    print(f"Writing to {args.output}")
    
    extract_project(
        args.dir, 
        args.output, 
        ignore_dirs=ignore_dirs,
        ignore_patterns=ignore_patterns,
        max_file_size_kb=args.max_size
    )
    
    print(f"Done! Structure and contents extracted to {args.output}")


if __name__ == "__main__":
    main()