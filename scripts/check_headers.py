# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: scripts/check_headers.py
# Description: Checks Python files for standard project headers and
#              optionally attempts to fix the copyright line (Line 2).
# Created: 2025-03-28
# Updated: 2025-03-28

import os
import re
import argparse
import logging
import sys
import datetime # For generating current year in copyright fix

# --- Configuration ---
# Determine project root assuming script is in project_root/scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_LINE_1 = "# MNIST Digit Classifier"

# Regex for Line 2: Allows year update, flexible name/org
# Matches: # Copyright (c) YYYY Name Optional(Org) - ADJUST NAME/ORG HERE!
COPYRIGHT_PATTERN = re.compile(
    r"^#\s+Copyright\s+\(c\)\s+\d{4}\s+YuriODev\s*\(YuriiOks\)\s*$"
    # --- ^^^ MAKE SURE THIS NAME/ORG MATCHES YOUR DESIRED FORMAT ^^^ ---
)
# Generate the correct copyright line format for fixing/comparing
# (Using current year automatically)
CORRECT_COPYRIGHT_LINE = (
    f"# Copyright (c) {datetime.date.today().year} YuriODev (YuriiOks)"
)

# Directories to exclude from checking
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'venv', 'env', '.venv', 'ENV',
    'outputs', 'data', 'build', 'dist', 'node_modules',
    '.vscode', '.idea',
}
# Specific files to exclude (e.g., if managed differently)
EXCLUDE_FILES = {
    'docs/conf.py',
    # Add other specific files if needed, relative to project root
}
# -------------------

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
# -------------------


def check_and_fix_file_header(filepath: str,
                              auto_fix_line2: bool = False) -> str | None:
    """
    Checks headers and optionally fixes the second (Copyright) line.

    Reads the file, checks line 1 and line 2. If auto_fix_line2 is True
    and line 1 is correct but line 2 is incorrect, it replaces line 2
    in memory and rewrites the file.

    Args:
        filepath: Absolute path to the Python file.
        auto_fix_line2: If True, attempts to replace incorrect line 2.

    Returns:
        A string describing the issue or fix status ("FIXED Line 2"),
        or None if the header is correct and no fix was needed.
    """
    lines = []
    original_content = ""
    try:
        # Read all lines first to avoid modifying while iterating
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
            # Splitlines(True) keeps line endings, easier for rewriting
            lines = original_content.splitlines(True)

    except FileNotFoundError:
        return "File Not Found (Skipped)"
    except Exception as e:
        # Catch potential decoding errors as well
        return f"Error Reading File: {e}"

    if not lines:
        return "Missing Line 1 (Empty File?)"

    # Use strip() for comparison, but keep original spacing for rewrite
    line1_stripped = lines[0].strip()
    error_message = None
    needs_rewrite = False

    # --- Check Line 1 ---
    if line1_stripped != EXPECTED_LINE_1:
        error_message = f"Incorrect Line 1: Found '{line1_stripped[:40]}...'"
        # NOTE: We are NOT auto-fixing Line 1 here, only reporting.
    else:
        # --- Check Line 2 (only if Line 1 is correct) ---
        if len(lines) < 2:
            error_message = "Missing Line 2"
            # If auto-fixing, we could potentially insert line 2
            # but that's more complex; let's just report for now.
        else:
            line2_stripped = lines[1].strip()
            # Check if the format matches (ignoring year for flexibility)
            if not COPYRIGHT_PATTERN.match(line2_stripped):
                error_message = (f"Incorrect Line 2 Format/Content: "
                                 f"Found '{line2_stripped[:60]}...'")
                if auto_fix_line2:
                    logger.warning(f"Attempting to fix Line 2 in "
                                 f"{os.path.basename(filepath)}")
                    # Replace line 2 in memory (keeping original line ending)
                    original_ending = ""
                    if lines[1].endswith("\r\n"): original_ending = "\r\n"
                    elif lines[1].endswith("\n"): original_ending = "\n"
                    lines[1] = CORRECT_COPYRIGHT_LINE + original_ending
                    needs_rewrite = True
                    error_message = "FIXED Line 2" # Indicate fix occurred
            # Optional: Update year even if pattern matches
            # elif str(datetime.date.today().year) not in line2_stripped \
            #      and auto_fix_line2:
            #     logger.warning(f"Updating year in Line 2 in "
            #                    f"{os.path.basename(filepath)}")
            #     original_ending = ... # Get original ending
            #     lines[1] = CORRECT_COPYRIGHT_LINE + original_ending
            #     needs_rewrite = True
            #     error_message = "FIXED Line 2 (Year Updated)"

    # --- Write changes back if needed ---
    if needs_rewrite and auto_fix_line2:
        try:
            new_content = "".join(lines) # Join lines preserving endings
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            # error_message is already set to "FIXED Line 2..."
        except Exception as e_write:
            error_message = f"Error Writing File during fix: {e_write}"

    return error_message


def find_python_files(root_dir: str) -> list[str]:
    """Recursively finds .py files, respecting exclusions."""
    python_files = []
    root_dir_abs = os.path.abspath(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_dir_abs, topdown=True):
        # Filter excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            relative_filepath = os.path.relpath(filepath, root_dir_abs)
            # Check file exclusions and if it's a python file
            if filename.endswith(".py") and \
               relative_filepath not in EXCLUDE_FILES:
                python_files.append(filepath)

    return python_files


def main(check_root: str, fix_l2: bool):
    """Finds Python files and checks/fixes their headers."""
    logger.info(f"🔍 Checking Python file headers in: {check_root}")
    logger.info(f"Ignoring directories: {', '.join(sorted(EXCLUDE_DIRS))}")
    if EXCLUDE_FILES:
        logger.info(f"Ignoring specific files: {', '.join(sorted(EXCLUDE_FILES))}")
    logger.info(f"Auto-fixing Line 2: {'Enabled' if fix_l2 else 'Disabled'}")
    logger.info("-" * 60)

    py_files = find_python_files(check_root)
    non_compliant_files = {}
    fixed_files = {}
    checked_count = 0

    if not py_files:
        logger.warning("⚠️ No Python files found to check.")
        return

    for filepath in py_files:
        relative_path = os.path.relpath(filepath, check_root)
        checked_count += 1
        result = check_and_fix_file_header(filepath, auto_fix_line2=fix_l2)

        if result and "FIXED" in result:
            fixed_files[relative_path] = result
            logger.info(f"🔧 {relative_path} - {result}")
        elif result:
            non_compliant_files[relative_path] = result
            logger.warning(f"❌ {relative_path} - {result}")

    logger.info("-" * 60)
    if fixed_files:
         logger.info(f"✅ Automatically fixed Line 2 in "
                     f"{len(fixed_files)} files.")
    if non_compliant_files:
        logger.warning(f"🚩 Found {len(non_compliant_files)} remaining non-compliant "
                     f"files out of {checked_count} checked:")
        # Optionally print list again
        for path, issue in non_compliant_files.items():
            print(f"  - {path}: {issue}")
    else:
        logger.info(f"✅ All {checked_count} Python files have correct headers "
                     f"(after potential fixes)!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check Python file headers in a project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dir',
        type=str,
        default=PROJECT_ROOT,
        help='Root directory to scan for Python files.'
    )
    parser.add_argument(
        '--fix-line2',
        action='store_true',
        help='Attempt to automatically fix incorrect Copyright lines (Line 2).'
             ' USE WITH CAUTION - Backup first!'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        logger.critical(f"🔥 Error: Directory not found: {args.dir}")
        sys.exit(1)

    # Warn if auto-fixing is enabled
    if args.fix_line2:
        confirm = input("⚠️ WARNING: Auto-fixing Line 2 is enabled. "
                        "This will modify files. Ensure you have backups "
                        "or use version control.\nProceed? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Aborting auto-fix.")
            sys.exit(0)

    main(args.dir, args.fix_line2)