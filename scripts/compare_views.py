#!/usr/bin/env python3
"""
Command-line tool for comparing text similarity.

This script compares two texts using semantic embeddings and cosine similarity.
It supports reading from files or stdin, and can output results in JSON format.

Usage:
    # Compare two text files
    python scripts/compare_views.py file1.txt file2.txt
    
    # Read from stdin (interactive mode)
    python scripts/compare_views.py --stdin
    
    # Output as JSON
    python scripts/compare_views.py file1.txt file2.txt --json
    
    # Custom threshold
    python scripts/compare_views.py file1.txt file2.txt --threshold 0.9
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from genie.similarity import is_semantic_similar, clear_cache


def read_text_from_file(filepath: str) -> str:
    """
    Read text content from a file.
    
    Args:
        filepath: Path to the text file.
        
    Returns:
        Content of the file as a string.
        
    Raises:
        FileNotFoundError: If file does not exist.
        IOError: If file cannot be read.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return path.read_text(encoding="utf-8")


def read_text_from_stdin() -> Tuple[str, str]:
    """
    Read two texts from stdin interactively.
    
    Returns:
        Tuple of (text1, text2) read from stdin.
    """
    print("Enter first text (end with empty line):", file=sys.stderr)
    lines1 = []
    while True:
        line = input()
        if line == "":
            break
        lines1.append(line)
    
    print("Enter second text (end with empty line):", file=sys.stderr)
    lines2 = []
    while True:
        line = input()
        if line == "":
            break
        lines2.append(line)
    
    return "\n".join(lines1), "\n".join(lines2)


def format_result(result: dict, json_output: bool = False) -> str:
    """
    Format the similarity result for output.
    
    Args:
        result: Dictionary with score, similar, and method keys.
        json_output: If True, output as JSON. Otherwise, human-readable.
        
    Returns:
        Formatted string representation of the result.
    """
    if json_output:
        return json.dumps(result, indent=2)
    
    lines = [
        f"Similarity Score: {result['score']:.4f}",
        f"Similar: {result['similar']}",
        f"Threshold Met: {'Yes' if result['similar'] else 'No'}",
        f"Method Used: {result['method']}"
    ]
    return "\n".join(lines)


def main() -> Optional[int]:
    """
    Main entry point for the CLI tool.
    
    Returns:
        Exit code (0 for similar, 1 for not similar, 2 for error).
    """
    parser = argparse.ArgumentParser(
        description="Compare semantic similarity between two texts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file1.txt file2.txt
  %(prog)s file1.txt file2.txt --json
  %(prog)s file1.txt file2.txt --threshold 0.9
  %(prog)s --stdin
  %(prog)s --clear-cache
        """
    )
    
    parser.add_argument(
        "file1",
        nargs="?",
        help="Path to first text file"
    )
    parser.add_argument(
        "file2",
        nargs="?",
        help="Path to second text file"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read texts interactively from stdin"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold (default: 0.8)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear embedding cache and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle cache clearing
    if args.clear_cache:
        clear_cache()
        print("Cache cleared.", file=sys.stderr)
        return 0
    
    # Validate arguments
    if args.stdin:
        if args.file1 or args.file2:
            print("Error: Cannot use --stdin with file arguments", file=sys.stderr)
            return 2
        text1, text2 = read_text_from_stdin()
    else:
        if not args.file1 or not args.file2:
            parser.print_help()
            return 2
        
        try:
            text1 = read_text_from_file(args.file1)
            text2 = read_text_from_file(args.file2)
        except (FileNotFoundError, IOError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0", file=sys.stderr)
        return 2
    
    # Perform comparison
    if args.verbose:
        print(f"Comparing texts with threshold={args.threshold}...", file=sys.stderr)
    
    result = is_semantic_similar(
        text1, text2,
        threshold=args.threshold,
        use_cache=not args.no_cache
    )
    
    # Output result
    print(format_result(result, json_output=args.json))
    
    # Return appropriate exit code
    if result["method"] == "failed":
        return 2
    return 0 if result["similar"] else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
