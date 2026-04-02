#!/usr/bin/env python3
"""Convert Python script to Jupyter notebook format"""

import json
import sys
import re
from pathlib import Path


def python_to_ipynb(py_file, ipynb_file=None):
    """Convert a Python script to Jupyter notebook format"""

    with open(py_file, 'r') as f:
        content = f.read()

    # Create notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Split content into cells based on markdown comments or code sections
    lines = content.split('\n')
    current_cell = []
    cell_type = "code"
    in_docstring = False

    for i, line in enumerate(lines):
        # Check for markdown headers (### or ## at start of line)
        if line.strip().startswith('###') or line.strip().startswith('##'):
            # Save previous cell
            if current_cell:
                notebook["cells"].append({
                    "cell_type": cell_type,
                    "metadata": {},
                    "source": current_cell
                })
                current_cell = []

            # Start new markdown cell
            cell_type = "markdown"
            # Convert markdown-style comments
            markdown_text = line.strip().lstrip('#').strip()
            current_cell = [markdown_text + '\n']
            cell_type = "markdown"
        elif line.strip().startswith('#') and not line.strip().startswith('#!'):
            # Regular comment - treat as markdown if on its own line
            if current_cell and cell_type == "markdown":
                markdown_text = line.strip().lstrip('#').strip()
                current_cell.append(markdown_text + '\n')
            elif not current_cell:
                # Start new markdown cell
                cell_type = "markdown"
                markdown_text = line.strip().lstrip('#').strip()
                current_cell = [markdown_text + '\n']
            else:
                # Add to current code cell
                current_cell.append(line + '\n')
        else:
            # Code line - switch to code cell if needed
            if cell_type == "markdown" and current_cell:
                notebook["cells"].append({
                    "cell_type": cell_type,
                    "metadata": {},
                    "source": current_cell
                })
                current_cell = []
                cell_type = "code"

            current_cell.append(line + '\n')

    # Add last cell
    if current_cell:
        notebook["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": current_cell
        })

    # Determine output filename
    if ipynb_file is None:
        ipynb_file = Path(py_file).with_suffix('.ipynb')

    # Write notebook
    with open(ipynb_file, 'w') as f:
        json.dump(notebook, f, indent=2)

    return ipynb_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python py_to_ipynb.py <file1.py> [file2.py] ...")
        sys.exit(1)

    for py_file in sys.argv[1:]:
        try:
            ipynb_file = python_to_ipynb(py_file)
            print(f"✓ Converted: {py_file} -> {ipynb_file}")
        except Exception as e:
            print(f"✗ Error converting {py_file}: {e}")
