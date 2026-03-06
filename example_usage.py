#!/usr/bin/env python3
"""
Example usage of the assessment comparison tool.

This script demonstrates how to use the compare_assessments.py script
with your existing JSONL file.
"""

import os
import subprocess
import sys
from pathlib import Path


def create_sample_excel():
    """Create a sample Excel file for demonstration purposes."""
    try:
        import openpyxl
        
        # Create a sample Excel file
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Sample Assessments"
        
        # Add some sample data
        sample_data = [
            ["Assessment", "Column1", "Column2", "Column3"],  # Header row
            ["Row1", "1", "0", "1"],  # Sample assessment data
            ["Row2", "0", "1", "0"],
            ["Row3", "1", "1", "0"],
            ["Row4", "0", "0", "1"],
        ]
        
        for row_idx, row_data in enumerate(sample_data, 1):
            for col_idx, value in enumerate(row_data, 1):
                sheet.cell(row=row_idx, column=col_idx, value=value)
        
        workbook.save("sample_assessments.xlsx")
        workbook.close()
        print("Created sample_assessments.xlsx")
        return True
        
    except ImportError:
        print("Error: openpyxl not installed. Run 'pip install openpyxl' first.")
        return False


def main():
    print("Assessment Comparison Tool - Example Usage")
    print("=" * 50)
    
    # Check if the JSONL file exists
    jsonl_file = "outputs/o4-mini-2025-04-16_batch_1_2_paraphrased_trap.jsonl"
    if not os.path.exists(jsonl_file):
        print(f"Error: JSONL file not found: {jsonl_file}")
        print("Please ensure the file exists in the correct location.")
        return
    
    # Create sample Excel file if it doesn't exist
    excel_file = "sample_assessments.xlsx"
    if not os.path.exists(excel_file):
        print("Creating sample Excel file...")
        if not create_sample_excel():
            return
    
    # Example 1: Basic usage with sample data
    print("\nExample 1: Basic comparison")
    print("-" * 30)
    
    cmd = [
        sys.executable, "compare_assessments.py",
        excel_file,           # Excel file
        jsonl_file,           # JSONL file  
        "2",                  # Start row (skip header)
        "2",                  # Start column (skip first column)
        "example_output.xlsx" # Output file
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Comparison completed successfully!")
            print("Output saved to: example_output.xlsx")
            print("\nSummary output:")
            print(result.stdout)
        else:
            print("✗ Error occurred during comparison:")
            print(result.stderr)
    
    except FileNotFoundError:
        print("Error: compare_assessments.py not found in current directory")
        print("Please ensure the script is in the same directory.")
    
    # Example 2: Different starting position
    print("\nExample 2: Different starting position (row 1, column 1)")
    print("-" * 30)
    
    cmd2 = [
        sys.executable, "compare_assessments.py",
        excel_file,
        jsonl_file,
        "1",                          # Start from row 1
        "1",                          # Start from column 1
        "example_output_alt.xlsx"     # Different output file
    ]
    
    print(f"Running command: {' '.join(cmd2)}")
    
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        
        if result2.returncode == 0:
            print("✓ Alternative comparison completed successfully!")
            print("Output saved to: example_output_alt.xlsx")
        else:
            print("✗ Error occurred during alternative comparison:")
            print(result2.stderr)
    
    except FileNotFoundError:
        print("Error: compare_assessments.py not found")
    
    print("\nExample completed. Check the generated Excel files for results.")


if __name__ == "__main__":
    main()