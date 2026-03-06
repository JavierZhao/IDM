#!/usr/bin/env python3
"""
Script to compare assessments between Excel file and JSONL file.

This script reads assessment data from both an Excel file and a JSONL file,
compares them according to specified matching rules, and outputs a new Excel file
with the comparison results.

Usage:
    python compare_assessments.py <excel_file> <jsonl_file> [output_file]

Arguments:
    excel_file: Path to the input Excel file
    jsonl_file: Path to the input JSONL file  
    output_file: Optional output Excel file path (default: comparison_output.xlsx)
Example:
python compare_assessments.py judge_file\judge-train.xlsx judge_file_2\judge-train.jsonl judge_file_2\comparison_result_train.xlsx
"""

import json
import sys
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill
from typing import List, Tuple, Any, Optional, Dict
import argparse


def read_jsonl_judge_results(jsonl_file: str) -> List[List[bool]]:
    """
    Read JSONL file and extract judge_result boolean arrays.
    
    Args:
        jsonl_file: Path to the JSONL file
        
    Returns:
        List of boolean arrays from judge_result field
    """
    judge_results = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'judge_result' in data:
                    # Extract the first element which should be a list of booleans
                    judge_result = data['judge_result']
                    if isinstance(judge_result, list) and len(judge_result) > 0:
                        first_element = judge_result[0]
                        if isinstance(first_element, list):
                            judge_results.append(first_element)
                        else:
                            print(f"Warning: Line {line_num} - judge_result[0] is not a list: {first_element}")
                            judge_results.append([])
                    else:
                        print(f"Warning: Line {line_num} - judge_result is not a list or is empty: {judge_result}")
                        judge_results.append([])
                else:
                    print(f"Warning: Line {line_num} - no judge_result field found")
                    judge_results.append([])
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                judge_results.append([])
    
    return judge_results


def read_excel_assessments(excel_file: str) -> Tuple[List[List[str]], int]:
    """
    Read assessment data from Excel file starting from row 1, column 1.
    
    Args:
        excel_file: Path to the Excel file
        
    Returns:
        Tuple of (list of assessment rows, number of rows read)
    """
    # Load the Excel file
    workbook = openpyxl.load_workbook(excel_file)
    sheet = workbook.active
    num_rows = sheet.max_row
    
    assessments = []
    row_count = 0

    # print(num_rows)
    # print('-----------------------------')
    
    # Read from row 1 to the end of the sheet
    for row in sheet.iter_rows(min_row=1, min_col=1, max_col=3):
        # Check if all cells in the row are empty
        #if all(cell.value is None or str(cell.value).strip() == '' for cell in row):
           # break
            
        # Extract the three column values
        row_values = []
        for cell in row:
            if cell.value is not None:
                row_values.append(str(cell.value).strip())
            else:
                row_values.append('')
        
        assessments.append(row_values)
        row_count += 1

    
    workbook.close()
    return assessments, row_count


def cell_to_boolean(cell_value: str) -> Optional[bool]:
    """
    Convert cell value to boolean based on first character.
    
    Args:
        cell_value: String value from Excel cell
        
    Returns:
        Boolean value or None if conversion not possible
    """
    if not cell_value:
        return None
    
    first_char = cell_value[0]
    if first_char == '1':
        return True
    elif first_char == '0':
        return False
    else:
        return None


def compare_assessments(excel_assessments: List[List[str]], jsonl_judge_results: List[List[bool]]) -> Tuple[List[dict], Dict]:
    """
    Compare Excel assessments with JSONL judge results and calculate confusion matrices.
    
    Args:
        excel_assessments: List of assessment rows from Excel (ground truth)
        jsonl_judge_results: List of boolean arrays from JSONL (predictions)
        
    Returns:
        Tuple of (comparison results, confusion matrices for 3 variables)
    """
    comparison_results = []
    
    # Initialize confusion matrices for 3 variables + overall assessment
    confusion_matrices = {}
    for var_idx in range(3):
        confusion_matrices[f'variable_{var_idx + 1}'] = {
            'tp': 0,  # True Positive: ground_truth=True, prediction=True
            'fp': 0,  # False Positive: ground_truth=False, prediction=True
            'tn': 0,  # True Negative: ground_truth=False, prediction=False
            'fn': 0   # False Negative: ground_truth=True, prediction=False
        }
    
    # Add overall assessment confusion matrix
    confusion_matrices['overall_assessment'] = {
        'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
    }
    
    min_length = min(len(excel_assessments), len(jsonl_judge_results))
    
    for i in range(min_length):
        excel_row = excel_assessments[i]
        jsonl_bools = jsonl_judge_results[i]
        
        # Check if Excel row is blank (all cells empty or None)
        excel_is_blank = all(cell_value == '' or cell_value is None for cell_value in excel_row)
        
        # Check if JSONL row is blank (empty list or all None values)
        jsonl_is_blank = len(jsonl_bools) == 0 or all(val is None for val in jsonl_bools)
        
        # Convert Excel cells to booleans
        excel_bools = []
        for cell_value in excel_row:
            bool_val = cell_to_boolean(cell_value)
            excel_bools.append(bool_val)
        
        # Compare the assessments
        matches = []
        all_match = True
        is_not_available = False
        
        # Check if either row is blank - mark as not available
        if excel_is_blank or jsonl_is_blank:
            is_not_available = True
            all_match = False
            matches = [None, None, None]  # Mark as not available
            overall_excel_assessment = None
            overall_jsonl_assessment = None
            overall_match = None
        else:
            # Check if first element in both Excel and JSONL are True
            first_excel_bool = excel_bools[0] if len(excel_bools) > 0 else None
            first_jsonl_bool = jsonl_bools[0] if len(jsonl_bools) > 0 else None
            
            # Calculate overall assessment for both Excel and JSONL
            # Rule: True if first element is True OR (first is False AND third is True)
            third_excel_bool = excel_bools[2] if len(excel_bools) > 2 else None
            third_jsonl_bool = jsonl_bools[2] if len(jsonl_bools) > 2 else None
            
            # Excel overall assessment
            if first_excel_bool is True:
                overall_excel_assessment = True
            elif first_excel_bool is False and third_excel_bool is True:
                overall_excel_assessment = True
            else:
                overall_excel_assessment = False
            
            # JSONL overall assessment
            if first_jsonl_bool is True:
                overall_jsonl_assessment = True
            elif first_jsonl_bool is False and third_jsonl_bool is True:
                overall_jsonl_assessment = True
            else:
                overall_jsonl_assessment = False
            
            # Overall assessment match
            overall_match = (overall_excel_assessment == overall_jsonl_assessment)
            
            # Update confusion matrices (only if data is available)
            if (first_excel_bool is True and first_jsonl_bool is True):
                # If both first elements are True, it's a match regardless of other elements
                all_match = True
                matches = [True, None, None]  # Only first variable matches, others are not compared
                
                # Only count the first variable in confusion matrix
                if first_excel_bool is not None and first_jsonl_bool is not None:
                    if first_excel_bool and first_jsonl_bool:
                        confusion_matrices['variable_1']['tp'] += 1
                    elif not first_excel_bool and first_jsonl_bool:
                        confusion_matrices['variable_1']['fp'] += 1
                    elif not first_excel_bool and not first_jsonl_bool:
                        confusion_matrices['variable_1']['tn'] += 1
                    elif first_excel_bool and not first_jsonl_bool:
                        confusion_matrices['variable_1']['fn'] += 1
            else:
                # Original comparison logic for all other cases
                comparison_length = min(3, len(excel_bools), len(jsonl_bools))
                
                for j in range(comparison_length):
                    excel_bool = excel_bools[j]
                    jsonl_bool = jsonl_bools[j] if j < len(jsonl_bools) else None
                    
                    # Check for match according to the rules
                    if excel_bool is not None and jsonl_bool is not None:
                        match = (excel_bool == jsonl_bool)
                        
                        # Update confusion matrix for this variable
                        var_key = f'variable_{j + 1}'
                        if excel_bool and jsonl_bool:
                            confusion_matrices[var_key]['tp'] += 1
                        elif not excel_bool and jsonl_bool:
                            confusion_matrices[var_key]['fp'] += 1
                        elif not excel_bool and not jsonl_bool:
                            confusion_matrices[var_key]['tn'] += 1
                        elif excel_bool and not jsonl_bool:
                            confusion_matrices[var_key]['fn'] += 1
                    else:
                        match = False
                        all_match = False
                    
                    matches.append(match)
                
                # If lengths don't match, it's not a complete match
                if len(excel_bools) != len(jsonl_bools) or comparison_length < 3:
                    all_match = False
                else:
                    all_match = all(matches)
            
            # Update overall assessment confusion matrix (only if data is available)
            if overall_excel_assessment is not None and overall_jsonl_assessment is not None:
                if overall_excel_assessment and overall_jsonl_assessment:
                    confusion_matrices['overall_assessment']['tp'] += 1
                elif not overall_excel_assessment and overall_jsonl_assessment:
                    confusion_matrices['overall_assessment']['fp'] += 1
                elif not overall_excel_assessment and not overall_jsonl_assessment:
                    confusion_matrices['overall_assessment']['tn'] += 1
                elif overall_excel_assessment and not overall_jsonl_assessment:
                    confusion_matrices['overall_assessment']['fn'] += 1
        
        comparison_results.append({
            'row_index': i,
            'excel_original': excel_row,
            'excel_booleans': excel_bools,
            'jsonl_booleans': jsonl_bools,
            'individual_matches': matches,
            'all_match': all_match,
            'is_not_available': is_not_available,
            'overall_excel_assessment': overall_excel_assessment,
            'overall_jsonl_assessment': overall_jsonl_assessment,
            'overall_match': overall_match
        })
    
    # Handle remaining rows if one list is longer
    max_length = max(len(excel_assessments), len(jsonl_judge_results))
    for i in range(min_length, max_length):
        if i < len(excel_assessments):
            excel_row = excel_assessments[i]
            excel_bools = [cell_to_boolean(cell) for cell in excel_row]
            jsonl_bools = []
        else:
            excel_row = ['', '', '']
            excel_bools = [None, None, None]
            jsonl_bools = jsonl_judge_results[i] if i < len(jsonl_judge_results) else []
        
        comparison_results.append({
            'row_index': i,
            'excel_original': excel_row,
            'excel_booleans': excel_bools,
            'jsonl_booleans': jsonl_bools,
            'individual_matches': [None, None, None],
            'all_match': False,
            'is_not_available': True,  # Remaining rows are considered not available
            'overall_excel_assessment': None,
            'overall_jsonl_assessment': None,
            'overall_match': None
        })
    
    return comparison_results, confusion_matrices


def create_output_excel(comparison_results: List[dict], confusion_matrices: Dict, output_file: str):
    """
    Create output Excel file with comparison results and confusion matrices.
    
    Args:
        comparison_results: Results from comparison
        confusion_matrices: Confusion matrices for the 3 Boolean variables
        output_file: Path for output Excel file
    """
    # Create a new workbook
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Assessment Comparison"
    
    # Define colors for highlighting
    match_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # Light green
    no_match_fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")  # Light pink
    not_available_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")  # Light gray
    
    # Write headers
    headers = [
        "Row Index", 
        "Excel Col 1", "Excel Col 2", "Excel Col 3",
        "Excel Bool 1", "Excel Bool 2", "Excel Bool 3",
        "JSONL Bool 1", "JSONL Bool 2", "JSONL Bool 3",
        "Var 1 Match", "Var 2 Match", "Var 3 Match",
        "Total Match Status",
        "Excel Overall", "JSONL Overall", "Overall Assessment Match"
    ]
    
    for col, header in enumerate(headers, 1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = openpyxl.styles.Font(bold=True)
    
    # Write data
    for i, result in enumerate(comparison_results, 2):
        row_data = [
            result['row_index'] + 1,  # Adjust for 1-based row numbering
        ]
        
        # Add original Excel values
        excel_orig = result['excel_original']
        for j in range(3):
            if j < len(excel_orig):
                row_data.append(excel_orig[j])
            else:
                row_data.append('')
        
        # Add Excel boolean conversions
        excel_bools = result['excel_booleans']
        for j in range(3):
            if j < len(excel_bools) and excel_bools[j] is not None:
                row_data.append(str(excel_bools[j]).lower())
            else:
                row_data.append('')
        
        # Add JSONL booleans
        jsonl_bools = result['jsonl_booleans']
        for j in range(3):
            if j < len(jsonl_bools):
                row_data.append(str(jsonl_bools[j]).lower())
            else:
                row_data.append('')
        
        # Add individual variable match status
        individual_matches = result['individual_matches']
        for j in range(3):
            if j < len(individual_matches):
                if result['is_not_available']:
                    row_data.append("NOT AVAILABLE")
                else:
                    match_val = individual_matches[j]
                    if match_val is True:
                        row_data.append("MATCH")
                    elif match_val is False:
                        row_data.append("NOT MATCH")
                    else:  # None case - not compared
                        row_data.append("")
            else:
                row_data.append("")
        
        # Add overall match status
        if result['is_not_available']:
            row_data.append("NOT AVAILABLE")
        elif result['all_match']:
            row_data.append("MATCH")
        else:
            row_data.append("NOT MATCH")
        
        # Add overall assessment columns
        if result['is_not_available']:
            row_data.append("NOT AVAILABLE")  # Excel Overall
            row_data.append("NOT AVAILABLE")  # JSONL Overall
            row_data.append("NOT AVAILABLE")  # Overall Assessment Match
        else:
            # Excel Overall Assessment
            excel_overall = result['overall_excel_assessment']
            row_data.append(str(excel_overall).lower() if excel_overall is not None else "")
            
            # JSONL Overall Assessment
            jsonl_overall = result['overall_jsonl_assessment']
            row_data.append(str(jsonl_overall).lower() if jsonl_overall is not None else "")
            
            # Overall Assessment Match
            overall_match = result['overall_match']
            if overall_match is True:
                row_data.append("MATCH")
            elif overall_match is False:
                row_data.append("NOT MATCH")
            else:
                row_data.append("")
        
        # Write the row data
        for col, value in enumerate(row_data, 1):
            cell = sheet.cell(row=i, column=col, value=value)
            
            # Apply color coding for match status columns (individual matches + overall match + overall assessment match)
            if col >= len(row_data) - 6:  # Last 7 columns are match status columns
                if value == "MATCH":
                    cell.fill = match_fill
                elif value == "NOT MATCH":
                    cell.fill = no_match_fill
                elif value == "NOT AVAILABLE":
                    cell.fill = not_available_fill
                # Empty cells (not compared) get no color
    
    # Auto-adjust column widths
    for column in sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Add confusion matrices as a new sheet
    confusion_sheet = workbook.create_sheet("Confusion Matrices")
    
    # Write confusion matrices
    current_row = 1
    for var_name, matrix in confusion_matrices.items():
        # Variable header
        display_name = var_name.replace('_', ' ').title()
        if var_name == 'overall_assessment':
            display_name = "Overall Assessment"
        confusion_sheet.cell(row=current_row, column=1, value=display_name)
        confusion_sheet.cell(row=current_row, column=1).font = openpyxl.styles.Font(bold=True)
        current_row += 1
        
        # Matrix headers
        confusion_sheet.cell(row=current_row, column=2, value="Predicted True")
        confusion_sheet.cell(row=current_row, column=3, value="Predicted False")
        confusion_sheet.cell(row=current_row, column=4, value="Total")
        current_row += 1
        
        # True row
        confusion_sheet.cell(row=current_row, column=1, value="Actual True")
        confusion_sheet.cell(row=current_row, column=2, value=matrix['tp'])  # True Positive
        confusion_sheet.cell(row=current_row, column=3, value=matrix['fn'])  # False Negative
        confusion_sheet.cell(row=current_row, column=4, value=matrix['tp'] + matrix['fn'])  # Total True
        current_row += 1
        
        # False row
        confusion_sheet.cell(row=current_row, column=1, value="Actual False")
        confusion_sheet.cell(row=current_row, column=2, value=matrix['fp'])  # False Positive
        confusion_sheet.cell(row=current_row, column=3, value=matrix['tn'])  # True Negative
        confusion_sheet.cell(row=current_row, column=4, value=matrix['fp'] + matrix['tn'])  # Total False
        current_row += 1
        
        # Total row
        confusion_sheet.cell(row=current_row, column=1, value="Total")
        confusion_sheet.cell(row=current_row, column=2, value=matrix['tp'] + matrix['fp'])  # Total Predicted True
        confusion_sheet.cell(row=current_row, column=3, value=matrix['fn'] + matrix['tn'])  # Total Predicted False
        confusion_sheet.cell(row=current_row, column=4, value=matrix['tp'] + matrix['fp'] + matrix['fn'] + matrix['tn'])  # Grand Total
        current_row += 1
        
        # Metrics
        total = matrix['tp'] + matrix['fp'] + matrix['fn'] + matrix['tn']
        if total > 0:
            accuracy = (matrix['tp'] + matrix['tn']) / total
            precision = matrix['tp'] / (matrix['tp'] + matrix['fp']) if (matrix['tp'] + matrix['fp']) > 0 else 0
            recall = matrix['tp'] / (matrix['tp'] + matrix['fn']) if (matrix['tp'] + matrix['fn']) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            confusion_sheet.cell(row=current_row, column=1, value="Accuracy")
            confusion_sheet.cell(row=current_row, column=2, value=f"{accuracy:.4f}")
            current_row += 1
            
            confusion_sheet.cell(row=current_row, column=1, value="Precision")
            confusion_sheet.cell(row=current_row, column=2, value=f"{precision:.4f}")
            current_row += 1
            
            confusion_sheet.cell(row=current_row, column=1, value="Recall")
            confusion_sheet.cell(row=current_row, column=2, value=f"{recall:.4f}")
            current_row += 1
            
            confusion_sheet.cell(row=current_row, column=1, value="F1-Score")
            confusion_sheet.cell(row=current_row, column=2, value=f"{f1_score:.4f}")
            current_row += 1
        
        current_row += 2  # Add space between variables
    
    # Auto-adjust column widths for confusion matrix sheet
    for column in confusion_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 30)
        confusion_sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Save the workbook
    workbook.save(output_file)
    workbook.close()


def main():
    parser = argparse.ArgumentParser(description='Compare assessments between Excel and JSONL files')
    parser.add_argument('excel_file', help='Path to the input Excel file')
    parser.add_argument('jsonl_file', help='Path to the input JSONL file')
    parser.add_argument('output_file', nargs='?', default='comparison_output.xlsx', 
                       help='Output Excel file path (default: comparison_output.xlsx)')
    
    args = parser.parse_args()
    
    try:
        print(f"Reading JSONL file: {args.jsonl_file}")
        jsonl_judge_results = read_jsonl_judge_results(args.jsonl_file)
        print(f"Found {len(jsonl_judge_results)} judge results")
        
        print(f"Reading Excel file: {args.excel_file}")
        print(f"Starting from row 1, column 1")
        excel_assessments, row_count = read_excel_assessments(args.excel_file)
        print(f"Found {row_count} assessment rows")
        
        print("Comparing assessments...")
        comparison_results, confusion_matrices = compare_assessments(excel_assessments, jsonl_judge_results)
        
        print(f"Creating output file: {args.output_file}")
        create_output_excel(comparison_results, confusion_matrices, args.output_file)
        
        # Print summary
        total_rows = len(comparison_results)
        matching_rows = sum(1 for result in comparison_results if result['all_match'])
        not_available_rows = sum(1 for result in comparison_results if result['is_not_available'])
        available_rows = total_rows - not_available_rows
        
        print(f"\nComparison Summary:")
        print(f"Total rows: {total_rows}")
        print(f"Available rows: {available_rows}")
        print(f"Not available rows: {not_available_rows}")
        print(f"Matching rows (among available): {matching_rows}")
        print(f"Non-matching rows (among available): {available_rows - matching_rows}")
        
        print(f"\nConfusion Matrix Summary:")
        for var_name, matrix in confusion_matrices.items():
            total = matrix['tp'] + matrix['fp'] + matrix['fn'] + matrix['tn']
            if total > 0:
                accuracy = (matrix['tp'] + matrix['tn']) / total
                precision = matrix['tp'] / (matrix['tp'] + matrix['fp']) if (matrix['tp'] + matrix['fp']) > 0 else 0
                recall = matrix['tp'] / (matrix['tp'] + matrix['fn']) if (matrix['tp'] + matrix['fn']) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                display_name = var_name.replace('_', ' ').title()
                if var_name == 'overall_assessment':
                    display_name = "Overall Assessment"
                print(f"{display_name}:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1_score:.4f}")
        
        # Print 0-based indices where Total Match Status is "NOT MATCH"
        not_match_indices = []
        for result in comparison_results:
            if not result['is_not_available'] and not result['all_match']:
                not_match_indices.append(result['row_index'])
        
        print(f"\n0-based indices where Total Match Status is 'NOT MATCH':")
        if not_match_indices:
            print(f"Indices: {not_match_indices}")
            print(f"Count: {len(not_match_indices)}")
        else:
            print("No rows with 'NOT MATCH' status found.")
        
        print(f"\nOutput saved to: {args.output_file}")
        print(f"Confusion matrices saved in 'Confusion Matrices' sheet")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()