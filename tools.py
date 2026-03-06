import pandas as pd
import json
import re
import os
from typing import Tuple
from openai import OpenAI
from tqdm import tqdm
# from utils.utils import load_jsonl
# from utils.parser import extract_answer


def j2x(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    # Get columns in the order of their first appearance in the file
    columns = []
    seen = set()
    for item in data:
        for key in item.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)
    print(columns)
    df = pd.DataFrame(data, columns=columns)
    df.to_excel(output_file, index=False)

def x2j(input_file, output_file):
    df = pd.read_excel(input_file)
    
    # Convert all columns to string type to avoid any type issues
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    # Check for columns that contain only 0 and 1 (and possibly NaN/empty) and convert to boolean
    for col in df.columns:
        # Get non-empty, non-NaN values
        non_empty_values = df[col][~df[col].isin(['nan', '', 'None'])].unique()
        # Remove any whitespace and check if only contains '0' and '1' (including float representations)
        clean_values = set()
        for v in non_empty_values:
            v_str = str(v).strip()
            if v_str != '':
                # Handle both integer and float representations
                try:
                    # Convert to float first, then to int to handle cases like '1.0' -> 1
                    num_val = int(float(v_str))
                    clean_values.add(str(num_val))
                except (ValueError, TypeError):
                    # If conversion fails, keep original string
                    clean_values.add(v_str)
        
        if clean_values.issubset({'0', '1'}) and len(clean_values) > 0:
            # Convert 0/1 to False/True (handle both int and float representations)
            def convert_to_bool(x):
                x_str = str(x).strip()
                if x_str in ['nan', '', 'None']:
                    return x
                try:
                    num_val = int(float(x_str))
                    return 'True' if num_val == 1 else 'False' if num_val == 0 else x
                except (ValueError, TypeError):
                    return x
            df[col] = df[col].apply(convert_to_bool)
    
    # For each row, if at least one cell is non-empty, set all empty cells to ''
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        # Check if at least one cell is non-empty (not NaN string and not empty string)
        has_non_empty = any(v not in ['nan', '', 'None'] for v in row_dict.values())
        if has_non_empty:
            for col, val in row_dict.items():
                if val in ['nan', 'None', '']:
                    df.at[idx, col] = ''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')





def id_matching(input_file, reference, paste_split):
    '''
    This function is used to add level, answer and subject to the input xlsx file based on the reference jsonl file.
    For matching, it removes equations from the original problem, discretizes it into words, and checks if all words
    are present in each reference row. If multiple matches are found, new rows are created under the current row.
    
    1. Please check whether there are any unmatched questions or questions with multiple matches (and manually select the right one).
    2. Then delete the duplicate rows you don't want to keep.
    3. Save the new xlsx file, and use x2j function to convert it to jsonl file.
    '''
    def remove_equations(text):
        """Remove mathematical equations from text (content enclosed by $$, \(\) or \[\])"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove $$ equations
        text = re.sub(r'\$\$((?:[^$\\]|\\.)*?)\$\$', '', text, flags=re.DOTALL)
        # Remove \( \) equations
        text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)
        # Remove \[ \] equations
        text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
        # Remove single $ equations
        text = re.sub(r'\$((?:[^$\\]|\\.)*?)\$', '', text, flags=re.DOTALL)
        
        return text
        # Find all matching reference rows using the word matching criterion


    def extract_equations(text):
        """Extract mathematical equations from text (content enclosed by $$, \(\) or \[\])"""
        if pd.isna(text):
            return []
        
        text = str(text)
        equations = []
        
        # Extract $$ equations
        equations.extend(re.findall(r'\$\$([^$]*(?:\\.[^$]*)*)\$\$', text, flags=re.DOTALL))
        # Extract \( \) equations
        equations.extend(re.findall(r'\\\((.*?)\\\)', text, flags=re.DOTALL))
        # Extract \[ \] equations
        equations.extend(re.findall(r'\\\[(.*?)\\\]', text, flags=re.DOTALL))
        # Extract single $ equations (handle escaped dollar signs properly)
        # This regex matches content between $ signs, allowing for escaped characters including \$
        single_dollar_matches = re.findall(r'\$([^$]*(?:\\.[^$]*)*)\$', text, flags=re.DOTALL)
        equations.extend(single_dollar_matches)
        
        # Clean up equations (strip whitespace)
        pattern = r"(\\quad|\\qquad|\\,|\\:|\\;|\\!|\\enspace|\\hspace\{.*?\}|\s+)"   
        equations = [re.sub(pattern, "", eq.strip()) for eq in equations if eq.strip()]
        
        return equations

    def equation_matching_criterion(query_problem, reference_problem):
        """
        Matching criterion function: Check if equations from query problem
        are present in the reference problem.
        
        Args:
            query_problem: The problem text to search for
            reference_problem: The reference problem text to search in
        
        Returns:
            bool: True if at least one equation from query is found in reference
        """
        # Extract equations from query problem
        query_equations = extract_equations(query_problem)
        reference_equations = extract_equations(reference_problem)
        
        if not query_equations:  # No equations to match
            if not reference_equations:
                return True
            return False

        # Check if any query equation is present in reference problem
        for equation in query_equations:
            if equation not in reference_equations:
                return False
        
        return True

    def word_matching_criterion(query_problem, reference_problem):
        """
        Matching criterion function: Check if all words from query (after removing equations)
        are present in the reference problem.
        
        Args:
            query_problem: The problem text to search for
            reference_problem: The reference problem text to search in
        
        Returns:
            bool: True if all words from query are found in reference
        """
        # Remove equations from query problem
        query_clean = remove_equations(query_problem)
        
        # Discretize into individual words by space
        query_words = query_clean.split()
        
        # Remove empty strings and convert to lowercase for case-insensitive matching
        query_words = [word.lower().strip() for word in query_words if word.strip()]
        
        if not query_words:  # No words to match
            return False
        
        # Convert reference problem to lowercase for case-insensitive matching
        reference_problem = remove_equations(reference_problem)
        reference_lower = str(reference_problem).lower() if pd.notna(reference_problem) else ""
        
        # Check if all query words are present in reference problem
        for word in query_words:
            if word not in reference_lower:
                # print(f"Word '{word}' not found in reference problem")
                return False
        
        return True

    def match_ref(orig_prob, ref_rows, criterion):
        matched_refs = []
        for ref_row in ref_rows:
            ref_problem = ref_row.get("problem", "")       
            if criterion(orig_prob, ref_problem):
                matched_refs.append(ref_row)
        return matched_refs
    
        
    # Read the input xlsx file
    df = pd.read_excel(input_file)
    ref_rows = list(load_jsonl(reference))
    all_rows = []
    
    # For each row in df, find matches and create new rows if multiple matches
    for idx, row in df.iterrows():
        orig_prob = row.get("original_problem")  
        if row.get("source") != "MATH" or (pd.notna(row.get("level")) and row.get("level") != ""):
            all_rows.append(row.to_dict())
            continue
        

        matched_refs = match_ref(orig_prob, ref_rows, word_matching_criterion)
        #if len(matched_refs) > 1:
            # Multiple matches found with word matching, try equation matching to narrow down
        matched_refs = match_ref(orig_prob, matched_refs, equation_matching_criterion)

            # if matched_refs2:
            #     matched_refs = matched_refs2

        if len(matched_refs) > 1:
            for i, ref_row in enumerate(matched_refs):
                if i == 0:
                    # First match: update the original row
                    new_row = row.to_dict()
                    
                    # Update fields based on reference
                    new_row["level"] = ref_row.get("level", "")
                    
                    # Add matched problem and other fields from reference
                    # Mark matched_problem in green using Excel cell background color code
                    matched_problem = ref_row.get("problem", "")
                    new_row["matched_problem"] = f'=HYPERLINK("","{matched_problem}")'  # Excel formula for hyperlink (no link, just text)
                    new_row["matched_problem__color"] = "green"  # Custom column to indicate color
                    new_row["answer"] = ref_row.get("answer", "")
                    new_row["split"] = paste_split
                    new_row["unique_id"] = ref_row.get("unique_id", "")
                    
                    all_rows.append(new_row)
                else:
                    # Additional matches: create new rows with only matched data, other columns empty
                    new_row = {}
                    
                    # Get all column names from the original dataframe
                    for col in df.columns:
                        new_row[col] = ""  # Initialize all columns as empty
                    
                    # Fill only the matched data columns
                    matched_problem = ref_row.get("problem", "")
                    new_row["matched_problem"] = f'=HYPERLINK("","{matched_problem}")'
                    new_row["matched_problem__color"] = "green"
                    new_row["answer"] = ref_row.get("answer", "")
                    new_row["split"] = paste_split
                    new_row["unique_id"] = ref_row.get("unique_id", "")
                    new_row["level"] = ref_row.get("level", "")
                    
                    all_rows.append(new_row)
        elif not matched_refs:
            # No matches found, keep original row as is
            new_row = row.to_dict()
            new_row["matched_problem"] = "No matches found"
            new_row["matched_problem__color"] = "red"
            all_rows.append(new_row)
            continue
        else:
            new_row = row.to_dict()
            ref_row = matched_refs[0]
            # Update fields based on reference
            new_row["level"] = ref_row.get("level", "")
            
            # Add matched problem and other fields from reference
            new_row["original_problem"] = ref_row.get("problem", "")
            new_row["matched_problem"] = ""
            new_row["answer"] = ref_row.get("answer", "")
            new_row["split"] = paste_split
            new_row["unique_id"] = ref_row.get("unique_id", "")
            
            all_rows.append(new_row)
        # Process matches
        

        
        # Print information about matches
        if len(matched_refs) > 1:
            print(f"Row {idx}: Found {len(matched_refs)} matches")
        elif len(matched_refs) == 1:
            print(f"Row {idx}: Found 1 match")
        else:
            print(f"Row {idx}: No matches found")
    
    # Create new DataFrame from all rows
    new_df = pd.DataFrame(all_rows)
    
    # Save the result
    parts = input_file.split('.')
    out_file = '.'.join(parts[:-1]) + "_lc" + '.' + parts[-1]
    new_df.to_excel(out_file, index=False)
    
    print(f"Results saved to: {out_file}")
    print(f"Original rows: {len(df)}")
    print(f"Total rows after matching: {len(new_df)}")
    
    return out_file


def remove(folder, index_list):
    """
    For all the .jsonl files in the folder, remove the rows corresponding to the index_list (0-based).
    The modified files will overwrite the originals.
    """
    for filename in os.listdir(folder):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder, filename)
            # Read all lines
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Remove lines at the specified indices
            new_lines = [line for idx, line in enumerate(lines) if idx not in index_list]
            # Overwrite the file with the filtered lines
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

def extract_rows(input_file, output_file, rows_to_extract):
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx in rows_to_extract:
                fout.write(line)
def scan(file_path, api_key=None):
    """
    For each row in the input jsonl file, call OpenAI API to check for typos, errors, or inconsistencies
    in the "paraphrased_original" (and its answer) and "paraphrased_trap" (and its annotation) fields. If any are found, save each point
    of error, typo, or inconsistency in a file named <file_path>_scan_report.txt.
    Each field ("paraphrased_original" and "paraphrased_trap") is assessed separately.

    Where should I put the api_key?
    - Pass the OpenAI API key as the `api_key` argument to this function, or set it via the OPENAI_API_KEY environment variable.
    - The function will set openai.api_key if `api_key` is provided.
    """
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    report_file = file_path + "_scan_report.txt"
    with open(file_path, 'r', encoding='utf-8') as fin:
        for idx, line in tqdm(list(enumerate(fin)), desc="Scanning rows"):
            try:
                row = json.loads(line)
            except Exception as e:
                continue
            def check_and_report(field_name, question_text, answer_text, template_file):
                with open(template_file, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
                if "org" in template_file:
                    prompt = prompt_template.format(question=question_text, answer=answer_text)
                elif "trap" in template_file:
                    prompt = prompt_template.format(question=question_text, annotation=answer_text)
                try:
                    response = client.chat.completions.create(
                                extra_body={},
                                model="deepseek/deepseek-chat-v3-0324",
                                messages=[{"role": "user", "content": prompt}]
                                )
                    content = response.choices[0].message.content.strip()
                except Exception as e:
                    content = f"OpenAI API error: {e}"
                if not content:
                    content = "Empty response"
                    with open(report_file, 'a', encoding='utf-8') as fout:
                        fout.write(f"Row {idx} [{field_name}]:\n{content}\n{'-'*40}\n")
                elif "no issues found" in content.lower():
                    print(f"Row {idx} has no issues found in {field_name}")
                else:
                    with open(report_file, 'a', encoding='utf-8') as fout:
                        fout.write(f"Row {idx} [{field_name}]:\n{content}\n{'-'*40}\n")
            orig = row.get("paraphrased_original", "")
            orig_answer = row.get("answer", "")
            trap = row.get("paraphrased_trap", "")
            annotation = row.get("annotation", "")
            "prompt/proofreading_check.tmpl"
            check_and_report("paraphrased_original", orig, orig_answer, "prompt/proofreading_org.tmpl")
            check_and_report("paraphrased_trap", trap, annotation, "prompt/proofreading_trap.tmpl")
    print(f"Scan complete. Report saved to {report_file}")


def paste_annotation(input_folder, target_file, paste_key, end_key):
    """
    For all the jsonl files ending with 'trap' in the input_folder,
    add a new key ["annotation"] in each file, whose value is the ["annotation"] in the annotation file for each row,
    matching row by row until one file ends (no need to check for equal length).
    """
    # Use load_jsonl to load annotation file lines as JSON objects
    annotation_objs = list(load_jsonl(target_file))

    # List all files in input_folder ending with 'trap' and '.jsonl'
    for fname in os.listdir(input_folder):
        if fname.endswith(end_key+'.jsonl'):
            file_path = os.path.join(input_folder, fname)
            file_objs = list(load_jsonl(file_path))
            n = min(len(file_objs), len(annotation_objs))
            if n == 0:
                print(f"File '{fname}': no rows to process. Skipping.")
                continue
            print(f"File '{fname}': adding new key for {n} rows (out of {len(file_objs)} file rows and {len(annotation_objs)} annotation rows).")
            # Add annotation from annotation_objs to file_objs, row by row, up to n rows
            added_count = 0
            for i in range(n):
                if paste_key not in file_objs[i]:
                    file_objs[i][paste_key] = annotation_objs[i].get(paste_key, None)
                    added_count += 1
            # Overwrite the file with updated objects
            with open(file_path, 'w', encoding='utf-8') as f:
                for obj in file_objs:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            print(f"New key added to '{fname}' for {added_count} rows (skipped {n - added_count} rows where key already existed).")


            
def compare_jsonl_keys(file_path_1, file_path_2, key_1, key_2, output_file=None):

    rows_1 = list(load_jsonl(file_path_1))
    rows_2 = list(load_jsonl(file_path_2))

    compare_len = min(len(rows_1), len(rows_2))

    # Default output path if none provided
    if output_file is None:
        base_dir = os.path.dirname(file_path_1) or '.'
        name_1 = os.path.splitext(os.path.basename(file_path_1))[0]
        name_2 = os.path.splitext(os.path.basename(file_path_2))[0]
        safe_key_1 = str(key_1).replace(os.sep, '_')
        safe_key_2 = str(key_2).replace(os.sep, '_')
        output_file = os.path.join(base_dir, f"{name_1}_vs_{name_2}_{safe_key_1}_{safe_key_2}_diff.txt")

    differences_found = 0
    with open(output_file, 'w', encoding='utf-8') as fout:
        for idx in range(compare_len):
            row_1 = rows_1[idx] if idx < len(rows_1) else {}
            row_2 = rows_2[idx] if idx < len(rows_2) else {}
            for key in [key_1, key_2]:
                val_1 = row_1.get(key, None)
                val_2 = row_2.get(key, None)
                if val_1 != val_2:
                    differences_found += 1
                    # Write a simple, machine-readable line
                    fout.write(json.dumps({
                        "row_index": idx,
                        "key": key,
                        "value_1": val_1,
                        "value_2": val_2
                    }, ensure_ascii=False) + "\n")

    if differences_found == 0:
        # Keep the (empty) file to indicate comparison was performed
        print(f"No differences found for keys '{key_1}' and '{key_2}'. Empty report at: {output_file}")
    else:
        print(f"Differences found: {differences_found}. Report saved to: {output_file}")

def scan_and_clean(root_dir: str) -> None:
    """
    For all the jsonl files in the root_dir, clean the "generated_responses" field by removing the prefix "Output:\n" and the suffix "\n\n".
    """
    def clean_response_text(text: str) -> str:
        cleaned = text
        prefix = "Output:\n"
        suffix = "\n\n"
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
        return cleaned


    def clean_jsonl_file(file_path: str) -> Tuple[int, int, bool]:
        """Clean a JSONL file in-place.

        Returns a tuple of (num_rows, num_elements_cleaned, changed_flag).
        """
        examples = list(load_jsonl(file_path))
        num_rows = len(examples)
        num_unchanged = 0

        for idx, example in enumerate(examples):
            responses = example.get("generated_responses")
            new_responses = []
            row_changed = False
            for item in responses:
                if isinstance(item, str):
                    cleaned = clean_response_text(item)
                    if cleaned == item:
                        print(f"Row {idx} has no changes")
                        num_unchanged += 1
                    new_responses.append(cleaned)
            example["generated_responses"] = new_responses

        with open(file_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"for file {file_path}, num_rows={num_rows}, num_unchanged={num_unchanged} \n")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            lower = filename.lower()
            if "claude" in lower and lower.endswith(".jsonl"):
                file_path = os.path.join(dirpath, filename)
                clean_jsonl_file(file_path)

def find_rows_with_prefix(file_path, key, prefix_string):
    """
    Read a jsonl file and find all row indices where the value of key starts with a specific string.
    
    Args:
        file_path (str): Path to the JSONL file
        key (str): The key to check if the value starts with
        prefix_string (str): The string to check if "generated_response" starts with
    
    Returns:
        list: List of indices where the condition is met
    """
    data = list(load_jsonl(file_path))
    matching_indices = []
    
    for index, row in enumerate(data):
        generated_response = row.get(key, "")[0]
        if generated_response.startswith(prefix_string):
            matching_indices.append(index)
    if len(matching_indices) > 0:
        print(f"Found {len(matching_indices)} rows where {key} starts with '{prefix_string}':")
        print(matching_indices)
def find_file_errors(folder, key, prefix_string):
    """
    Read a jsonl file and find all row indices where the value of key is empty.
    
    Args:
        file_path (str): Path to the JSONL file
        key (str): The key to check if the value is empty
    
    Returns:
        list: List of indices where the condition is met
    """
    if "batch1" in folder:
        desired_length = 140
    elif "batch2" in folder:
        desired_length = 160
    for file_path in os.listdir(folder):
        if file_path.endswith(".jsonl") and ("gpt-5" in file_path.lower() or 'o3' in file_path.lower()):
            file_path = os.path.join(folder, file_path)
            data = list(load_jsonl(file_path))
            if len(data) != desired_length:
                print(f"In file {file_path}, found {len(data)} rows")
                print("--------------------------------")
            if "paraphrased" not in file_path:
                print(f"In file {file_path}, the problem key is wrong")
                print("--------------------------------")
            empty_indices = []
            matching_indices = []
            zero_token_indices = []
            long_token_indices = []
            early_stop_indices = []
            
            for index, row in enumerate(data):
                generated_response = row.get(key, "")[0]
                token_usage = row.get("token_usage", "")
                if generated_response.startswith(prefix_string):
                    matching_indices.append(index)
                elif generated_response == "":
                    empty_indices.append(index)
                elif extract_answer(generated_response) == "":
                    if token_usage and token_usage < 19000:
                        early_stop_indices.append(index)
                    elif not token_usage:
                        early_stop_indices.append(index)
                if  token_usage== 0 or (generated_response.endswith("Output:\n") and token_usage and token_usage < 19000):
                    zero_token_indices.append(index)
                elif token_usage and token_usage > 21000:
                    long_token_indices.append(index)
            if len(empty_indices) > 0:
                print(f"In file {file_path}, found {len(empty_indices)} rows where {key} is empty:")
                print(empty_indices)
            if len(matching_indices) > 0:
                print(f"In file {file_path}, found {len(matching_indices)} rows where {key} starts with '{prefix_string}':")
                print(matching_indices)
            if len(zero_token_indices) > 0:
                print(f"In file {file_path}, found {len(zero_token_indices)} rows where token_usage is 0:")
                print(zero_token_indices)
            # if (len(matching_indices) > 0) + (len(empty_indices) > 0) + (len(zero_token_indices) > 0) > 1:
            #     print("Total list:")
            #     union = list(set(matching_indices) | set(empty_indices) | set(zero_token_indices))
            #     print(union)
            # if (len(matching_indices) > 0) + (len(empty_indices) > 0) + (len(zero_token_indices) > 0) > 0:
            #     print("--------------------------------")
            # if len(long_token_indices) > 0:
            #     print(f"In file {file_path}, found {len(long_token_indices)} rows where token_usage is greater than 21000:")
            #     print(long_token_indices)
            #     print("--------------------------------")
            if len(early_stop_indices) > 0:
                print(f"In file {file_path}, found {len(early_stop_indices)} rows with early stop:")
                print(early_stop_indices)
            print("--------------------------------")
def find_early_stop(folder):
    for file_path in os.listdir(folder):
        if file_path.endswith("trap.jsonl") and "4o" in file_path.lower():
            file_path = os.path.join(folder, file_path)
            data = list(load_jsonl(file_path))
            early_indices = []          
            for index, row in enumerate(data):
                if len(row.get("generated_answers", "")) == 0:
                    print(f"In file {file_path}, row {index} has no generated answers")
                    # print("--------------------------------")
                    continue
                generated_answer = row.get("generated_answers", "")[0]
                token_usage = row.get("token_usage", "")
                if token_usage and generated_answer == "" and token_usage < 19000 and "boxed{}" not in row.get("generated_responses", "")[0][-10:]:
                    early_indices.append(index)
            if len(early_indices) > 0:
                print(f"In file {file_path}, found {len(early_indices)} rows with early stop:")
                print(early_indices)
                print("--------------------------------")
def token_string_ratio(file_path):
    data = list(load_jsonl(file_path))
    ratio_list = []

    for row in data:
        token_usage = row.get("token_usage", 0)
        generated_responses = row.get("generated_responses", [])[0]
        if token_usage and token_usage > 0:
            ratio = len(generated_responses) / token_usage
            ratio_list.append((ratio, token_usage))
    ranked_list = sorted(ratio_list, key=lambda x: x[1], reverse=True)
    print(ranked_list)

def modify_token(folder_path, new_token_usage):
    for file_path in os.listdir(folder_path):
        if file_path.endswith(".jsonl") and "qwen3" in file_path:
            file_path = os.path.join(folder_path, file_path)
            correct_num = 0
            data = list(load_jsonl(file_path))
            for idx, row in enumerate(data):
                try:
                    usage = row.get("token_usage", '')
                    if usage == 0:
                        generated_responses = row.get("generated_responses", [])[0]
                        question = row.get("question", "")
                        if len(generated_responses) > 1.8 * new_token_usage:
                            row["token_usage"] = new_token_usage
                            correct_num += 1
                        else:
                            print(len(generated_responses)/new_token_usage)
                            print(f"row {idx} has zero token usage but short answer")
                            print(usage)
                except Exception as e:
                    print(f"Error in file {file_path}, row {idx}: {e}")
            print(f"In file {file_path}, {correct_num} rows have been modified")
            with open(file_path, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
def paste_annotation(reference, folder):        
    reference_rows = list(load_jsonl(reference))
    for file_path in os.listdir(folder):
        if file_path.endswith("trap.jsonl"):
            file_path = os.path.join(folder, file_path)
            data = list(load_jsonl(file_path))
            tot = min(len(data), len(reference_rows))
            for i in range(tot):
                data[i]["annotation"] = reference_rows[i].get("annotation", "")
                if "answer" in data[i]:
                    del data[i]["answer"]
            with open(file_path, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
def download_dataset(dataset_name, split, file_name):
    from datasets import load_dataset
    # 1) 读数据集
    ds = load_dataset(dataset_name, split=split)  # 如果不是 train，改成 validation/test/...
    # 2) 导出为 jsonl
    ds.to_json(file_name, orient="records", lines=True, force_ascii=False)
    print(ds)
    print("saved to BeyondAIME_train.jsonl")

if __name__ == "__main__":
    # jsonl_file = './MATH_train.jsonl'
    # xlsx_file = './Problem batch 1.xlsx'
    # id_matching('./new_match/batch2.xlsx', './MATH_test.jsonl', "test")
    # id_matching('./new_match/batch2_lc.xlsx', './MATH_train.jsonl', "train")
    # input_file = './MathTrap_batch_1_0811.xlsx'
    # output_file = './MathTrap_batch_1_0811.jsonl'
    # x2j('./new_problem_pools.xlsx', './example_problems.jsonl')
    # scan('./data/batch2_0907.jsonl', os.getenv('OPENROUTER_API_KEY'))
    # jsonl_file = './data/MathTrap_batch_1_2.jsonl'
    # xlsx_file = './data/MathTrap_batch_1_2.xlsx'
    # j2x(jsonl_file, xlsx_file)
    # paste_annotation('./outputs', './data/MathTrap_batch_1_2.jsonl', 'annotation', 'trap')
    # rows = [2, 34, 40, 47, 49, 59, 74, 80, 95, 97, 98, 101, 105, 119]
    # extract_rows("outputs/o4-mini-2025-04-16_batch_1_2_paraphrased_trap.jsonl", "outputs/o4-mini-2025-04-16_batch_1_2_paraphrased_trap_selected.jsonl", rows)
    # compare_jsonl_keys('./data/MathTrap_batch_1_2_delete.jsonl', './data/MathTrap_batch_1_0816.jsonl', 'paraphrased_original', 'paraphrased_trap')
    # scan_and_clean('./outputs/test')
    # find_rows_with_prefix("outputs/batch1/qwen3-235b-a22b-thinking-2507_batch_1_0816_paraphrased_trap.jsonl", 'generated_responses', 'Error')
    # find_file_errors("outputs/batch2", 'generated_responses', "Error:")
    # token_string_ratio("outputs/batch2/kimi-k2-0905_batch_2_0918_paraphrased_trap.jsonl")
    # modify_token("outputs/batch2", 20000)
    # find_early_stop("outputs/batch2")
    # paste_annotation("data/MathTrap_batch_1_0816.jsonl", "outputs/batch1")
    download_dataset("ByteDance-Seed/BeyondAIME", "test", "BeyondAIME.jsonl")
