import json
import argparse
import os
# from sympy.core.numbers import I
from tqdm import tqdm
from utils.utils import load_jsonl
from utils.parser import extract_answer
from utils.grader import check_is_correct
import openai
import time
# from evaluate import answer_check, extract_predicted_answer
# from evaluation.eval_utils import math_equal
from typing import Optional
import concurrent.futures
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help="Input file path (output from generate.py)")
    parser.add_argument('--use_llm_check', action='store_true', default=True, help="Use LLM for equivalence checking when initial assessment is wrong")
    parser.add_argument('--api_key', type=str, required=True, help="API key for LLM service")
    parser.add_argument('--model', type=str, required=True, help="LLM model to use for equivalence checking")
    parser.add_argument('--url', type=str, required=False, help="API URL for LLM service (optional, uses default OpenAI if not provided)")
    parser.add_argument('--max_retries', type=int, default=3, help="Maximum retries for LLM API calls")
    parser.add_argument('--retry_delay', type=float, default=1.0, help="Delay between retries in seconds")
    args = parser.parse_args()
    return args

def call_llm_for_equivalence(prompt: str, client, model: str, max_retries: int = 3, retry_delay: float = 1.0) -> Optional[bool]:
    """
    Call LLM to check if two answers are equivalent
    Returns True if equivalent, False if not equivalent, None if error
    """ 
    for attempt in range(max_retries):
        try:
            # response = client.chat.completions.create(
            #     model=model,
            #     messages=[{"role": "user", "content": prompt}],
            #     max_tokens=3000
            # )

            if "/" in model:
                completion = client.chat.completions.create(
                        model=model,
                            messages=[{"role": "user", "content": prompt}]
                            )
                result = completion.choices[0].message.content
            else:
                response = client.responses.create(model=model, input=prompt)
                result = response.output_text

            # Extract the final decision
            marker = result.split("Equivalent")[-1].lower()
            if "true" in marker:
                return True, result
            elif "false" in marker:
                return False, result
            else:
                print(f"Warning: Could not parse LLM response: {result[-100:]}")
                return "Fail to find key words", result
        except Exception as e:
            print(f"Error calling LLM (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"Failed to get LLM response after {max_retries} attempts")
                return "Fail to call the LLM", "Fail to call the LLM"


def assess_responses(args):
    """
    Read responses from the input file, assess correctness, and save back to the same file
    """
    print(f"Reading responses from: {args.input_file}")
    
    # Initialize OpenAI client if using LLM check
    client = None
    prompt_template = None
    llm_corrections_file = None
    if args.use_llm_check:
        # Set up API key and client
        if args.url:
            client = openai.OpenAI(api_key=args.api_key, base_url=args.url)
        else:
            client = openai.OpenAI(api_key=args.api_key)
        input_filename = os.path.splitext(os.path.basename(args.input_file))[0]
        
        llm_corrections_dir = "LLM_correction"
        os.makedirs(llm_corrections_dir, exist_ok=True)
        llm_corrections_file = os.path.join(llm_corrections_dir, f"{input_filename}_LLM.txt")
        with open("prompt/equivalence_check.tmpl", 'r', encoding='utf-8') as f:
            final_prompt_template = f.read()
        with open("prompt/equivalence_check_extract_answer.tmpl", 'r', encoding='utf-8') as f:
            final_prompt_template_extract_answer = f.read()
    # Load the generated responses
    examples = list(load_jsonl(args.input_file))
    correct_cnt = 0
    llm_corrections = 0
    llm_calls_made = 0
    llm_errors = 0
    llm_correction_cases = []  # Store cases where LLM corrected the assessment
    empty_response_cases = []  # Store cases where generated response is empty
    total_answers = 0
    empty_responses_count = 0
    # Process each example and assess correctness
    for i , example in enumerate(tqdm(examples, desc="Assessing responses")):
        if "oss-120b_batch_2" in args.input_file and i == 91:
            continue
        if "original" in args.input_file:
            gt_ans = example['gold_answer']
        generated_responses = example['generated_responses']
        question = example['question']
        # Assess each generated response
        generated_answers = []
        answers_correctness = []
        llm_equivalence_checks = []
        for response_idx, generated_response in enumerate(generated_responses):
            total_answers += 1
            # Check if the generated response is empty
            if not generated_response or generated_response.strip() == "":
                empty_responses_count += 1
                empty_response_case = {
                    'example_index': i,
                    'response_index': response_idx,
                    'question': question,
                    # 'ground_truth': str(gt_ans),
                    'empty_response': True
                }
                empty_response_cases.append(empty_response_case)
                
                # For empty responses, skip assessment and mark as incorrect
                generated_answers.append("")
                answers_correctness.append(False)
                llm_equivalence_checks.append(None)
                continue


            generated_answer = extract_answer(generated_response)
            generated_answers.append(generated_answer)
            if "original" in args.input_file:
                is_correct = check_is_correct(generated_answer, gt_ans)
            # If initial check is wrong and LLM check is enabled, use LLM
            llm_check_result = None
            if args.use_llm_check  and not is_correct and client is not None:
                llm_calls_made += 1
                feed_response = generated_response if len(generated_response) < 600 else generated_response[-600:]
                prompt = final_prompt_template_extract_answer.format(ground_truth=gt_ans, question=question, response=feed_response)
                llm_check_result, analysis = call_llm_for_equivalence(prompt, client, args.model,  args.max_retries, args.retry_delay)
                # print(analysis)
                if isinstance(llm_check_result, str):
                    llm_errors += 1
                    llm_corrections += 1
                    # Store the correction case
                    correction_case = {
                        'question': question,
                        'ground_truth': str(gt_ans),
                        'prediction': str(generated_answer),
                        'original_assessment': 'Wrong',
                        'llm_assessment': str(llm_check_result),
                        'llm_analysis': analysis
                    }
                    llm_correction_cases.append(correction_case)
                elif llm_check_result is True:
                    is_correct = True  # LLM says they are equivalent
                    llm_corrections += 1
                    # Store the correction case
                    correction_case = {
                        'question': question,
                        'ground_truth': str(gt_ans),
                        'prediction': str(generated_answer),
                        'original_assessment': 'Wrong',
                        'llm_assessment': 'Correct',
                        'llm_analysis': str(analysis)
                    }
                    llm_correction_cases.append(correction_case)           
            
            llm_equivalence_checks.append(llm_check_result)
            if "original" in args.input_file:
                if is_correct:
                    correct_cnt += 1
                answers_correctness.append(is_correct)
        
        # Add assessment information directly to the example
        example['generated_answers'] = generated_answers
        if "original" in args.input_file:
            example['answers_correctness'] = answers_correctness
            example['correct_number'] = sum(answers_correctness)
    # Save the assessed results back to the same file
    print(f"Saving assessed results back to: {args.input_file}")
    with open(args.input_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False))
            f.write("\n")
        f.flush()
    # Save LLM corrections to separate file
    if args.use_llm_check and (llm_correction_cases or empty_response_cases):
        print(f"Saving LLM corrections to: {llm_corrections_file}")
        with open(llm_corrections_file, 'w', encoding='utf-8') as f:
            # Write LLM corrections section
            if llm_correction_cases:
                f.write("LLM Corrections - Cases where original assessment was wrong but LLM found them equivalent\n")
                f.write("=" * 80 + "\n\n")
                for case in llm_correction_cases:
                    f.write(f"Question: {case['question']}:\n")
                    f.write(f"Ground Truth: {case['ground_truth']}\n")
                    f.write(f"Prediction: {case['prediction']}\n")
                    f.write(f"Original Assessment: {case['original_assessment']}\n")
                    f.write(f"LLM Assessment: {case['llm_assessment']}\n")
                    f.write(f"LLM Analysis: {case['llm_analysis']}\n")
                    f.write("-" * 40 + "\n")
                f.write("\n")
            
            # Write empty responses section
            if empty_response_cases:
                f.write("Empty Responses - Cases where generated response was empty\n")
                f.write("=" * 60 + "\n\n")
                for case in empty_response_cases:
                    f.write(f"Example Index: {case['example_index']}\n")
                    f.write(f"Response Index: {case['response_index']}\n")
                    f.write(f"Question: {case['question']}\n")
                    # f.write(f"Ground Truth: {case['ground_truth']}\n")
                    f.write(f"Generated Response: [EMPTY]\n")
                    f.write("-" * 40 + "\n")
        
        print(f"Saved {len(llm_correction_cases)} LLM correction cases and {len(empty_response_cases)} empty response cases")
    if total_answers > 0:
        accuracy = correct_cnt / total_answers
        print(f"Correct answers: {correct_cnt}")
        print(f"Empty responses: {empty_responses_count}")
        print(f"Final Accuracy: {accuracy:.4f}")
        
        # Optionally save results to a summary file
        result_txt_file = "assessment_results_2.txt"
        llm_info = f" (LLM calls: {llm_calls_made}; LLM corrections: {llm_corrections}; LLM errors: {llm_errors})" if args.use_llm_check else ""
        empty_info = f" (Empty responses: {empty_responses_count})" if empty_responses_count > 0 else ""
        result_line = f"{args.input_file}\t{accuracy:.4f}\t{correct_cnt}/{total_answers}{llm_info}{empty_info}\n"
        write_mode = 'a' if os.path.exists(result_txt_file) else 'w'
        with open(result_txt_file, write_mode, encoding='utf-8') as f:
            f.write(result_line)
        print(f"Results summary saved to: {result_txt_file}")
    else:
        print("No assessable examples found (missing gold answers)")


if __name__ == "__main__":
    args = parse_args()
    assess_responses(args)