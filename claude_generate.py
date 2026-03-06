import json
import os
from datetime import datetime
from tqdm import tqdm
from string import Template
from utils.utils import  load_jsonl
import httpx
from openai import OpenAI
import requests
from utils.parser import extract_answer

def save_single_result(file_path, result, index, total_results):
    """Save a single result to file, handling updates for existing files."""
    # Load existing data
    if os.path.exists(file_path):
        existing_data = list(load_jsonl(file_path))   
    else:
        existing_data = []
    # Ensure list is large enough
    while len(existing_data) <= index:
        existing_data.append({})  
    # Update the specific index
    existing_data[index] = result   
    # Write all data back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in existing_data:
            if item:  # Skip empty items
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")
        f.flush()

def api_inference_single(question):
    """Process a single question through Anthropic API."""
    if USE_UNSOLVABLE_PROMPT:
        SYSTEM_PROMPT = "You are a math expert. Please reason step by step and show your step-by-step reasoning first, and then put your final answer within \\boxed{}. Please pay attention that the problem can be insolvable. If it is insolvable (e.g., there is a contradiction, ill-defined, or insufficient conditions), please say 'the problem is insolvable' in the final answer part."
    else:
        SYSTEM_PROMPT = "You are a math expert. Please reason step by step and show your step-by-step reasoning first, and then put your final answer within \\boxed{}."

    if 'claude' in MODEL:
        import anthropic
        try:
            client = anthropic.Anthropic(api_key=API_KEY, timeout=600.0)
            params = {"model": MODEL,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": MAX_TOKENS,
                "top_p": TOP_P,}    
            if THINKING:
                params.update({"thinking": {"type": "enabled","budget_tokens": int(MAX_TOKENS * 0.9)}})    
            if "4-1" not in MODEL:
                params.update({"temperature": TEMPERATURE,})
            response = client.messages.create(**params)
            token_usage = response.usage.output_tokens
            reasoning_content, generated_text = "", ""
            for block in response.content:
                if block.type == "thinking":
                    reasoning_content = block.thinking
                elif block.type == "text":
                    generated_text = block.text
            final_text = f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}" if reasoning_content else generated_text
        except Exception as e:
            print(f"API error: {e}")
            final_text = f"Error: {str(e)}" + f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}"
        return final_text, token_usage
    elif 'grok' in MODEL:
        try:
            client = OpenAI(base_url="https://api.x.ai/v1", api_key= API_KEY, timeout=httpx.Timeout(600.0))
            params = {"model": MODEL,
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "stream": True,
                        "stream_options": {"include_usage": True},}
            if "grok-3" in MODEL:
                params.update({
                    "reasoning_effort": "high",
                })
            generated_text, reasoning_content = "", ""
            token_usage = 0
            completion = client.chat.completions.create(**params)

            for chunk in completion:
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                if chunk.usage:
                    token_usage = chunk.usage.total_tokens - chunk.usage.prompt_tokens
                if len(chunk.choices) > 0 and chunk.choices[0].finish_reason == "length":
                    token_usage = MAX_TOKENS
                if len(chunk.choices) > 0 and chunk.choices[0].finish_reason and chunk.choices[0].finish_reason != "stop":
                    print(chunk.choices[0].finish_reason)

            final_text = f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}" if reasoning_content else generated_text   
        except Exception as e:
            print(f"API error: {e}")
            final_text = f"Error: {str(e)}" + f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}"
            token_usage = 0
        return final_text, token_usage
    elif 'gemini' in MODEL:
        try:
            from google import genai
            from google.genai import types
            thinking_text = ""
            output_text = ""
            token_usage = 0
            client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=600_000))
            if not THINKING and "flash" in MODEL:
                THINKING_BUDGET = 0
            else:
                THINKING_BUDGET = int(MAX_TOKENS * 0.9)
            response = client.models.generate_content_stream(
                model= MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET,
                                                        include_thoughts=True)
                ),
            )

            for chunk in response:
                if chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.text and part.thought:
                            thinking_text += part.text
                        elif part.text:
                            output_text += part.text
                token_usage = chunk.usage_metadata.total_token_count
            final_text = f"Reasoning:\n{thinking_text}\n\nOutput:\n{output_text}" if thinking_text else output_text
        except Exception as e:
            print(f"API error: {e}")
            final_text = f"Error: {str(e)}" + f"Reasoning:\n{thinking_text}\n\nOutput:\n{output_text}"
        return final_text, token_usage
    elif "/" in MODEL:
        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY, timeout=httpx.Timeout(600.0))
            params = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question}
                    ],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_tokens": MAX_TOKENS,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            if THINKING:
                params.update({"extra_body": {"reasoning": {"enabled": True}}})
            generated_text, reasoning_content = "", ""
            token_usage = 0
            completion = client.chat.completions.create(**params)
            for chunk in completion:
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'reasoning') and chunk.choices[0].delta.reasoning:
                    reasoning_content += chunk.choices[0].delta.reasoning
                if chunk.usage:
                    token_usage = chunk.usage.completion_tokens
                if len(chunk.choices) > 0 and chunk.choices[0].finish_reason == "length" and token_usage == 0:
                    token_usage = MAX_TOKENS
                if len(chunk.choices) > 0 and chunk.choices[0].finish_reason and chunk.choices[0].finish_reason != "stop":
                    print(chunk.choices[0].finish_reason)
            final_text = f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}" if reasoning_content else generated_text
            if len(final_text) > 1.8 *MAX_TOKENS and token_usage == 0:
                token_usage = MAX_TOKENS
            print(token_usage)
        except Exception as e:
            print(f"API error: {e}")
            final_text = f"Error: {str(e)}" + f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}"
        return final_text, token_usage
    elif MODEL == "deepseek-reasoner" or MODEL == "deepseek-chat":
        try:
            client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com", timeout=httpx.Timeout(600.0))
            params = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question}
                    ],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_tokens": MAX_TOKENS,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            # if THINKING:
            #     params.update({"extra_body": {"reasoning": {"enabled": True}}})
            generated_text, reasoning_content = "", ""
            token_usage = 0
            completion = client.chat.completions.create(**params)
            for chunk in completion:
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                if chunk.usage:
                    token_usage = chunk.usage.completion_tokens
                if len(chunk.choices) > 0 and chunk.choices[0].finish_reason == "length" and token_usage == 0:
                    token_usage = MAX_TOKENS
                if len(chunk.choices) > 0 and chunk.choices[0].finish_reason and chunk.choices[0].finish_reason != "stop":
                    print(chunk.choices[0].finish_reason)
            final_text = f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}" if reasoning_content else generated_text
            # if len(final_text) > 1.8 *MAX_TOKENS and token_usage == 0:
            #     token_usage = MAX_TOKENS
        except Exception as e:
            print(f"API error: {e}")
            final_text = f"Error: {str(e)}" + f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}"
        return final_text, token_usage
    elif "qwen" in MODEL:
        try:
            client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=API_KEY, timeout=httpx.Timeout(600.0))
            params = {
                    "model": MODEL,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_tokens": MAX_TOKENS,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            if "8b" in MODEL or "thinking" in MODEL:
                params.update({"extra_body": {"thinking_budget": int(0.9 * MAX_TOKENS)}, "max_tokens": int(0.1 * MAX_TOKENS)})
            generated_text, reasoning_content = "", ""
            token_usage = 0
            completion = client.chat.completions.create(**params)
            for chunk in completion:
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                if chunk.usage:
                    token_usage = chunk.usage.completion_tokens
            final_text = f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}" if reasoning_content else generated_text
        except Exception as e:
            print(f"API error: {e}")
            final_text = f"Error: {str(e)}" + f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}"
        return final_text, token_usage        
    else:
        try:
            
            final_text = ""
            token_usage = 0 
            client = OpenAI(api_key=API_KEY, timeout=httpx.Timeout(600.0))
            if any(name in MODEL for name in ['o4', 'o1', 'o3', 'gpt-5']):
                completion = client.responses.create(
                model=MODEL,
                reasoning={"summary": "auto"},
                max_output_tokens=MAX_TOKENS,
                instructions = SYSTEM_PROMPT,
                input = question
                )
                summaries = []
                for item in completion.output:
                    if item.type == "reasoning":
                        for s in getattr(item, "summary", []):
                            summaries.append(s.text)
                reasoning_summary = "\n\n".join(summaries)
                token_usage = completion.usage.output_tokens
                final_text = f"Reasoning:\n{reasoning_summary}\n\nOutput:\n{completion.output_text}"
                return final_text, token_usage
            else:
                completion = client.responses.create(
                        model=MODEL,
                        max_output_tokens=MAX_TOKENS,
                        instructions = SYSTEM_PROMPT,
                        input = question,
                        temperature = TEMPERATURE,
                        top_p = TOP_P
                        )
                output_text = completion.output_text
                token_usage = completion.usage.output_tokens
                return output_text, token_usage
        except Exception as e:
            print(f"API error: {e}")
            final_text = f"Error: {str(e)}" + final_text
        return final_text, token_usage


def main():
    
    data_file = f"{DATA_DIR}/{DATA_NAME}.jsonl"
    full_examples = list(load_jsonl(data_file))
    print(f"Loaded {len(full_examples)} examples from {data_file}")
    if SPECIFIC_INDICES is not None:
        examples = []
        indices_to_process = []
        for idx in SPECIFIC_INDICES:
            if 0 <= idx < len(full_examples):
                examples.append(full_examples[idx])
                indices_to_process.append(idx)
        print(f"Processing specific indices: {indices_to_process}")
    else:
        end_idx = END_IDX if END_IDX != -1 else len(full_examples)
        examples = full_examples[START_IDX:end_idx]
        indices_to_process = list(range(START_IDX, min(end_idx, len(full_examples))))
        print(f"Processing range {START_IDX}:{end_idx}")
    print(f"Total examples to process: {len(examples)}")
    parts = DATA_NAME.split('batch')
    batch_part = 'batch' + parts[-1] if len(parts) > 1 else DATA_NAME
    model_name = MODEL.split('/')[-1] if '/' in MODEL else MODEL
    if THINKING:
        model_name += '_thinking'
    out_file = f'{OUTPUT_DIR}/{model_name}_{batch_part}_{PROBLEM_KEY}.jsonl'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(out_file):
        existing_output = list(load_jsonl(out_file))
    else:
        existing_output = []
    file_exists = len(existing_output) > 0
    if file_exists:
        print(f"Output file ({out_file}) exists, will update specified indices")
    else:
        print(f"Creating new output file: {out_file}")
    for i, (example, idx) in enumerate(tqdm(zip(examples, indices_to_process), 
                                            total=len(examples), 
                                            desc="Processing examples")):
        question = example.get(PROBLEM_KEY, None)
        if question is None:
            print(f"Warning: No question found for index {idx} with key '{PROBLEM_KEY}'")
            continue
        print(f"\nProcessing example {i+1}/{len(examples)} (index {idx})...")
        token_usage = 0
        trial = 0
        generated_response = ""
        while (generated_response.endswith("Output:\n") and token_usage < 19900) or token_usage == 0 or generated_response.startswith("Error:") or ( extract_answer(generated_response) == "" and token_usage < 19000):
            generated_response, token_usage = api_inference_single(question)
            trial += 1
            if trial > 2:
                break
        result = {
            "question": question,
            "generated_responses": [generated_response],
            "token_usage": token_usage,
        }
        # Add gold answer or annotation if available
        if "original" in PROBLEM_KEY and 'answer' in example:
            result['gold_answer'] = example.get('answer', None)
        else:
            result['annotation'] = example.get('annotation', None)
        # Save result immediately
        save_single_result(out_file, result, idx, len(full_examples))
        print(f"Saved result for index {idx}")
    
    # Print final summary
    print(f"\n{'='*50}")
    print(f"Generation Complete!")
    print(f"Total examples processed: {len(examples)}")
    print(f"Output file: {out_file}")
    print(f"{'='*50}")

if __name__ == "__main__":


    
    API_KEY_OR = os.getenv("OPENROUTER_API_KEY")
    API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
    API_KEY_XAI = os.getenv("XAI_API_KEY")
    API_KEY_ANTHROPIC = os.getenv("ANTHROPIC_API_KEY")
    API_KEY_OPENAI = os.getenv("OPENAI_API_KEY")
    API_KEY_DS = os.getenv("DEEPSEEK_API_KEY")
    API_KEY_ALIYUN = os.getenv("ALIYUN_API_KEY")
    MAX_TOKENS = 20000
    DATA_DIR = "./data"
    


    OUTPUT_DIR = "./outputs/batch1"
    DATA_NAME = "MathTrap_batch_1_0816" #"batch_2_0918" #"MathTrap_batch_1_0816"  # Name of your data file (without .jsonl extension)
    START_IDX = 0  # Starting index
    END_IDX = -1  # Ending index (-1 means process till the end)
    USE_UNSOLVABLE_PROMPT = False  # Whether to use prompt that considers unsolvable problems
    
    SPECIFIC_INDICES = None  # [60]
    # [36, 71, 85, 107, 129, 140, 159, 0, 1, 3, 4, 5, 6, 7, 9, 11]
   # [  12, 13, 14, 15, 17, 18, 21, 25, 27, 32, 33, 34, 35, 37, 39, 40, 43, 45, 47, 48, 51, 53] 
   # 56, 57, 58, 59, 60, 61, 62, 63, 65, 68, 69, 70, 72, 73, 76, 77, 78, 79,
   #  80, 81, 84, 86, 88, 89, 90, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 
   # 104, 105, 108, 109, 110, 111, 113, 114, 115, 116, 118, 119, 120, 121, 123,
   #  125, 128, 130, 133, 134, 135, 138, 139, 141, 142, 143, 144, 145, 147, 
   # 148, 149, 150, 151, 152, 153, 154, 155]

    # for MODEL, THINKING in [("moonshotai/kimi-k2-0905", False)]: # only Claude, Gemini-2.5-flash, and DeepSeek 3.1 need to use THINKING to switch between thinking and non-thinking mode
    # for MODEL, THINKING in [("gemini-2.5-flash", False)]: # only Claude, Gemini-2.5-flash, and DeepSeek 3.1 need to use THINKING to switch between thinking and non-thinking mode
    for MODEL, THINKING in [("gemini-2.5-flash", False)]: # only Claude, Gemini-2.5-flash, and DeepSeek 3.1 need to use THINKING to switch between thinking and non-thinking mode
        for PROBLEM_KEY, SPECIFIC_INDICES in [ ("paraphrased_original", [26, 79])]:
            if "qwen3-235b-a22b-thinking-2507" in MODEL or "qwen3-8b" in MODEL or "qwen3-30b-a3b-thinking-2507" in MODEL:
                TEMPERATURE, TOP_P, TOP_K, API_KEY = 0.6, 0.95, 20, API_KEY_ALIYUN
            elif "qwen3-235b-a22b" in MODEL or "qwen3-30b-a3b-instruct-2507" in MODEL:
                TEMPERATURE, TOP_P, TOP_K, API_KEY = 0.7, 0.8, 20, API_KEY_ALIYUN
            elif MODEL == "microsoft/phi-4-reasoning-plus" or"phi-4-reasoning" in MODEL:
                TEMPERATURE, TOP_P, TOP_K, API_KEY = 0.8, 0.95, 50, API_KEY_OR
            elif MODEL == "openai/gpt-oss-120b" or MODEL == "openai/gpt-oss-20b":
                TEMPERATURE, TOP_P, API_KEY = 1, 1, API_KEY_OR
            elif "gpt-oss" in MODEL:
                TEMPERATURE, TOP_P, API_KEY = 1, 1, API_KEY_OPENAI
            elif MODEL == "deepseek-reasoner" or MODEL == "deepseek-chat":
                TEMPERATURE, TOP_P, API_KEY = 0, 0.95, API_KEY_DS
            elif MODEL == "meta-llama/llama-4-maverick" or MODEL == "meta-llama/llama-4-scout" or MODEL == "deepseek/deepseek-chat-v3.1":
                TEMPERATURE, API_KEY, TOP_P = 0, API_KEY_OR, 0.95
            elif MODEL == "moonshotai/kimi-k2-0905":
                TEMPERATURE, TOP_P, API_KEY = 0.6, 0.95, API_KEY_OR
            elif "gemini" in MODEL:
                TEMPERATURE, TOP_P, TOP_K, API_KEY = 1, 0.95, 64, API_KEY_GEMINI
            elif "grok" in MODEL:
                TEMPERATURE, TOP_P, API_KEY = 0.6, 0.95, API_KEY_XAI
            elif "claude" in MODEL:
                TEMPERATURE, TOP_P, API_KEY = 1, 0.95, API_KEY_ANTHROPIC
            elif "gpt-4" in MODEL:
                TEMPERATURE, TOP_P, API_KEY = 0.6, 0.95, API_KEY_OPENAI
                if "gpt-4o-" in MODEL:
                    MAX_TOKENS = 16000
            elif 'o1' in MODEL or 'o4' in MODEL or "o3" in MODEL or "gpt-5" in MODEL:
                API_KEY = API_KEY_OPENAI
            if not API_KEY:
                raise ValueError(f"Missing API key for selected model: {MODEL}")
            main()

