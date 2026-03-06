import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import json
#from transformers import AutoTokenizer

# Disable torch compilation to avoid CUDA linking issues
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

#from vllm import LLM, SamplingParams
import os
import argparse
#import vllm.envs as envs
from datetime import datetime
from tqdm import tqdm
from utils.utils import  load_jsonl#, set_seed, save_jsonl, construct_prompt
from utils.data_loader import load_data
from math import comb
from string import Template
from openai import OpenAI
import anthropic

# envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"

def load_existing_output(file_path):
    """Load existing output file if it exists."""
    if not os.path.exists(file_path):
        return []
    
    existing_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                existing_data.append(json.loads(line.strip()))
    return existing_data

def update_existing_output(existing_data, new_data, indices_to_update):
    """Update existing output data with new data at specified indices."""
    # Create a copy of existing data
    updated_data = existing_data.copy()
    
    # Extend the list if needed
    max_index = max(indices_to_update) if indices_to_update else -1
    while len(updated_data) <= max_index:
        updated_data.append({})
    
    # Update specified indices with new data
    for i, new_item in enumerate(new_data):
        if i < len(indices_to_update):
            update_index = indices_to_update[i]
            if update_index < len(updated_data):
                updated_data[update_index] = new_item
            else:
                # Extend list if necessary
                while len(updated_data) <= update_index:
                    updated_data.append({})
                updated_data[update_index] = new_item
    
    return updated_data

def save_output_data(file_path, data):
    """Save data to output file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            if item:  # Skip empty items
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")
        f.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help="model dir (for VLLM)")
    parser.add_argument('--url', type=str, default=None, help="API endpoint URL (for API mode)")
    parser.add_argument('--api_key', type=str, default=None, help="API key (for API mode)")
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help="Model name for API mode (e.g., gpt-3.5-turbo, gpt-4)")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument('--start_idx', type=str, default="0", help="data[start:end] or comma-separated list of indices")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int) 
    parser.add_argument("--use_unsolvable_prompt", action="store_true")
    parser.add_argument("--unsolvable_prompt_file_path", default="./prompts/few_shot.tmpl", type=str)
    parser.add_argument("--use_fewshot_prompt", action="store_true")
    parser.add_argument("--fewshot_prompt_file_path", default="./prompts/few_shot.tmpl", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--top_p", type=float, default=None, help="Optional: nucleus sampling parameter. If not provided, will use default behavior.")
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument('--problem_key', type=str, default='paraphrased_original', help='Specify the key for the problem; default is "paraphrased_original"')
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 and not args.top_p else args.top_p # top_p must be 1 when using greedy 
    
    # Parse start_idx - can be single number or comma-separated list
    if ',' in args.start_idx:
        args.start_idx_list = [int(x.strip()) for x in args.start_idx.split(',')]
        args.start_idx = None
        args.use_index_list = True
    else:
        args.start_idx = int(args.start_idx)
        args.start_idx_list = None
        args.use_index_list = False
    
    # Validate arguments
    if args.url or args.api_key:
        print("Using API mode")
    elif args.model_path:
        print("Using VLLM mode")
    else:
        raise ValueError("Either provide --url or --token for API mode, or --model_path for VLLM mode")
    
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def api_inference(prompt_batch, args, system_prompt, api_type="openai"):
    if api_type == "openai":
        client = OpenAI(api_key=args.api_key, base_url=args.url if args.url else None) 
        results = []
        for messages in tqdm(prompt_batch, desc="API inference"):
            try:
                use_stream = not (args.url and "openrouter" in args.url.lower())
                params = {
                    "model": args.model,
                    "messages":  [{"role": "system", "content": system_prompt}, {"role": "user", "content": messages}],
                    "stream": use_stream,
                }
                if any(model in args.model for model in ['o4', 'o1', 'o3', 'gpt-5']):
                    params.update({
                        "reasoning_effort": "medium",
                        "max_completion_tokens": args.max_tokens,
                    })
                else:
                    params.update({
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "top_p": args.top_p,
                    })
                generated_text = ""
                reasoning_content = ""
                if use_stream:
                    stream = client.chat.completions.create(**params)
                    for chunk in stream:
                        choices = getattr(chunk, "choices", None)
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)
                        if not delta:
                            continue

                        content_piece = getattr(delta, "content", None)
                        if isinstance(content_piece, str):
                            generated_text += content_piece
                        elif content_piece:
                            generated_text += str(content_piece)

                        reasoning_piece = getattr(delta, "reasoning_content", None)
                        if reasoning_piece is None:
                            reasoning_piece = getattr(delta, "reasoning", None)
                        if isinstance(reasoning_piece, str):
                            reasoning_content += reasoning_piece
                        elif reasoning_piece:
                            reasoning_content += str(reasoning_piece)
                else:
                    completion = client.chat.completions.create(**params)
                    message = completion.choices[0].message if completion.choices else None
                    if message and getattr(message, "content", None):
                        if isinstance(message.content, str):
                            generated_text = message.content
                        else:
                            generated_text = str(message.content)

                if reasoning_content:
                    final_text = f"Reasoning:\n{reasoning_content}\n\nOutput:\n{generated_text}"
                else:
                    final_text = generated_text
            except Exception as e:
                print(f"API error: {e}")
                final_text = f"Error: {str(e)}"
            results.append(final_text)
    elif api_type == "anthropic":
        client = anthropic.Anthropic(api_key=args.api_key)
        results = []
        for messages in tqdm(prompt_batch, desc="API inference"):
            try:
                params = {
                    "model": args.model,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": messages}],
                    "max_tokens": args.max_tokens,
                    "top_p": args.top_p,
                }
                if args.thinking:
                    params.update({
                                    "thinking": {
                                        "type": "enabled",
                                        "budget_tokens": int(args.max_tokens * 0.75)
                                    }
                    })
                if "4-1" not in args.model:
                    params.update({
                        "temperature": args.temperature,
                    })
                response = client.messages.create(**params)
                if args.thinking:
                    thinking_text = ""
                    output_text = ""
                    for block in response.content:
                        if block.type == "thinking":
                            thinking_text = f"Reasoning:\n{block.thinking}\n\n"
                        elif block.type == "text":
                            output_text = f"Output:\n{block.text}"
                    final_text = thinking_text + output_text
                else:
                    for block in response.content:
                        if block.type == "text":
                            final_text = block.text
                            break
            except Exception as e:
                print(f"API error: {e}")
                final_text = f"Error: {str(e)}"
            results.append(final_text)
    return results

def infer(args):
    # Determine inference mode
    # use_api = args.url and args.api_key
    use_vllm = args.model_path
    if use_vllm:
        use_api = False
    else:
        use_api = args.model

    if use_api:
        # print(f"Using API endpoint: {args.url}")
        model_name = args.model.split('/')[-1] if '/' in args.model else args.model
        tokenizer = None  # We'll handle tokenization differently for API
    else:
        model_path = args.model_path
        print(f"current eval model: {model_path}")
        model_name = args.model_path.split('/')[-1] if '/' in args.model_path else args.model_path
    if args.use_unsolvable_prompt:
        system_prompt = "You are a math expert. Please reason step by step and show your step-by-step reasoning first, and then put your final answer within \\boxed{}. Please pay attention that the problem can be insolvable. If it is insolvable (e.g., there is a contradiction, ill-defined, or insiffucient conditions), please say 'the probolem is insolvable' in the final answer part."
    else:
        system_prompt = "You are a math expert. Please reason step by step and show your step-by-step reasoning first, and then put your final answer within \\boxed{}."
    # Load and slice data
    data_file = f"{args.data_dir}/{args.data_name}.jsonl"
    full_examples = list(load_jsonl(data_file))

    # Handle different indexing modes
    if args.use_index_list:
        # Use specific indices
        examples = []
        indices_to_process = []
        for idx in args.start_idx_list:
            if 0 <= idx < len(full_examples):
                examples.append(full_examples[idx])
                indices_to_process.append(idx)
        print(f"Processing specific indices: {indices_to_process}")
    else:
        # Use range-based indexing
        if args.end_idx == -1:
            args.end_idx = len(full_examples)
        examples = full_examples[args.start_idx:args.end_idx]
        indices_to_process = list(range(args.start_idx, min(args.end_idx, len(full_examples))))
        print(f"Processing range {args.start_idx}:{args.end_idx}")
    
    print(f"len(examples): {len(examples)}")
    
    # Setup output file
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    parts = args.data_name.split('batch')
    batch_part = 'batch' + parts[-1] if len(parts) > 1 else args.data_name
    if args.thinking:
        model_name += '_thinking'
    out_file = f'{args.output_dir}/{model_name}_{batch_part}_{args.problem_key}.jsonl'
    
    os.makedirs(f'./{args.output_dir}', exist_ok=True)
    
    # Load existing output if it exists
    existing_output = load_existing_output(out_file)
    file_exists = len(existing_output) > 0
    
    if file_exists:
        print(f"Output file ({out_file}) exists, will update specified indices")
    else:
        print(f"Creating new output file: {out_file}")
    # Initialize models based on mode

    if use_vllm:
        # VLLM initialization
        print("--------------------------------")
        print(use_vllm)
        
        sampling_params = SamplingParams(
            temperature=args.temperature, 
            max_tokens=args.max_tokens, 
            n=1,
            top_p=args.top_p,
        )   
        
        # Setup GPU and model
        available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
        if len(available_gpus) == 1:
            envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
        print(f"available_gpus: {available_gpus}")
        print("*******************************")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(
            model=model_path, 
            tensor_parallel_size=len(available_gpus),
            trust_remote_code=True, 
            gpu_memory_utilization=0.90,
            dtype=args.dtype,
        )
        

    else:
        # API mode - no model initialization needed
        llm = None
        sampling_params = None
        tokenizer = None
    
    # Prepare prompts for all examples
    prompt_batch = []
    for example in examples:
        question = example.get(args.problem_key, None)
        # if args.use_unsolvable_prompt:
            # with open(args.unsolvable_prompt_file_path, "r", encoding="utf-8") as f:
            #     template = Template(f.read())
            #     cur_prompt = template.substitute(original_problem=question)
        # elif args.use_fewshot_prompt:
        #     with open(args.fewshot_prompt_file_path, "r", encoding="utf-8") as f:
        #         template = Template(f.read())
        #         cur_prompt = template.substitute(original_problem=question)
        # else:
        #     cur_prompt = question
            
        if use_vllm and tokenizer:
            # Use chat template for VLLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
            prompt_batch.append(cur_prompt)
        elif use_api:
            # For API, store as message objects for direct use with OpenAI format
            # messages = [
            #     {"role": "system", "content": "You are a math expert. Please reason step by step and show your step-by-step reasoning first, and then put your final answer within \\boxed{}."},
            #     {"role": "user", "content": cur_prompt}
            # ]
            prompt_batch.append(question)
    
    # Generate responses for all examples
    print('-------------before inference--------------------')
    if use_vllm:
        completions = llm.generate(prompt_batch, sampling_params)
        generated_responses = [completions[i].outputs[0].text for i in range(len(completions))]
    else:
        if "claude" in args.model.lower() and args.url is None:
            generated_responses = api_inference(prompt_batch, args, system_prompt, api_type="anthropic")
        else:
            generated_responses = api_inference(prompt_batch, args, system_prompt, api_type="openai")
    print('-------------after inference--------------------')
    
    # Process results - only store generation results
    new_outputs = []
    
    for i, example in enumerate(examples):
        question = example.get(args.problem_key, None)
        # print(question)
        generated_response = generated_responses[i]
        
        result = {
            "question": question,
            "generated_responses": [generated_response],
        }
        
        # Store original answer for later assessment if available
        if "original" in args.problem_key and 'answer' in example:
            result['gold_answer'] = example.get('answer', None)
        else:
            result['annotation'] = example.get('annotation', None)
        
        new_outputs.append(result)
    
    # Save results - handle existing file updates or new file creation
    if file_exists:
        # Update existing output with new results
        final_output = update_existing_output(existing_output, new_outputs, indices_to_process)
        save_output_data(out_file, final_output)
        print(f"Updated {len(indices_to_process)} entries in existing file")
    else:
        # Create new file with results
        save_output_data(out_file, new_outputs)
        print(f"Created new file with {len(new_outputs)} entries")
    
    # Print generation results
    total_examples = len(examples)
    print(f"\nGeneration Results:")
    print(f"Total examples processed: {total_examples}")
    print(f"Output file: {out_file}")
    
    return out_file

if __name__ == "__main__":
    args = parse_args()
    infer(args)
