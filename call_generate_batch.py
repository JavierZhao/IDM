import argparse
import os
from generate import infer

# This block is used for local models, use command like "CUDA_VISIBLE_DEVICES=0,1 python call_generate_batch.py" to run

# models_to_process = [
#     "../models/Llama-3.3-70B-Instruct"
# ]
# base_config = {
#     "data_dir": "./data",
#     "data_name": "test",
#     "start_idx": 0, # use "0,1" if you want to process multiple individual examples
#     "end_idx": -1,
#     "temperature": 0, # It is unjustable for OpenAI API o-series models
#     "max_tokens": 20000, # automacally change to 16384 for GPT-4o
#     "output_dir": "./outputs",
#     "top_p": 0.99,  # It is unjustable for OpenAI API o-series models. So it will be automatically ignore.
#     "problem_key": "paraphrased_original",
#     "use_unsolvable_prompt": False,
#     "use_fewshot_prompt": False,
#     "dtype": "bfloat16"
# }

# This block is used for API models
models_to_process = [
    # 'gpt-4.1-2025-04-14',
    # 'gpt-4o-2024-11-20',
    # "o3-2025-04-16",
    # 'o1-2024-12-17'
    # 'deepseek-r1-250528'
    # "openai/gpt-5"
    # "openai/o3"
    # "claude-sonnet-4-20250514",
    # "claude-opus-4-1-20250805"
    os.getenv("GEN_MODEL", "openai/gpt-4o-mini"),
    #"anthropic/claude-sonnet-4"
]
base_config = {
    "data_dir": ".",
    "data_name": "BeyondAIME_test",
    "start_idx": 0,
    "end_idx": -1,
    "temperature": 0.6, # It is unjustable for OpenAI API o-series models
    "max_tokens": 20000, # automacally change to 16384 for GPT-4o
    "output_dir": "./outputs",
    "top_p": 0.95,  # It is unjustable for OpenAI API o-series models. So it will be automatically ignore.
    "problem_key": "problem",
    "use_unsolvable_prompt": False,
    "use_fewshot_prompt": False,
    "thinking": False,
    "url": os.getenv("GEN_API_URL", "https://openrouter.ai/api/v1"),
    "api_key": os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    # key source: OPENROUTER_API_KEY or OPENAI_API_KEY
}

if __name__ == '__main__':
    if not base_config["api_key"]:
        raise ValueError(
            "Missing API key. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) before running."
        )

    for model in models_to_process:

        print(f"Generating with model: {model}")
        
        args = argparse.Namespace(
            model_path=None, # Set to None for API models
            model=model, # Set to None for local models
            **base_config
        )
        
        # Process start_idx like in generate.py parse_args()
        if isinstance(args.start_idx, str) and ',' in args.start_idx:
            args.start_idx_list = [int(x.strip()) for x in args.start_idx.split(',')]
            args.start_idx = None
            args.use_index_list = True
        else:
            args.start_idx = int(args.start_idx)
            args.start_idx_list = None
            args.use_index_list = False

        # If model is gpt-4o, set max_tokens to 16384
        if args.model and args.model and "gpt-4o" in args.model:
            args.max_tokens = 16384
        if "claude" in args.model:
            args.temperature = 1.0
        

        output_file = infer(args)
        print(f"Generation completed successfully for {model}!")
        print(f"Output saved to: {output_file}")


    print(f"\n{'='*60}")
    print("Batch generation completed for all models!")
    print(f"{'='*60}") 
