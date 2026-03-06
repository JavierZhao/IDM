
import argparse
from assess import assess_responses
import os


folder = "outputs/batch2"
files_to_process = []
for file_path in os.listdir(folder):
    # if file_path.endswith("original.jsonl") and "oss" not in file_path.lower() and "deepseek" not in file_path.lower() and "qwen3" not in file_path.lower():
    # if file_path.endswith("original.jsonl") and ("qwen2.5" in file_path.lower() or "phi" in file_path.lower() or "llama" in file_path.lower()):
    # if file_path.endswith("original.jsonl") and ("gemini" in file_path.lower() or "grok" in file_path.lower() or "claude" in file_path.lower()):
    # if file_path.endswith("original.jsonl") and "qwen3" in file_path.lower():
    if file_path.endswith("original.jsonl") and ( 'gpt-5' in file_path.lower()):
        file_path = os.path.join(folder, file_path)
        files_to_process.append(file_path)
print(f"Found {len(files_to_process)} files to process")
API_KEY_OPENAI = os.getenv("OPENAI_API_KEY")
API_KEY_OR = os.getenv("OPENROUTER_API_KEY")

if not API_KEY_OR:
    raise ValueError("Missing OPENROUTER_API_KEY environment variable.")

# Essential for Windows multiprocessing
for file_path in files_to_process:
    args = argparse.Namespace(
        input_file=file_path,
        use_llm_check=True,
        api_key=API_KEY_OR,
        model="moonshotai/kimi-k2-0905", # "gpt-4o-2024-11-20",#"moonshotai/kimi-k2",#,
        url="https://openrouter.ai/api/v1",
        max_retries=3,
        retry_delay=0.5
    )
    
    # Call the function
    try:
        assess_responses(args)
        print("Assessment completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("\nTrying to continue with next file...")

