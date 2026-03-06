#!/usr/bin/env python3
"""
Thin wrapper that "used to call" the judge.
You can tweak arguments here instead of the CLI, then:

  python run_judge.py
"""

import os
import sys
import subprocess

folder = "outputs/batch1"
files_to_process = []
for file_path in os.listdir(folder):
    if file_path.endswith("trap.jsonl") and  any(model in file_path.lower() for model in ["gemini"]) :
        # any(model in file_path.lower() for model in ["gemini", "claude"]) #1
        # any(model in file_path.lower() for model in ["gpt"]) #2
        # any(model in file_path.lower() for model in ["o3", "o4", "grok"]) #3
        # any(model in file_path.lower() for model in ["kimi", "llama"]) #4
        # any(model in file_path.lower() for model in ["phi", "deepseek", "qwen2.5"]) #5
        # any(model in file_path.lower() for model in ["qwen3"]) #6
        file_path = os.path.join(folder, file_path)
        files_to_process.append(file_path)
print(f"Found {len(files_to_process)} files to process")

for input_file in files_to_process:
    # Edit these as needed or pull from env
    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
    OR_API_KEY       = os.getenv("OPENROUTER_API_KEY")
    XAI_API_KEY      = os.getenv("XAI_API_KEY")
    input_file       = input_file
    api_key_final    = XAI_API_KEY
    model_final      = "grok-4-fast-reasoning" # "moonshotai/kimi-k2-0905"
    api_key_process  = XAI_API_KEY
    model_process    = "grok-4-fast-reasoning" # "o4-mini-2025-04-16" # "o4-mini-2025-04-16" # "moonshotai/kimi-k2-0905" # "o4-mini-2025-04-16"
    template_dir     = "prompt"
    index_range      =  None # "38,"  # e.g. "0:100" or "1,4,6"
    mode             = "async"  # or "threaded"

    # Async knobs
    concurrency = "8"
    rpm         = "60"  # or "" to disable

    # Threaded knob (only if mode=threaded)
    workers = "8"

    if not api_key_final or not api_key_process:
        raise ValueError("Missing XAI_API_KEY environment variable.")

    cmd = [
        sys.executable, "trap_judge_parallel.py",
        "--input_file", input_file,
        "--api_key_final", api_key_final,
        "--model_final", model_final,
        "--api_key_process", api_key_process,
        "--model_process", model_process,
        "--template_dir", template_dir,
        "--mode", mode,
        "--concurrency", concurrency,
        "--workers", workers,
        "--url_final", "https://api.x.ai/v1", #"https://openrouter.ai/api/v1",
        "--url_process", "https://api.x.ai/v1"  #"https://openrouter.ai/api/v1"
        # "--max_retries", 3,
        # "--index_range", index_range,
        # "--retry_delay", 0.5,
    ]
    if rpm:
        cmd += ["--rpm", rpm]
    if index_range:
        cmd += ["--index_range", index_range]

    # print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"Judging completed successfully for {input_file}!")
    except Exception as e:
        print(f"Error with file {input_file}: {e}")
        continue
