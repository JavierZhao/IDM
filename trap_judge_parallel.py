#!/usr/bin/env python3
"""
Parallel Trap Problem Judge

- Async backend (default): asyncio + OpenAI Async client with concurrency/RPM caps
- Threaded backend (optional): thread pool around the existing sync LLM client

This keeps the 3-step pipeline per response sequential:
  Final Answer -> Identification -> Modification
but parallelizes across examples (and optionally across responses if you want).

Dependencies:
  pip install openai httpx tqdm

Usage (async, recommended):
  python trap_judge_parallel.py \
    --input_file path/to/data.jsonl \
    --api_key_final sk-... \
    --model_final o4-mini-2025-04-16 \
    --api_key_process sk-... \
    --model_process gpt-4o-mini-2024-07-18 \
    --concurrency 8 --rpm 60

Threaded backend:
  python trap_judge_parallel.py ... --mode threaded --workers 8
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
# from utils.parser import extract_answer
import httpx
from tqdm import tqdm

# --- OpenAI clients (both async and sync) ---
try:
    from openai import OpenAI, AsyncOpenAI
except Exception as _e:
    OpenAI = None
    AsyncOpenAI = None

# -----------------------
# Utilities
# -----------------------
def load_jsonl(path: str):
    """Simple JSONL loader (yields dicts)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -----------------------
# Core Judge
# -----------------------
class TrapProblemJudge:
    """Hierarchical LLM-as-a-judge framework for trap problem assessment"""

    def __init__(self, template_dir: str = "./prompt"):
        self.template_dir = Path(template_dir)
        self.load_templates()

    def load_templates(self):
        """Load all judge templates"""
        template_files = {
            'final_answer': 'final_answer_judge.tmpl',
            'identification': 'identification_judge.tmpl',
            'modification': 'problem_modification_judge.tmpl'
        }
        self.templates = {}
        for judge_name, filename in template_files.items():
            template_path = self.template_dir / filename
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    self.templates[judge_name] = f.read()
            else:
                print(f"Warning: Template {filename} not found (dir={self.template_dir})")
                self.templates[judge_name] = f"{judge_name} :: {{question}}\n---\n{judge_name} uses {{response}} and {{trap_annotation}}"

    # ---------- sync path (for threaded mode) ----------
    def judge_response(self, question: str, response: str, annotation: str,
                       llm_call_func_final, llm_call_func_process) -> Dict[str, Any]:
        results = {'judgments': [], 'results': []}
        feed_response = extract_answer(response)
        if feed_response == "":
            feed_response = response[-100:] if len(response) >= 100 else response
        fa_result, fa_reason = self.judge_final_answer(question, feed_response, annotation, llm_call_func_final)
        results['judgments'].append(fa_reason)
        results['results'].append(fa_result)
        if fa_result:
            return results

        id_result, id_reason = self.judge_identification(question, response, annotation, llm_call_func_process)
        results['judgments'].append(id_reason)
        results['results'].append(id_result)

        mod_result, mod_reason = self.judge_modification(question, response, annotation, llm_call_func_process)
        results['judgments'].append(mod_reason)
        results['results'].append(mod_result)
        return results

    def judge_final_answer(self, question: str, response: str, annotation: str, llm_call_func_final):
        prompt = self.templates['final_answer'].format(
            question=question, trap_annotation=annotation, response=response
        )
        judgment = llm_call_func_final(prompt)
        says_insolvable = self._extract_boolean(judgment, "says_insolvable")
        reasoning = self._extract_field(judgment, "reasoning")
        return says_insolvable, reasoning

    def judge_identification(self, question: str, response: str, annotation: str, llm_call_func_process):
        prompt = self.templates['identification'].format(
            question=question, response=response, trap_annotation=annotation
        )
        judgment = llm_call_func_process(prompt)
        identified_trap = self._extract_boolean(judgment, "identified_trap")
        reasoning = self._extract_field(judgment, "reasoning")
        return identified_trap, reasoning

    def judge_modification(self, question: str, response: str, annotation: str, llm_call_func_process):
        prompt = self.templates['modification'].format(
            question=question, response=response, trap_annotation=annotation
        )
        judgment = llm_call_func_process(prompt)
        attempted_modification = self._extract_boolean(judgment, "attempted_modification")
        reasoning = self._extract_field(judgment, "reasoning")
        return attempted_modification, reasoning

    # ---------- async path (for asyncio mode) ----------
    async def judge_response_async(self, question: str, response: str, annotation: str,
                                   llm_call_func_final_async, llm_call_func_process_async) -> Dict[str, Any]:
        results = {'judgments': [], 'results': []}
        if response == "":
            results['results'] = ["empty"]
            results['judgments'] = ["empty"]
            return results
        # feed_response = extract_answer(response)
        # if feed_response == "":
        feed_response = response[-100:] if len(response) >= 100 else response
        retry_count = 0
        while True:
            fa_result, fa_reason = await self.judge_final_answer_async(
                question, feed_response, annotation, llm_call_func_final_async
            )
            if not isinstance(fa_result, str):
                break
            retry_count += 1
            if retry_count >= 3:
                break
        results['judgments'].append(fa_reason)
        results['results'].append(fa_result)
        if fa_result:
            return results
        retry_count = 0
        while True:
            mod_result, mod_reason = await self.judge_modification_async(
                question, response, annotation, llm_call_func_process_async
            )
            if not isinstance(mod_result, str):
                break
            retry_count += 1
            if retry_count >= 3:
                break

        if mod_result:
            results['results'] = results['results'] + [True, True]
            results['judgments'] = results['judgments'] + ["default as true due to modification", mod_reason]
            return results
        retry_count = 0
        while True:
            id_result, id_reason = await self.judge_identification_async(
                question, response, annotation, llm_call_func_process_async
            )
            if not isinstance(id_result, str):
                break
            retry_count += 1
            if retry_count >= 3:
                break

        results['judgments'].append(id_reason)
        results['results'].append(id_result)
        results['judgments'].append(mod_reason)
        results['results'].append(mod_result)


        # id_result, id_reason = await self.judge_identification_async(
        #     question, response, annotation, llm_call_func_process_async
        # )
        # results['judgments'].append(id_reason)
        # results['results'].append(id_result)
        # mod_result, mod_reason = await self.judge_modification_async(
        #     question, response, annotation, llm_call_func_process_async
        # )
        # results['judgments'].append(mod_reason)
        # results['results'].append(mod_result)


        return results



    async def judge_final_answer_async(self, question: str, response: str, annotation: str, llm_call_func_final_async):
        prompt = self.templates['final_answer'].format(
            question=question, trap_annotation=annotation, response=response
        )
        judgment = await llm_call_func_final_async(prompt)
        says_insolvable = self._extract_boolean(judgment, "says_insolvable")
        reasoning = self._extract_field(judgment, "reasoning")
        return says_insolvable, reasoning

    async def judge_identification_async(self, question: str, response: str, annotation: str, llm_call_func_process_async):
        prompt = self.templates['identification'].format(
            question=question, response=response, trap_annotation=annotation
        )
        judgment = await llm_call_func_process_async(prompt)
        identified_trap = self._extract_boolean(judgment, "identified_trap")
        reasoning = self._extract_field(judgment, "reasoning")
        return identified_trap, reasoning

    async def judge_modification_async(self, question: str, response: str, annotation: str, llm_call_func_process_async):
        prompt = self.templates['modification'].format(
            question=question, response=response, trap_annotation=annotation
        )
        judgment = await llm_call_func_process_async(prompt)
        attempted_modification = self._extract_boolean(judgment, "attempted_modification")
        reasoning = self._extract_field(judgment, "reasoning")
        return attempted_modification, reasoning

    # ---------- helpers ----------
    def _extract_boolean(self, text: str, field: str) -> Optional[bool]:
        
        if field not in text:
            return "Invalid judgment"
        parts = text.split(field)
        after_field = parts[-1][:10]  # look a bit further to be safe
        s = after_field.lower()
        if 'true' in s or 'yes' in s:
            return True
        if 'false' in s or 'no' in s:
            return False
        return "Invalid judgment"

    def _extract_field(self, text: str, field: str) -> str:
        parts = text.split(field)
        if len(parts) < 2:
            return ""
        after_field = parts[-1].lstrip(':*_ \t')
        lines = after_field.split('\n')
        return lines[0].strip() if lines else ""

    def parse_index_range(self, index_range: Optional[str], total_examples: int) -> List[int]:
        if not index_range:
            return list(range(total_examples))
        if ':' in index_range:
            a, b = index_range.split(':')
            start = max(0, int(a))
            end = min(total_examples, int(b)) if int(b) > 0 else total_examples
            if start >= end:
                raise ValueError(f"Invalid range: start ({start+1}) must be less than end ({end})")
            return list(range(start, end))
        if ',' in index_range:
            if index_range[-1] == ',':
                return [int(index_range.split(',')[0].strip())]
            return sorted(int(x.strip()) for x in index_range.split(','))
        # single index
        return [int(index_range.strip())]


# -----------------------
# LLM call functions
# -----------------------
def create_llm_call_function_sync(
    api_key: str,
    model: str,
    url: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
):
    if OpenAI is None:
        raise RuntimeError("openai package not available")

    client = OpenAI(api_key=api_key, base_url=url, timeout=httpx.Timeout(120.0)) if url \
        else OpenAI(api_key=api_key, timeout=httpx.Timeout(120.0))

    def llm_call(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                if url and "openrouter" in url and "o4-mini-high" in model.lower():
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                    )
                    return completion.choices[0].message.content
                elif url and "openrouter" in url and "kimi-k2" in model.lower():
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return completion.choices[0].message.content
                elif (not url) and ('o4-mini' in model.lower()):
                    response = client.responses.create(model=model, reasoning={"effort": "high"}, input=prompt)
                    return response.output_text
                else:
                    completion = client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}]
                    )
                    return completion.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt) * (1 + 0.1 * random.random()))
                else:
                    return f"Error in LLM call after {max_retries} attempts: {e}"
    return llm_call


def create_llm_call_function_async(
    api_key: str,
    model: str,
    url: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    concurrency: int = 8,
    rpm: Optional[int] = None,
):
    if AsyncOpenAI is None:
        raise RuntimeError("openai package not available")

    client = AsyncOpenAI(api_key=api_key, base_url=url, timeout=httpx.Timeout(120.0)) if url \
        else AsyncOpenAI(api_key=api_key, timeout=httpx.Timeout(120.0))

    import asyncio
    sem = asyncio.Semaphore(concurrency)
    last_calls: List[float] = []

    async def _rate_limit():
        if rpm is None:
            return
        loop = asyncio.get_event_loop()
        now = loop.time()
        window = 60.0
        # prune
        while last_calls and (now - last_calls[0] >= window):
            last_calls.pop(0)
        if len(last_calls) >= rpm:
            sleep_for = window - (now - last_calls[0]) + 0.01
            await asyncio.sleep(max(0.0, sleep_for))

    async def llm_call(prompt: str) -> str:
        async with sem:
            for attempt in range(max_retries):
                try:
                    await _rate_limit()
                    if url and "openrouter" in url and "o4-mini-high" in model.lower():
                        completion = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                        )
                        last_calls.append(asyncio.get_event_loop().time())
                        return completion.choices[0].message.content
                    elif url and "openrouter" in url and "kimi-k2" in model.lower():
                        completion = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        last_calls.append(asyncio.get_event_loop().time())
                        return completion.choices[0].message.content
                    elif (not url) and ('o4-mini' in model.lower()):
                        resp = await client.responses.create(
                            model=model, reasoning={"effort": "high"}, input=prompt
                        )
                        last_calls.append(asyncio.get_event_loop().time())
                        return resp.output_text
                    elif 'grok' in model.lower():
                        # client = OpenAI(api_key=, base_url="https://api.x.ai/v1", timeout=httpx.Timeout(3600.0),)
                        response = await client.responses.create(model=model, input=prompt)
                        last_calls.append(asyncio.get_event_loop().time())
                        return response.output[1].content[0].text
                    else:
                        completion = await client.chat.completions.create(
                            model=model, messages=[{"role": "user", "content": prompt}]
                        )
                        last_calls.append(asyncio.get_event_loop().time())
                        return completion.choices[0].message.content
                except Exception as e:
                    if attempt < max_retries - 1:
                        import asyncio
                        await asyncio.sleep(retry_delay * (2 ** attempt) * (1 + 0.1 * random.random()))
                    else:
                        return f"Error in LLM call after {max_retries} attempts: {e}"
    return llm_call


# -----------------------
# Processing loops
# -----------------------
def _accumulate_stats(result, counters):
    counters['total'] += 1
    if result['results'][0] == "empty":
        counters['empty'] += 1
        return
    if result['results'][0] or result['results'][2]:
        counters['correct'] += 1
        counters['identified'] += 1
    elif result['results'][1]:
        counters['identified'] += 1
    if result['results'][0]:
        counters['expression'] += 1


def process_jsonl_file_threaded(
    judge: TrapProblemJudge,
    input_file: str,
    llm_call_func_final,
    llm_call_func_process,
    index_range: Optional[str],
    workers: int = 8,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_judgment_strings: List[str] = []
    examples = list(load_jsonl(input_file))
    indices_to_process = judge.parse_index_range(index_range, len(examples))
    selected = [(i, examples[i]) for i in indices_to_process]

    def _process_one(ex_idx: int, example: dict):
        question = example['question']
        annotation = example['annotation']
        header = (
            f"=== Problem #{ex_idx+1} ===\n"
            f"Question: {question}\n"
            f"Annotation: {annotation}\n"
        )
        problem_judgment_str = header
        results_pack = []
        for resp_idx, response in enumerate(example['generated_responses']):
            result = judge.judge_response(question, response, annotation, llm_call_func_final, llm_call_func_process)
            results_pack.append(result['results'])
            problem_judgment_str += f"=== Response #{resp_idx+1} ===\n"
            for j in result['judgments']:
                problem_judgment_str += f"--- Judgment ---\n{j.strip()}\n\n"
        example['judge_result'] = results_pack
        return ex_idx, example, problem_judgment_str, results_pack

    counters = {'total': 0, 'correct': 0, 'identified': 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one, ex_idx, ex_obj) for ex_idx, ex_obj in selected]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging examples"):
            ex_idx, updated_example, judgment_str, results_pack = fut.result()
            examples[ex_idx] = updated_example
            all_judgment_strings.append(judgment_str)
            # update stats
            for results_one in results_pack:
                # reconstruct result dict shape to reuse stat fn
                _accumulate_stats({'results': results_one, 'judgments': []}, counters)

    # print stats
    if counters['total'] > 0:
        print(f"Accuracy: {counters['correct'] / counters['total']}")
        print(f"Identification Rate: {counters['identified'] / counters['total']}")
    print(f"Total examples: {counters['total']}")
    print(f"Correct examples: {counters['correct']}")
    print(f"Identified examples: {counters['identified']}")

    # write back
    with open(input_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    
    judgments_txt_file = input_file.replace('.jsonl', "_judgments.txt")
    file_mode = 'a' if os.path.exists(judgments_txt_file) else 'w'
    with open(judgments_txt_file, file_mode, encoding='utf-8') as txtout:
        for s in all_judgment_strings:
            txtout.write(s.strip() + "\n\n")


async def process_jsonl_file_async(
    judge: TrapProblemJudge,
    input_file: str,
    llm_call_func_final_async,
    llm_call_func_process_async,
    index_range: Optional[str],
):
    import asyncio
    from tqdm.asyncio import tqdm_asyncio

    all_judgment_strings: List[str] = []
    examples = list(load_jsonl(input_file))
    indices_to_process = judge.parse_index_range(index_range, len(examples))
    selected = [(i, examples[i]) for i in indices_to_process]

    async def _process_one(ex_idx: int, example: dict):
        question = example['question']
        annotation = example['annotation']
        header = (
            f"=== Problem #{ex_idx+1} ===\n"
            f"Question: {question}\n"
            f"Annotation: {annotation}\n"
        )
        problem_judgment_str = header
        results_pack = []
        for resp_idx, response in enumerate(example['generated_responses']):
            result = await judge.judge_response_async(
                question, response, annotation, llm_call_func_final_async, llm_call_func_process_async
            )
            results_pack.append(result['results'])
            problem_judgment_str += f"=== Response #{resp_idx+1} ===\n"
            for j in result['judgments']:
                problem_judgment_str += f"--- Judgment ---\n{j.strip()}\n\n"
        example['judge_result'] = results_pack
        return ex_idx, example, problem_judgment_str, results_pack

    tasks = [_process_one(ex_idx, example) for ex_idx, example in selected]
    results = await tqdm_asyncio.gather(*tasks, desc="Judging examples")

    counters = {'total': 0, 'correct': 0, "expression": 0, 'identified': 0, "empty": 0}
    for ex_idx, updated_example, judgment_str, results_pack in sorted(results, key=lambda x: x[0]):
        examples[ex_idx] = updated_example
        all_judgment_strings.append(judgment_str)
        for results_one in results_pack:
            _accumulate_stats({'results': results_one, 'judgments': []}, counters)

    # Write summary statistics to a txt file as specified
    
    stats_txt_file = "judge_statistics_1.txt"
    mode = "a" if os.path.exists(stats_txt_file) else "w"
    with open(stats_txt_file, mode, encoding="utf-8") as statout:
        statout.write(f"{os.path.basename(input_file)}\n")
        statout.write(f"total: {counters['total']}\n")
        statout.write(f"empty: {counters['empty']}\n")
        statout.write(f"expression: {counters['expression']}\n")
        statout.write(f"correct: {counters['correct']}\n")
        statout.write(f"identify: {counters['identified']}\n")
        statout.write("\n")

    with open(input_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # judgments_txt_file = input_file.replace('.jsonl', "_judgments.txt")
    # mode = "a" if os.path.exists(judgments_txt_file) else "w"

    # with open(judgments_txt_file, mode, encoding='utf-8') as txtout:
    #     for s in all_judgment_strings:
    #         txtout.write(s.strip() + "\n\n")


# -----------------------
# CLI wiring
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Trap Problem Judge - Parallel")
    parser.add_argument('--input_file', type=str, required=True, help="Input JSONL with trap problems and responses")
    # separate creds/models for final vs process
    parser.add_argument('--api_key_final', type=str, required=True, help="API key for Final Answer judge")
    parser.add_argument('--model_final', type=str, required=True, help="Model for Final Answer judge")
    parser.add_argument('--api_key_process', type=str, required=True, help="API key for Identification/Modification judge")
    parser.add_argument('--model_process', type=str, required=True, help="Model for Identification/Modification judge")
    parser.add_argument('--url_final', type=str, help="Optional API base URL (used for the *final* judge if provided)")
    parser.add_argument('--url_process', type=str, help="Optional API base URL (used for the *process* judge if provided)")
    parser.add_argument('--template_dir', default='prompt', help="Directory containing judge templates")
    parser.add_argument('--index_range', type=str, help="Range 'a:b', list '1,4,6', single '7', or empty=all")
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--retry_delay', type=float, default=1.0)
    parser.add_argument('--mode', choices=['async', 'threaded'], default='async', help="Concurrency backend")
    # async controls
    parser.add_argument('--concurrency', type=int, default=8, help="Max in-flight calls (async)")
    parser.add_argument('--rpm', type=int, help="Requests per minute cap (async)")
    # threaded controls
    parser.add_argument('--workers', type=int, default=8, help="Max worker threads (threaded)")
    return parser.parse_args()


def main():
    args = parse_args()
    judge = TrapProblemJudge(template_dir=args.template_dir)

    if args.mode == 'threaded':
        # Build sync callers
        llm_call_func_final = create_llm_call_function_sync(
            api_key=args.api_key_final, model=args.model_final, url=args.url_final,
            max_retries=args.max_retries, retry_delay=args.retry_delay
        )
        llm_call_func_process = create_llm_call_function_sync(
            api_key=args.api_key_process, model=args.model_process, url=args.url_process,
            max_retries=args.max_retries, retry_delay=args.retry_delay
        )
        print(f"[threaded] Processing {args.input_file} with {args.workers} workers...")
        process_jsonl_file_threaded(
            judge, args.input_file, llm_call_func_final, llm_call_func_process,
            args.index_range, workers=args.workers
        )
        print(f"Results saved to {args.input_file} (updated in-place)")
        print("Assessment completed successfully!")
        return

    # async mode
    llm_call_func_final_async = create_llm_call_function_async(
        api_key=args.api_key_final, model=args.model_final, url=args.url_final,
        max_retries=args.max_retries, retry_delay=args.retry_delay,
        concurrency=args.concurrency, rpm=args.rpm
    )
    llm_call_func_process_async = create_llm_call_function_async(
        api_key=args.api_key_process, model=args.model_process, url=args.url_process,
        max_retries=args.max_retries, retry_delay=args.retry_delay,
        concurrency=args.concurrency, rpm=args.rpm
    )

    print(f"[async] Processing {args.input_file} with concurrency={args.concurrency}, rpm={args.rpm or 'None'} ...")
    import asyncio
    asyncio.run(process_jsonl_file_async(
        judge, args.input_file, llm_call_func_final_async, llm_call_func_process_async, args.index_range
    ))
    print(f"Results saved to {args.input_file} (updated in-place)")
    print("Assessment completed successfully!")


if __name__ == "__main__":
    main()
