import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI
import logging
from dataclasses import dataclass
from trap_judge_parallel import TrapProblemJudge, create_llm_call_function_async


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Keep framework logs visible while hiding verbose HTTP client request logs.
for noisy_logger_name in ("httpx", "httpcore", "openai", "openai._base_client"):
    logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

@dataclass
class ProblemResult:
    """Data class to store results of problem transformation"""
    original_problem: str
    modified_problem: str
    annotation: str
    verification_result: bool
    confusion_result: bool

class MultiAgentFramework:
    """Framework for transforming well-defined problems into ill-defined ones"""
    
    def __init__(self, api_key: str, prompt_dir: str = "prompt"):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.modify_model = os.getenv("MODIFY_MODEL", "google/gemini-3-pro-preview")
        self.verify_model = os.getenv("VERIFY_MODEL", self.modify_model)
        self.test_model = os.getenv("TEST_MODEL", "openai/gpt-4o-mini")
        self.test_model_max_retries = int(os.getenv("TEST_MODEL_MAX_RETRIES", "6"))
        self.test_model_retry_base_delay = float(
            os.getenv("TEST_MODEL_RETRY_BASE_DELAY", "2.0")
        )
        self.modify_attempts = max(1, int(os.getenv("MODIFY_ATTEMPTS", "3")))
        self.prompt_dir = Path(prompt_dir)
        self.verify_raw_log_file = Path("logs/gemini_verify_raw.jsonl")
        self._verify_log_lock = asyncio.Lock()
        self.prompts = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompt templates from the prompt directory"""
        prompt_files = {
            'modify': 'modify_problem.tmpl',
            'verify': 'verify_problem.tmpl',
            'test_gpt': 'test_gpt.tmpl',
        }
        
        for key, filename in prompt_files.items():
            filepath = self.prompt_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.prompts[key] = f.read()
                logger.info(f"Loaded prompt template: {filename}")
            else:
                logger.warning(f"Prompt template not found: {filename}")

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove one outer markdown code fence if present."""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    @staticmethod
    def _parse_ill_defined_verdict(response: str) -> Optional[bool]:
        """
        Parse Gemini verification output into a boolean verdict.
        Returns None if no reliable verdict is found.
        """
        cleaned = MultiAgentFramework._strip_code_fence(response)
        response_lower = cleaned.lower()

        # JSON-style outputs, e.g. {"ill_defined": true} or {"ill-defined": false}
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                for key in ("ill-defined", "ill_defined", "is_ill_defined", "ill defined"):
                    if key in data and isinstance(data[key], bool):
                        return data[key]
        except json.JSONDecodeError:
            pass

        # Label-based outputs (preferred by prompt).
        for pattern in [
            r"ill[-_\s]?defined\s*[:=]\s*(?:\*\*|__|`)?\s*(true|false|yes|no)\b",
            r"answer\s*:\s*(?:\*\*|__|`)?\s*(yes|no)\b",
        ]:
            matches = re.findall(pattern, response_lower)
            if matches:
                return matches[-1] in {"true", "yes"}

        # Final fallback with explicit negative check first.
        if re.search(r"\b(is not|isn't|not)\s+(a\s+)?ill[-_\s]?defined\b", response_lower):
            return False
        if re.search(r"\bill[-_\s]?defined\b", response_lower):
            return True

        return None

    async def _log_verify_raw_response(
        self,
        modified_problem: str,
        response: str,
        verdict: Optional[bool]
    ) -> None:
        """Persist raw Gemini verification outputs for offline debugging."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modified_problem": modified_problem,
            "response": response,
            "parsed_verdict": verdict
        }
        try:
            self.verify_raw_log_file.parent.mkdir(parents=True, exist_ok=True)
            async with self._verify_log_lock:
                with open(self.verify_raw_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning("Failed to write Gemini verify raw log: %s", e)
    
    async def call_gemini(
        self,
        messages: List[Dict],
        use_reasoning: bool = True,
        model: Optional[str] = None
    ) -> str:
        """Call Gemini 3 API with optional reasoning"""
        try:
            extra_body = {"reasoning": {"enabled": True}} if use_reasoning else {}
            model_name = model or self.modify_model
            
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body=extra_body
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error calling model %s: %s", model_name, e)
            raise
    
    async def call_gpt4o(self, messages: List[Dict]) -> str:
        """Call the confusion-test model with retry/backoff on transient failures."""
        for attempt in range(self.test_model_max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.test_model,
                    messages=messages
                )
                if not response.choices:
                    raise RuntimeError("Empty choices in test model response.")
                return response.choices[0].message.content
            except Exception as e:
                message = str(e).lower()
                status_code = getattr(e, "status_code", None)
                is_rate_limit = (
                    status_code == 429
                    or "429" in message
                    or "rate limit" in message
                    or "temporarily rate-limited" in message
                )
                is_last_attempt = attempt >= self.test_model_max_retries - 1
                if is_rate_limit and not is_last_attempt:
                    delay = self.test_model_retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate-limited on model %s; retrying in %.1fs (%d/%d)",
                        self.test_model,
                        delay,
                        attempt + 1,
                        self.test_model_max_retries
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Error calling confusion-test model %s: %s", self.test_model, e)
                raise
    
    async def modify_problem(
        self,
        original_problem: str,
        attempt_idx: int = 0
    ) -> Tuple[Optional[str], Optional[str]]:
        """Step 2: Modify well-defined problem into ill-defined using Gemini"""
        try:
            prompt = self.prompts['modify'].format(problem=original_problem)
            if attempt_idx > 0:
                prompt += (
                    "\n\nPrevious candidate was rejected. Generate a different mutation strategy "
                    "than before (e.g., remove a different necessary condition, or add a "
                    "different contradiction) while keeping the math topic close to the original."
                )
            messages = [{"role": "user", "content": prompt}]
            response = await self.call_gemini(
                messages,
                use_reasoning=True,
                model=self.modify_model
            )
            
            # Parse response to extract modified problem and annotation
            # Expected format: JSON with 'modified_problem' and 'annotation' keys
            try:
                result = json.loads(response)
                return result.get('modified_problem'), result.get('annotation')
            except json.JSONDecodeError:
                # If not JSON, try to parse structured text
                lines = response.strip().split('\n')
                modified_problem = None
                annotation = None
                for i, line in enumerate(lines):
                    if 'modified_problem' in line.lower():
                        modified_problem = lines[i+1] if i+1 < len(lines) else None
                    if 'annotation' in line.lower() or 'why' in line.lower():
                        annotation = lines[i+1] if i+1 < len(lines) else None
                
                return modified_problem, annotation
        except Exception as e:
            logger.error(f"Error modifying problem: {e}")
            return None, None
    
    async def verify_ill_defined(self, modified_problem: str) -> bool:
        """Step 3: Verify if the modified problem is truly ill-defined using Gemini"""
        try:
            prompt = self.prompts['verify'].format(problem=modified_problem)
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.call_gemini(
                messages,
                use_reasoning=True,
                model=self.verify_model
            )
            verdict = self._parse_ill_defined_verdict(response)
            await self._log_verify_raw_response(modified_problem, response, verdict)
            if verdict is None:
                logger.warning(
                    "Could not parse verification verdict from Gemini response. "
                    "Defaulting to False. Response preview: %s",
                    response.replace("\n", " ")[:300]
                )
                return False
            logger.info("Gemini verification verdict: %s", verdict)
            return verdict
        except Exception as e:
            logger.error(f"Error verifying ill-defined problem: {e}")
            return False
    async def test_confusion_gpt4o(
        self, 
        modified_problem: str,
        annotation: str,
        trap_judge: TrapProblemJudge,
        llm_call_func_final_async,
        llm_call_func_process_async
    ) -> Tuple[bool, Optional[str]]:
        """
        Step 4: Test if the problem confuses GPT-4o using trap_judge
        
        Args:
            modified_problem: The ill-defined problem
            annotation: Annotation explaining why it's ill-defined
            trap_judge: TrapProblemJudge instance
            judge_llm_call_func_async: Async LLM caller for judge
            
        Returns:
            Tuple of (is_confused, gpt4o_response)
        """
        try:
            # Get GPT-4o response
            prompt_template = self.prompts.get('test_gpt', '{problem}')
            prompt = prompt_template.format(problem=modified_problem)
            messages = [{"role": "user", "content": prompt}]
            
            gpt4o_response = await self.call_gpt4o(messages)
            
            # Use trap_judge to evaluate confusion
            result = await trap_judge.judge_response_async(
                question=modified_problem,
                response=gpt4o_response,
                annotation=annotation,
                llm_call_func_final_async=llm_call_func_final_async,
                llm_call_func_process_async=llm_call_func_process_async
            )
            
            # Check if response is empty
            if result['results'][0] == "empty":
                logger.warning("Empty response from GPT-4o")
                return False, gpt4o_response
            
            # Model shows confusion if any of the three judgments are True:
            # 1. Says the problem is insolvable (results[0])
            # 2. Identifies the trap (results[1])
            # 3. Attempts modification (results[2])
            is_confused = any([
                results if isinstance(results, bool) else False
                for results in result['results']
            ])
            
            if is_confused:
                logger.info(f"GPT-4o showed confusion. Results: {result['results']}")
            
            return is_confused, gpt4o_response
            
        except Exception as e:
            logger.error(f"Error testing GPT-4o confusion: {e}")
            return False, None
    
    async def process_single_problem(
        self, 
        problem_data: Dict,
        trap_judge: TrapProblemJudge,
        llm_call_func_final_async,
        llm_call_func_process_async
    ) -> Optional[ProblemResult]:
        """Process a single problem through the entire pipeline"""
        original_problem = problem_data.get('problem', '')
        logger.info(f"Processing problem: {original_problem[:50]}...")
        
        for attempt_idx in range(self.modify_attempts):
            # Step 2: Modify problem
            modified_problem, annotation = await self.modify_problem(
                original_problem,
                attempt_idx=attempt_idx
            )
            if not modified_problem or not annotation:
                logger.warning("Failed to modify problem (attempt %d)", attempt_idx + 1)
                continue

            # Step 3: Verify it's ill-defined
            is_ill_defined = await self.verify_ill_defined(modified_problem)
            if not is_ill_defined:
                logger.info(
                    "Problem not verified as ill-defined (attempt %d)",
                    attempt_idx + 1
                )
                continue

            # Step 4: Test if it confuses GPT-4o
            is_confusing, gpt4o_response = await self.test_confusion_gpt4o(
                modified_problem,
                annotation,
                trap_judge,
                llm_call_func_final_async,
                llm_call_func_process_async
            )
            if not is_confusing:
                logger.info("Problem does not confuse GPT-4o (attempt %d)", attempt_idx + 1)
                continue

            # Step 5: Return successful result
            logger.info("Problem successfully processed on attempt %d!", attempt_idx + 1)
            return ProblemResult(
                original_problem=original_problem,
                modified_problem=modified_problem,
                annotation=annotation,
                verification_result=is_ill_defined,
                confusion_result=is_confusing
            )

        return None
    
    async def process_problems_batch(
        self,
        input_file: str,
        output_file: str,
        trap_judge: TrapProblemJudge,
        llm_call_func_final_async,
        llm_call_func_process_async,
        max_concurrent: int = 5
    ):
        """Process problems in parallel with concurrency control"""
        # Step 1: Read original problems
        problems = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        logger.info(f"Loaded {len(problems)} problems from {input_file}")
        if not problems:
            logger.warning("No input problems found in %s", input_file)
            self._save_results([], output_file)
            return

        print(problems[0])
        # Process problems with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(problem):
            async with semaphore:
                return await self.process_single_problem(
                    problem, trap_judge, llm_call_func_final_async,
                    llm_call_func_process_async
                )
        
        # Process all problems in parallel
        tasks = [process_with_semaphore(p) for p in problems]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results and save
        successful_results = []
        for result in results:
            if isinstance(result, ProblemResult):
                successful_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error processing problem: {result}")
        
        # Step 5: Save qualified problems
        self._save_results(successful_results, output_file)
        
        logger.info(f"Processed {len(problems)} problems, "
                   f"saved {len(successful_results)} qualified ill-defined problems")
    
    def _save_results(self, results: List[ProblemResult], output_file: str):
        """Save results to JSONL file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', 
                    exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                output_data = {
                    'original_problem': result.original_problem,
                    'modified_problem': result.modified_problem,
                    'annotation': result.annotation
                }
                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(results)} results to {output_file}")


async def main():
    """Main execution function"""
    
    # Configuration
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    JUDGE_API_KEY1 = os.getenv("JUDGE_API_KEY1") or os.getenv("XAI_API_KEY")
    JUDGE_API_KEY2 = os.getenv("JUDGE_API_KEY2") or JUDGE_API_KEY1
    INPUT_FILE = os.getenv("INPUT_FILE", "BeyondAIME_test.jsonl")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "ill_defined_problems.jsonl")
    PROMPT_DIR = os.getenv("PROMPT_DIR", "prompt")
    TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", "prompt")
    MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))
    JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "5"))
    JUDGE_RPM = int(os.getenv("JUDGE_RPM", "60"))
    JUDGE_MODEL_FINAL = os.getenv("JUDGE_MODEL_FINAL", "grok-4-fast-reasoning")
    JUDGE_MODEL_PROCESS = os.getenv("JUDGE_MODEL_PROCESS", "grok-4-fast-reasoning")
    JUDGE_URL = os.getenv("JUDGE_URL", "https://api.x.ai/v1")

    if not API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY for problem transformation calls.")
    if not JUDGE_API_KEY1:
        raise ValueError(
            "Missing judge API key. Set JUDGE_API_KEY1 or XAI_API_KEY environment variable."
        )
    
    # Initialize framework
    framework = MultiAgentFramework(api_key=API_KEY, prompt_dir=PROMPT_DIR)
    
    # Initialize trap_judge directly
    trap_judge = TrapProblemJudge(template_dir=TEMPLATE_DIR)
    
    # Create async LLM caller for the judge
    judge_llm_call_func_final_async = create_llm_call_function_async(
        api_key=JUDGE_API_KEY1,
        model=JUDGE_MODEL_FINAL,
        url=JUDGE_URL,
        max_retries=3,
        retry_delay=1.0,
        concurrency=JUDGE_CONCURRENCY,
        rpm=JUDGE_RPM
    )
    judge_llm_call_func_process_async = create_llm_call_function_async(
        api_key=JUDGE_API_KEY2,
        model=JUDGE_MODEL_PROCESS,
        url=JUDGE_URL,
        max_retries=3,
        retry_delay=1.0,
        concurrency=JUDGE_CONCURRENCY,
        rpm=JUDGE_RPM
    )
    
    # Process problems
    await framework.process_problems_batch(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        trap_judge=trap_judge,
        llm_call_func_final_async = judge_llm_call_func_final_async,
        llm_call_func_process_async = judge_llm_call_func_process_async,
        max_concurrent=MAX_CONCURRENT
    )


if __name__ == "__main__":
    asyncio.run(main())
