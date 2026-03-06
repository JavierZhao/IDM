# IDM Problem Generation

## 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Generate model responses for existing problems

This uses `BeyondAIME_test.jsonl` in this repo and writes outputs to `./outputs`.

```bash
export OPENROUTER_API_KEY="your_key_here"
python3 call_generate_batch.py
```

Optional overrides:

```bash
export GEN_MODEL="openai/gpt-4o-mini"
export GEN_API_URL="https://openrouter.ai/api/v1"
```

## 3) Generate ill-defined problems with the multi-agent pipeline

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
export XAI_API_KEY="your_xai_key"
python3 prob_gen.py
```

Optional overrides:

```bash
export INPUT_FILE="BeyondAIME_test.jsonl"
export OUTPUT_FILE="ill_defined_problems.jsonl"
export TEST_MODEL="openai/gpt-4o-mini"
export TEST_MODEL_MAX_RETRIES="6"
export TEST_MODEL_RETRY_BASE_DELAY="2.0"
export MAX_CONCURRENT="1"
export JUDGE_MODEL_FINAL="grok-4-fast-reasoning"
export JUDGE_MODEL_PROCESS="grok-4-fast-reasoning"
export JUDGE_URL="https://api.x.ai/v1"
```

## Notes

- Prompt templates are expected in `./prompt`.
- `prob_gen.py` will fail fast if required API keys are missing.
