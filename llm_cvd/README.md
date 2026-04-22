# LLM-CVD Juliet API RAG Experiment

This directory contains the API-based RAG few-shot evaluation pipeline.

## Files

| File | Purpose |
| --- | --- |
| `juliet_loader.py` | Loads Juliet train functions from `/home/dayoung/juliet-playground/juliet-test-suite-v1.3` and Real_Vul test samples from `/home/dayoung/juliet-playground/cases/Real_Vul_data.csv`. |
| `rag_retriever.py` | Builds or loads CodeBERT CLS embeddings and a FAISS index under this directory. |
| `prompting.py` | Shared system prompt, few-shot prompt template, and label parser. |
| `providers.py` | Unified ChatGPT, Claude, Gemini, and Grok API wrappers. |
| `run_rag_api_eval.py` | Main inference loop. It appends every provider result immediately to JSONL and CSV. |
| `analyze_results.py` | Computes Accuracy, Precision, Recall, F1, latency, token averages, and estimated cost. |
| `pricing_config.json` | Fill in provider/model prices per 1M input/output tokens for cost estimates. |

## Environment

Create and use the isolated virtual environment:

```bash
cd /home/dayoung/llm-cvd/LLM-CVD
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

Then create `LLM-CVD/.env` or export these variables:

```dotenv
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...

OPENAI_MODEL=gpt-4o
CLAUDE_MODEL=claude-3-5-sonnet-latest
GEMINI_MODEL=gemini-1.5-flash
GROK_MODEL=grok-beta
```

The RAG side uses `torch`, `transformers`, `faiss-cpu`, `numpy`, and `tqdm`.
Analysis uses `pandas` and `scikit-learn`.

## Data

The script uses only:

- RAG dataset root: `/home/dayoung/juliet-playground/juliet-test-suite-v1.3`
- RAG test CSV: `/home/dayoung/juliet-playground/cases/Real_Vul_data.csv`

Juliet train records are extracted at function level from C/C++ files whose
function signatures contain `good` or `bad`.

`Real_Vul_data.csv` is parsed with a custom parser because the `processed_func`
column can contain raw C/C++ quotes that break a normal CSV reader.

Observed local counts:

- Juliet train functions: `346775`
- Real_Vul test rows: `13`
- Real_Vul labels: `8 Vulnerable`, `5 Safe`

## Run

From this directory:

```bash
.venv/bin/python run_rag_api_eval.py \
  --db-name juliet-real \
  --rag-dataset-root /home/dayoung/juliet-playground/juliet-test-suite-v1.3 \
  --rag-test-csv /home/dayoung/juliet-playground/cases/Real_Vul_data.csv \
  --providers chatgpt,claude,gemini,grok \
  --k 6 \
  --output-dir results
```

Building CodeBERT embeddings for all Juliet train functions can take a while.
For a first pass, use a capped train sample:

```bash
.venv/bin/python run_rag_api_eval.py \
  --providers chatgpt \
  --k 6 \
  --max-train-samples 5000 \
  --limit 2
```

The run writes both `results/juliet-real_k6.jsonl` and
`results/juliet-real_k6.csv`. Each provider result is flushed and fsynced
immediately.

## Analyze

```bash
.venv/bin/python analyze_results.py results/juliet-real_k6.jsonl \
  --pricing pricing_config.json \
  --output results/juliet-real_k6_summary.csv
```
