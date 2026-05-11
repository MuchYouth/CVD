# Batch API 실행 방법

이 폴더에는 CVD 평가용 Batch API 스크립트가 제공자별로 분리되어 있습니다.
명령은 `llm_cvd` 디렉터리에서 실행하는 것을 기준으로 합니다.

```bash
cd /home/dayoung/CVD/llm_cvd
```

## 공통 준비

`.env` 파일 또는 쉘 환경변수에 사용할 API 키를 설정합니다.

```dotenv
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...
```

모델은 `--model`로 직접 지정할 수 있고, 생략하면 아래 환경변수 또는 스크립트 기본값을 사용합니다.

| 제공자 | 스크립트 | 모델 환경변수 | 기본 모델 | 필요 패키지 |
| --- | --- | --- | --- | --- |
| ChatGPT/OpenAI | `batch_api_script/batch_chatgpt_api_eval.py` | `OPENAI_MODEL` | `gpt-4o` | `openai` |
| Claude/Anthropic | `batch_api_script/batch_claude_api_eval.py` | `CLAUDE_MODEL` | `claude-sonnet-4-5-20250929` | `anthropic` |
| Gemini/Google | `batch_api_script/batch_gemini_api_eval.py` | `GEMINI_MODEL` | `gemini-2.5-flash` | `google-genai` |
| Grok/xAI | `batch_api_script/batch_grok_api_eval.py` | `GROK_MODEL` | `grok-4.3` | `xai-sdk` |

## 기본 실행 패턴

한 번에 준비, 제출, 상태 확인, 결과 수집까지 하려면 `run`을 사용합니다.

```bash
../.venv/bin/python batch_api_script/<script>.py run \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --run-name <run-name> \
  --repeats 3 \
  --output-dir results
```

단계를 나눠서 실행하려면 아래 순서를 사용합니다. 긴 배치 작업은 이 방식이 중간 상태를 확인하기 좋습니다.

```bash
../.venv/bin/python batch_api_script/<script>.py prepare --run-name <run-name> [dataset options]
../.venv/bin/python batch_api_script/<script>.py submit  --run-name <run-name>
../.venv/bin/python batch_api_script/<script>.py status  --run-name <run-name>
../.venv/bin/python batch_api_script/<script>.py collect --run-name <run-name>
```

`prepare`에서 만든 요청 파일, 매니페스트, job metadata, 최종 결과는 기본적으로 `results/` 아래에 저장됩니다.

## 모델별 명령어

### ChatGPT/OpenAI

```bash
../.venv/bin/python batch_api_script/batch_chatgpt_api_eval.py run \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --model gpt-4o \
  --run-name juliet-real_chatgpt_batch \
  --repeats 3 \
  --output-dir results
```

단계별 실행:

```bash
../.venv/bin/python batch_api_script/batch_chatgpt_api_eval.py prepare \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --model gpt-4o \
  --run-name juliet-real_chatgpt_batch \
  --repeats 3 \
  --output-dir results

../.venv/bin/python batch_api_script/batch_chatgpt_api_eval.py submit --run-name juliet-real_chatgpt_batch --output-dir results
../.venv/bin/python batch_api_script/batch_chatgpt_api_eval.py status --run-name juliet-real_chatgpt_batch --output-dir results
../.venv/bin/python batch_api_script/batch_chatgpt_api_eval.py collect --run-name juliet-real_chatgpt_batch --output-dir results
```

### Claude/Anthropic

```bash
../.venv/bin/python batch_api_script/batch_claude_api_eval.py run \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --model claude-sonnet-4-5-20250929 \
  --run-name juliet-real_claude_batch \
  --repeats 3 \
  --output-dir results
```

단계별 실행:

```bash
../.venv/bin/python batch_api_script/batch_claude_api_eval.py prepare \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --model claude-sonnet-4-5-20250929 \
  --run-name juliet-real_claude_batch \
  --repeats 3 \
  --output-dir results

../.venv/bin/python batch_api_script/batch_claude_api_eval.py submit --run-name juliet-real_claude_batch --output-dir results
../.venv/bin/python batch_api_script/batch_claude_api_eval.py status --run-name juliet-real_claude_batch --output-dir results
../.venv/bin/python batch_api_script/batch_claude_api_eval.py collect --run-name juliet-real_claude_batch --output-dir results
```

### Gemini/Google

```bash
../.venv/bin/python batch_api_script/batch_gemini_api_eval.py run \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --model gemini-2.5-flash \
  --thinking-budget 0 \
  --run-name juliet-real_gemini_batch \
  --repeats 3 \
  --output-dir results
```

Gemini 2.5 계열은 `--thinking-budget`을 사용할 수 있습니다. `-1`이면 `thinkingConfig`를 생략하고, 기본값 `0`은 2.5 Flash에는 적용되지만 2.5 Pro에서는 생략됩니다.

단계별 실행:

```bash
../.venv/bin/python batch_api_script/batch_gemini_api_eval.py prepare \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --model gemini-2.5-flash \
  --thinking-budget 0 \
  --run-name juliet-real_gemini_batch \
  --repeats 3 \
  --output-dir results

../.venv/bin/python batch_api_script/batch_gemini_api_eval.py submit --run-name juliet-real_gemini_batch --output-dir results
../.venv/bin/python batch_api_script/batch_gemini_api_eval.py status --run-name juliet-real_gemini_batch --output-dir results
../.venv/bin/python batch_api_script/batch_gemini_api_eval.py collect --run-name juliet-real_gemini_batch --output-dir results
```

### Grok/xAI

```bash
../.venv/bin/python batch_api_script/batch_grok_api_eval.py run \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --model grok-4.3 \
  --submit-mode inline \
  --run-name juliet-real_grok_batch \
  --repeats 3 \
  --output-dir results
```

Grok은 기본 제출 방식이 `--submit-mode inline`입니다. 필요하면 `--submit-mode file`로 JSONL 업로드 기반 제출을 사용할 수 있습니다.

단계별 실행:

```bash
../.venv/bin/python batch_api_script/batch_grok_api_eval.py prepare \
  --db-name juliet-real \
  --prompt-mode zero-shot \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --model grok-4.3 \
  --run-name juliet-real_grok_batch \
  --repeats 3 \
  --output-dir results

../.venv/bin/python batch_api_script/batch_grok_api_eval.py submit --run-name juliet-real_grok_batch --output-dir results --submit-mode inline
../.venv/bin/python batch_api_script/batch_grok_api_eval.py status --run-name juliet-real_grok_batch --output-dir results
../.venv/bin/python batch_api_script/batch_grok_api_eval.py collect --run-name juliet-real_grok_batch --output-dir results
```

## RAG few-shot으로 실행

`--prompt-mode rag`를 쓰면 CodeBERT/FAISS로 top-k 예시를 검색해서 few-shot 프롬프트를 만든 뒤 Batch API에 제출합니다.

```bash
../.venv/bin/python batch_api_script/batch_chatgpt_api_eval.py run \
  --db-name juliet-real \
  --prompt-mode rag \
  --rag-dataset-root ../dataset/juliet_Real_Vul_data.csv \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --k 6 \
  --model gpt-4o \
  --run-name juliet-real_k6_chatgpt_batch \
  --repeats 3 \
  --output-dir results
```

위 예시에서 스크립트 파일과 `--model`, `--run-name`만 바꾸면 Claude, Gemini, Grok에도 같은 방식으로 적용할 수 있습니다.

## 자주 쓰는 옵션

| 옵션 | 설명 |
| --- | --- |
| `--prompt-mode zero-shot` | 검색 예시 없이 대상 코드만 분류합니다. 결과의 `k`는 `0`입니다. |
| `--prompt-mode rag` | `--rag-dataset-root`에서 top-k 예시를 검색해 few-shot 프롬프트를 만듭니다. |
| `--target-dataset-csv` | 평가할 대상 CSV입니다. 스크립트의 이전 별칭 `--rag-test-csv`, `--real-vul-csv`도 같은 인자로 동작합니다. |
| `--rag-dataset-root` | RAG 검색용 학습 데이터입니다. Juliet raw 디렉터리 또는 Real_Vul 스타일 CSV를 사용할 수 있습니다. |
| `--k` | RAG에서 검색할 예시 개수입니다. 기본값은 `6`입니다. |
| `--start`, `--limit` | 일부 샘플만 실행할 때 사용합니다. 먼저 `--limit 2`처럼 작은 값으로 확인하면 좋습니다. |
| `--repeats` | 같은 샘플을 반복 호출하는 횟수입니다. 기본값은 `1`입니다. |
| `--run-name` | 결과 파일 이름을 결정합니다. 생략하면 제공자와 prompt mode 기반 이름이 자동 생성됩니다. |
| `--output-dir` | 결과 저장 위치입니다. 기본값은 `results`입니다. |
| `--poll-interval-sec` | `run`에서 상태를 다시 확인하는 간격입니다. 기본값은 `60`초입니다. |
| `--max-output-tokens` | 최대 출력 토큰 수입니다. ChatGPT/Claude 기본값은 `32`, Gemini/Grok 기본값은 `64`입니다. |
| `--temperature` | 기본값은 `0.0`입니다. |

## 결과 파일

`--run-name juliet-real_chatgpt_batch --output-dir results` 기준으로 주요 파일은 다음과 같이 생성됩니다.

```text
results/batch_chatgpt_juliet-real_chatgpt_batch_requests.jsonl
results/batch_chatgpt_juliet-real_chatgpt_batch_manifest.jsonl
results/batch_chatgpt_juliet-real_chatgpt_batch_job.json
results/batch_chatgpt_juliet-real_chatgpt_batch_raw_results.jsonl
results/juliet-real_chatgpt_batch.jsonl
results/juliet-real_chatgpt_batch.csv
```

최종 분석에는 보통 `<run-name>.jsonl` 또는 `<run-name>.csv`를 사용하면 됩니다.
