# LLM-CVD

`llm_cvd`는 Juliet/Real_Vul 형식 데이터를 이용해 LLM 기반 취약점 분류 실험을 실행하고 분석하는 폴더입니다.
CodeBERT 임베딩과 FAISS 인덱스로 RAG few-shot 예시를 검색하고, ChatGPT, Claude, Gemini, Grok API 평가 결과를 저장합니다.

## 가능한 기능

- Juliet 또는 Real_Vul 스타일 CSV를 평가용/검색용 데이터로 로드
- CodeBERT 임베딩과 FAISS 기반 RAG 인덱스 생성
- RAG few-shot 취약점 분류 평가 실행
- Zero-shot 기준선 평가 실행
- ChatGPT, Claude, Gemini, Grok Batch API 평가 실행
- 결과 CSV/JSONL 저장, 재개 실행, 비용 추정용 토큰 정보 기록
- RAG 검색 결과 점검, 정성 평가용 CSV/HTML 생성, t-SNE 시각화, 반복 실험 결과 요약

## 폴더 구성

| 경로 | 설명 |
| --- | --- |
| `data/` | Juliet/Real_Vul 데이터 로더입니다. |
| `retrieval/` | CodeBERT 임베딩, FAISS 검색, RAG 인덱스 생성 로직입니다. |
| `evaluation/` | zero-shot/RAG API 평가 실행 로직과 결과 저장, 로그, resume 공통 유틸입니다. |
| `llm/` | LLM 제공자별 API 호출 래퍼입니다. |
| `prompts/` | 프롬프트 템플릿과 응답 라벨 파서입니다. |
| `batch_api_script/` | 제공자별 Batch API 실행 스크립트와 상세 README가 있습니다. |
| `analysis/` | 검색 점검, 정성 평가, 시각화, 결과 요약 스크립트와 상세 README가 있습니다. |
| `results/` | 실험 결과 CSV/JSONL, 리뷰 파일, 시각화 산출물이 저장됩니다. |
| `indexes/` | FAISS 인덱스와 메타데이터가 저장됩니다. |
| `cache/` | 데이터 로딩/전처리 캐시가 저장됩니다. |
| `pricing_config.json` | 비용 추정용 제공자별 토큰 가격 설정 파일입니다. |

## 세부 문서

- Batch API 실행법: [`batch_api_script/README.md`](batch_api_script/README.md)
- 분석 스크립트 사용법: [`analysis/README.md`](analysis/README.md)

## 기본 전제

프로젝트 루트의 `requirements.txt`를 설치한 Python 환경에서 실행합니다.
API 키와 모델명은 `llm_cvd/.env` 또는 쉘 환경변수로 설정합니다.

주요 기본 데이터 경로는 다음을 기준으로 합니다.

- RAG 학습 CSV: `../dataset/juliet_Real_Vul_data.csv`
- 평가 대상 CSV: `../dataset/cve_Real_Vul_data.csv`

## 동작 확인 범위

리팩터링 후 다음 수준까지 확인했습니다.

- `python3 -m compileall -q llm_cvd dataset/split_juliet_real_vul_by_dataset_type.py`
- zero-shot, few-shot RAG, RAG 인덱스 생성 모듈의 `--help` 실행
- Batch API 스크립트와 analysis 스크립트의 `--help` 실행

위 확인은 프로젝트 가상환경인 `.venv`의 Python으로 수행했습니다.
새 환경에서 실행할 때는 프로젝트 루트에서 `requirements.txt`를 설치해야 합니다.

```bash
python3 -m pip install -r requirements.txt
```

## RAG DB 구축

RAG few-shot 평가 전에 CodeBERT 임베딩과 FAISS 인덱스를 생성합니다.

```bash
python3 -m llm_cvd.retrieval.build_index \
  --rag-dataset-root dataset/juliet_Real_Vul_data.csv \
  --cache-dir llm_cvd/cache \
  --index-dir llm_cvd/indexes \
  --index-name juliet_Real_Vul_data_codebert \
  --index-batch-size 16
```

인덱스를 강제로 다시 만들려면 `--rebuild-index`를 추가합니다.
데이터 로딩 캐시까지 다시 만들려면 `--rebuild-cache`도 함께 추가합니다.

## Zero-Shot 평가

검색 예시 없이 target CSV의 코드만 모델에 전달해 평가합니다.

```bash
python3 -m llm_cvd.evaluation.baseline_api \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --providers chatgpt,claude,gemini,grok \
  --output-dir llm_cvd/results \
  --run-name cve_zeroshot \
  --repeats 1 \
  --max-workers 4 \
  --env-file llm_cvd/.env
```

일부 샘플만 테스트하려면 `--start 0 --limit 5`처럼 범위를 제한합니다.

### FreeLLMAPI로 GPT-4o 평가

FreeLLMAPI는 OpenAI-compatible endpoint이므로 `freellm` provider로 사용할 수 있습니다.
먼저 FreeLLMAPI dashboard에서 unified API key를 발급한 뒤 `llm_cvd/.env`에 설정합니다.

```dotenv
FREELLMAPI_BASE_URL=http://localhost:3001/v1
FREELLMAPI_API_KEY=freellmapi-...
FREELLMAPI_MODEL=gpt-4o
```

Zero-shot smoke test는 다음처럼 실행합니다.

```bash
python3 -m llm_cvd.evaluation.baseline_api \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --providers freellm \
  --models freellm=gpt-4o \
  --limit 1 \
  --max-workers 1 \
  --output-dir llm_cvd/results \
  --run-name cve_zeroshot_freellm_gpt4o_smoke \
  --env-file llm_cvd/.env
```

FreeLLMAPI가 실제로 어떤 upstream provider/model로 라우팅했는지는 결과 파일의
`routed_provider`, `routed_model`, `fallback_attempts` 컬럼에 저장됩니다.
Dashboard 또는 `/v1/models`에서 GPT-4o 모델 ID가 다르게 보이면
`--models freellm=<실제 모델 ID>`로 바꿔 실행합니다.

## Few-Shot RAG 평가

```bash
python3 -m llm_cvd.evaluation.rag_api \
  --rag-dataset-root dataset/juliet_Real_Vul_data.csv \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --cache-dir llm_cvd/cache \
  --index-dir llm_cvd/indexes \
  --index-name juliet_Real_Vul_data_codebert \
  --providers chatgpt,claude,gemini,grok \
  --k 6 \
  --output-dir llm_cvd/results \
  --run-name cve_k6 \
  --repeats 1 \
  --max-workers 4 \
  --env-file llm_cvd/.env
```

실패한 API 호출만 다시 시도하려면 기존 결과 파일을 지정합니다.

```bash
python3 -m llm_cvd.evaluation.rag_api \
  --retry-errors-from llm_cvd/results/cve_k6.jsonl \
  --rag-dataset-root dataset/juliet_Real_Vul_data.csv \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --cache-dir llm_cvd/cache \
  --index-dir llm_cvd/indexes \
  --index-name juliet_Real_Vul_data_codebert \
  --output-dir llm_cvd/results \
  --run-name cve_k6_retry
```

## 분석 스크립트

결과 요약 CSV를 생성합니다.

```bash
python3 llm_cvd/analysis/analyze_results.py llm_cvd/results/cve_k6.jsonl \
  --pricing llm_cvd/pricing_config.json \
  --per-repeat-output llm_cvd/results/cve_k6_by_repeat.csv \
  --output llm_cvd/results/cve_k6_summary.csv
```

검색된 top-k 예시를 CSV로 점검합니다.

```bash
python3 llm_cvd/analysis/inspect_rag_retrieval.py \
  --rag-dataset-root dataset/juliet_Real_Vul_data.csv \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --cache-dir llm_cvd/cache \
  --index-dir llm_cvd/indexes \
  --index-name juliet_Real_Vul_data_codebert \
  --k 6 \
  --output-csv llm_cvd/results/rag_retrieval_inspection_cve_k6.csv
```

정성 평가용 CSV/HTML을 생성합니다.

```bash
python3 llm_cvd/analysis/prepare_rag_qualitative_review.py \
  --input-csv llm_cvd/results/rag_retrieval_inspection_cve_k6.csv \
  --output-csv llm_cvd/results/rag_retrieval_qualitative_review_cve_k6.csv \
  --output-html llm_cvd/results/rag_retrieval_qualitative_review_cve_k6.html
```

Excel 검토용 파일과 query별 요약을 생성합니다.

```bash
python3 llm_cvd/analysis/prepare_rag_excel_review.py \
  --review-csv llm_cvd/results/rag_retrieval_qualitative_review_cve_k6.csv \
  --inspection-csv llm_cvd/results/rag_retrieval_inspection_cve_k6.csv \
  --output-csv llm_cvd/results/rag_retrieval_qualitative_review_cve_k6_excel_review.csv \
  --summary-csv llm_cvd/results/rag_retrieval_qualitative_review_cve_k6_excel_summary.csv
```

t-SNE 시각화를 생성합니다.

```bash
python3 llm_cvd/analysis/visualize_rag_tsne.py \
  --inspection-csv llm_cvd/results/rag_retrieval_inspection_cve_k6.csv \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --rag-dataset-root dataset/juliet_Real_Vul_data.csv \
  --index-dir llm_cvd/indexes \
  --index-name juliet_Real_Vul_data_codebert \
  --output-csv llm_cvd/results/rag_tsne_cve_k6_points.csv \
  --output-html llm_cvd/results/rag_tsne_cve_k6.html \
  --output-png llm_cvd/results/rag_tsne_cve_k6.png
```
