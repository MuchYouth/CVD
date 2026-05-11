# CVD

LLM을 이용한 코드 취약점 탐지(CVD, Code Vulnerability Detection) 실험용 프로젝트입니다.
Juliet/Real_Vul 계열 CSV 데이터셋을 기반으로 zero-shot baseline과 CodeBERT+FAISS RAG few-shot 실험을 수행하고, ChatGPT/Claude/Gemini/Grok API 결과를 저장/분석합니다.

## 디렉터리 역할

| 경로 | 역할 |
| --- | --- |
| `llm_cvd/` | 취약점 탐지 실험 파이프라인. 데이터 로드, 프롬프트 생성, RAG 검색, API 평가 실행, 결과 분석을 담당합니다. |
| `llm_api/` | ChatGPT, Claude, Gemini, Grok을 공통 인터페이스로 호출하기 위한 LLM 클라이언트 모듈입니다. |
| `dataset/` | RAG 구축 및 API 평가에 사용하는 Juliet/Real_Vul CSV 데이터셋과 분할 스크립트를 보관합니다. |
| `requirements.txt` | 실험 실행에 필요한 Python 패키지 목록입니다. |

## 주요 실행 파일

| 파일 | 역할 |
| --- | --- |
| `python3 -m llm_cvd.evaluation.baseline_api` | 검색 없이 target CSV만 사용해 zero-shot 취약점 탐지 API 평가를 실행합니다. 결과는 JSONL/CSV로 즉시 저장합니다. |
| `python3 -m llm_cvd.evaluation.rag_api` | Juliet/Real_Vul 학습 corpus에서 유사 코드를 검색한 뒤 few-shot RAG 프롬프트로 API 평가를 실행합니다. |
| `python3 -m llm_cvd.retrieval.build_index` | API 호출 없이 CodeBERT 임베딩과 FAISS 인덱스를 미리 생성하거나 갱신합니다. |
| `llm_cvd/analysis/analyze_results.py` | 평가 결과 JSONL/CSV를 읽어 Accuracy, Precision, Recall, F1, latency, token, cost 요약을 계산합니다. |
| `llm_cvd/analysis/inspect_rag_retrieval.py` | target 샘플별 top-k RAG 검색 결과를 CSV로 확인해 검색 품질을 점검합니다. |
| `llm_cvd/analysis/visualize_rag_tsne.py` | CVE query와 검색된 Juliet 예제 임베딩을 t-SNE로 시각화합니다. |
| `llm_cvd/analysis/prepare_rag_qualitative_review.py` | RAG 검색 결과를 정성 평가용 CSV/HTML 리뷰 자료로 변환합니다. |
| `llm_cvd/analysis/prepare_rag_excel_review.py` | Excel에서 보기 좋은 RAG top-k 정성 평가 시트를 생성합니다. |

## `llm_cvd/` 파일 역할

| 파일 | 역할 |
| --- | --- |
| `prompts/templates.py` | 시스템 프롬프트, zero-shot/RAG few-shot 프롬프트 템플릿, 모델 응답 라벨 파서를 정의합니다. |
| `llm/providers.py` | `llm_api` 클라이언트를 실험용 인터페이스로 감싸 provider/model 선택, 사용량 추적, 응답 표준화를 처리합니다. |
| `data/juliet_loader.py` | Juliet 원본 디렉터리 또는 Real_Vul 형식 CSV에서 학습/평가 레코드를 로드합니다. 코드 내 raw quote가 있는 CSV도 처리합니다. |
| `retrieval/rag_retriever.py` | CodeBERT CLS 임베딩을 만들고 FAISS 인덱스로 유사 취약점 예제를 검색합니다. |
| `evaluation/utils.py` | 결과 저장, resume 키 로드, CSV 헤더 생성, `.env` 로드, 로그/진행 표시 등 평가 공통 유틸입니다. |
| `pricing_config.json` | provider/model별 100만 토큰당 입력/출력 가격을 기록해 비용 추정에 사용합니다. |
| `__init__.py` | `llm_cvd` 패키지 초기화 파일입니다. |

## Batch API 스크립트

| 파일 | 역할 |
| --- | --- |
| `llm_cvd/batch_api_script/batch_chatgpt_api_eval.py` | OpenAI Batch API용 요청 JSONL 생성, 제출, 상태 확인, 결과 수집을 수행합니다. zero-shot/RAG 모드를 지원합니다. |
| `llm_cvd/batch_api_script/batch_claude_api_eval.py` | Anthropic Claude Message Batch API 평가 작업을 생성하고 수집합니다. |
| `llm_cvd/batch_api_script/batch_gemini_api_eval.py` | Gemini Batch API 평가 작업을 생성하고 수집합니다. zero-shot/RAG 모드를 지원합니다. |
| `llm_cvd/batch_api_script/batch_grok_api_eval.py` | xAI Grok Batch API 평가 작업을 생성하고 수집합니다. |
| `llm_cvd/batch_api_script/README.md` | Batch API 스크립트 사용법과 옵션 설명입니다. |

## `llm_api/` 파일 역할

| 파일 | 역할 |
| --- | --- |
| `base.py` | 모든 LLM 클라이언트가 구현해야 하는 `LLMClient` 추상 기본 클래스를 정의합니다. |
| `factory.py` | 환경변수 또는 인자로 지정한 provider에 맞는 LLM 클라이언트 인스턴스를 반환합니다. |
| `chatgpt.py` | OpenAI ChatGPT API 클라이언트 구현체입니다. |
| `claude.py` | Anthropic Claude API 클라이언트 구현체입니다. |
| `gemini.py` | Google Gemini API 클라이언트 구현체입니다. |
| `grok.py` | xAI Grok API 클라이언트 구현체입니다. |
| `README.md` | LLM 클라이언트 모듈 사용법과 환경변수 설명입니다. |

## `dataset/` 파일 역할

| 파일 | 역할 |
| --- | --- |
| `juliet_Real_Vul_data.csv` | RAG 지식베이스 구축에 사용하는 Juliet 기반 주요 취약점 corpus입니다. |
| `juliet_Real_Vul_train_val.csv` | `juliet_Real_Vul_data.csv`에서 train/validation 용도로 분리한 데이터입니다. |
| `juliet_Real_Vul_test_val.csv` | Juliet 기반 test/validation 용도 데이터입니다. |
| `juliet_Pair_Real_Vul_data.csv` | 취약/비취약 쌍을 비교하는 in-domain pairwise 평가 데이터셋입니다. |
| `cve_Real_Vul_data.csv` | 실제 CVE 및 오픈소스 프로젝트 기반 real-world 평가 데이터셋입니다. |
| `Real_Vul_data.csv` | Real_Vul 형식의 취약점 탐지 데이터셋입니다. |
| `Extended_Real_Vul_data_train.csv` | 확장 Real_Vul 계열 학습 데이터입니다. |
| `Extended_Real_Vul_data_test.csv` | 확장 Real_Vul 계열 테스트 데이터입니다. |
| `VulnPatchDS_Vul_data.csv` | VulnPatchDS 기반 API 테스트용 취약점 데이터셋입니다. |
| `split_juliet_real_vul_by_dataset_type.py` | `dataset_type` 컬럼 기준으로 `juliet_Real_Vul_data.csv`를 분할하는 스크립트입니다. |
| `README.md` | 데이터셋별 샘플 수, 라벨 분포, 실험 내 사용 구분을 설명합니다. |

## 기본 실행 흐름

1. `dataset/`의 CSV를 준비합니다.
2. RAG 실험을 할 경우 `python3 -m llm_cvd.retrieval.build_index`로 CodeBERT+FAISS 인덱스를 생성합니다.
3. `python3 -m llm_cvd.evaluation.baseline_api` 또는 `python3 -m llm_cvd.evaluation.rag_api`로 모델별 평가를 실행합니다.
4. `llm_cvd/analysis/analyze_results.py`로 결과 지표와 비용을 요약합니다.
5. 필요하면 `llm_cvd/analysis/inspect_rag_retrieval.py`, `llm_cvd/analysis/prepare_rag_qualitative_review.py`, `llm_cvd/analysis/visualize_rag_tsne.py`로 검색 결과를 추가 분석합니다.

## 실행 예시

실행 전 프로젝트 루트에서 의존성을 설치합니다.

```bash
python3 -m pip install -r requirements.txt
```

RAG DB를 구축합니다.

```bash
python3 -m llm_cvd.retrieval.build_index \
  --rag-dataset-root dataset/juliet_Real_Vul_data.csv \
  --cache-dir llm_cvd/cache \
  --index-dir llm_cvd/indexes \
  --index-name juliet_Real_Vul_data_codebert
```

zero-shot 평가를 실행합니다.

```bash
python3 -m llm_cvd.evaluation.baseline_api \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --providers chatgpt,claude,gemini,grok \
  --output-dir llm_cvd/results \
  --run-name cve_zeroshot \
  --env-file llm_cvd/.env
```

few-shot RAG 평가를 실행합니다.

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
  --env-file llm_cvd/.env
```

결과를 분석합니다.

```bash
python3 llm_cvd/analysis/analyze_results.py llm_cvd/results/cve_k6.jsonl \
  --pricing llm_cvd/pricing_config.json \
  --per-repeat-output llm_cvd/results/cve_k6_by_repeat.csv \
  --output llm_cvd/results/cve_k6_summary.csv
```

더 자세한 실행법은 [`llm_cvd/README.md`](llm_cvd/README.md)를 참고합니다.
