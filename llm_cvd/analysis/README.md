# LLM-CVD 분석 스크립트

이 폴더에는 RAG 검색 결과 점검, 정성 평가 CSV/HTML 생성, t-SNE 시각화, 반복 실행 결과 요약 스크립트가 있습니다.

아래 예시는 `llm_cvd` 디렉터리에서 실행한다고 가정합니다.

## 파일 구성

| 파일 | 설명 |
| --- | --- |
| `analyze_results.py` | Accuracy, Precision, Recall, F1, 지연시간, 토큰 평균, 추정 비용을 계산합니다. |
| `inspect_rag_retrieval.py` | 각 대상 샘플에 대해 검색된 top-k RAG 예시를 납작한 CSV로 저장합니다. |
| `compute_rag_similarity_metrics.py` | inspection CSV를 바탕으로 cosine similarity, DB 내 거리 percentile, 출처 분포, CVE/OSS 참조율 그래프를 생성합니다. |
| `prepare_rag_qualitative_review.py` | RAG 검색 결과 CSV를 정성 평가용 컬럼과 빈 메모 필드가 있는 리뷰 CSV로 변환합니다. |
| `prepare_rag_excel_review.py` | 정성 평가 결과를 엑셀 검토용 컬럼과 per-query 요약 CSV로 재구성합니다. |
| `visualize_rag_tsne.py` | CVE query, 검색된 RAG 예시, 배경 샘플 임베딩을 t-SNE CSV/HTML/PNG로 시각화합니다. |
| `visualize_query_local_rag.py` | CVE query별로 query, top-6 RAG 예시, 주변 background sample을 local t-SNE 패널로 시각화합니다. |

## RAG 검색 결과 점검

각 CVE 샘플에 대해 어떤 Juliet 예시가 검색됐는지 CSV로 확인합니다.

```bash
python analysis/inspect_rag_retrieval.py \
  --rag-dataset-root ../dataset/juliet_Real_Vul_data.csv \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --index-name juliet_Real_Vul_data_codebert \
  --k 6 \
  --output-csv results/rag_retrieval_inspection_cve_k6.csv
```

정성 평가용 CSV를 만들려면 다음을 실행합니다.

```bash
python analysis/prepare_rag_qualitative_review.py \
  --input-csv results/rag_retrieval_inspection_cve_k6.csv \
  --output-csv results/rag_retrieval_qualitative_review_cve_k6.csv
```

리뷰어는 `cause_match`, `context_match`, `helpful_as_fewshot`에 `yes`, `partial`, `no`를 입력하고, `note`에는 짧은 근거를 남깁니다.
라벨 일치 여부보다 취약점 원인, 코드 문맥, few-shot 예시로서의 유용성에 집중하도록 자동 라벨 관련 필드는 제거됩니다.

엑셀 검토용 시트와 per-query 요약을 만들려면 다음을 실행합니다.

```bash
python analysis/prepare_rag_excel_review.py \
  --review-csv results/rag_retrieval_qualitative_review_cve_k6.csv \
  --output-csv results/rag_retrieval_qualitative_review_cve_k6_excel_review.csv \
  --summary-csv results/rag_retrieval_qualitative_review_cve_k6_excel_summary.csv
```

## t-SNE 시각화

검색된 예시와 CVE query 임베딩의 분포를 t-SNE로 확인합니다.

```bash
python analysis/visualize_rag_tsne.py \
  --inspection-csv results/rag_retrieval_inspection_cve_k6.csv \
  --target-dataset-csv ../dataset/cve_Real_Vul_data.csv \
  --rag-dataset-root ../dataset/juliet_Real_Vul_data.csv \
  --index-name juliet_Real_Vul_data_codebert \
  --output-csv results/rag_tsne_cve_k6_points.csv \
  --output-html results/rag_tsne_cve_k6.html \
  --output-png results/rag_tsne_cve_k6.png
```

이 스크립트는 기본적으로 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 설정하므로, 필요한 CodeBERT 모델과 FAISS 인덱스가 로컬에 준비되어 있어야 합니다.

## Signature DB 유사도 분석

`dataset/cve_Real_Vul_data.csv`를 query로 사용하고, `signature_db_Vul_data_codebert` RAG DB에서 검색된 top-6 예시가 실제로 얼마나 가깝고 어떤 출처에서 왔는지 확인합니다.
아래 명령은 프로젝트 루트에서 실행한다고 가정하며, 모든 산출물은 `llm_cvd/analysis/similarity_rag_result/` 아래에 저장됩니다.

산출물은 용도별 하위 폴더에 저장됩니다.

| 폴더 | 의미 |
| --- | --- |
| `retrieval/` | CVE query별 top-6 RAG 검색 원본 inspection CSV |
| `metrics/` | L2 거리, cosine similarity, DB 내 거리 percentile, 출처/참조 계열 상세 CSV |
| `plots/` | rank별 유사도 bar chart와 cosine heatmap |
| `source_distribution/` | retrieved top-6의 출처 분포와 CVE/Open Source Project 참조율 |
| `tsne_overview/` | 전체 query/retrieved/background embedding t-SNE overview |
| `query_local_panels/` | query별 top-6 local neighborhood 시각화 |

먼저 top-6 검색 결과를 inspection CSV로 저장합니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  llm_cvd/analysis/inspect_rag_retrieval.py \
  --rag-dataset-root dataset/signature_db_Vul_data.csv \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --cache-dir llm_cvd/cache \
  --index-dir llm_cvd/indexes \
  --index-name signature_db_Vul_data_codebert \
  --k 6 \
  --output-csv llm_cvd/analysis/similarity_rag_result/retrieval/rag_retrieval_inspection_cve_signature_db_k6.csv
```

이 결과는 각 query 샘플마다 rank 1-6의 검색 결과, FAISS L2 거리, lexical Jaccard, label 일치 여부, query/retrieved snippet을 담습니다.
LLM을 호출하기 전에 RAG가 실제로 어떤 예시를 few-shot 후보로 가져오는지 확인하는 기본 자료입니다.

다음으로 embedding 유사도 지표와 출처 분포 그래프를 생성합니다.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cvd HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python llm_cvd/analysis/compute_rag_similarity_metrics.py \
  --inspection-csv llm_cvd/analysis/similarity_rag_result/retrieval/rag_retrieval_inspection_cve_signature_db_k6.csv \
  --rag-dataset-root dataset/signature_db_Vul_data.csv \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --index-dir llm_cvd/indexes \
  --index-name signature_db_Vul_data_codebert
```

이 스크립트의 주요 산출물은 다음과 같습니다.

| 산출물 | 의미 |
| --- | --- |
| `metrics/rag_similarity_metrics_cve_signature_db_k6.csv` | inspection CSV에 `cosine_similarity`, `distance_gap_from_rank1`, `distance_percentile_in_db`, `retrieved_source_category`, `retrieved_reference_family` 등을 추가한 상세 분석 파일입니다. 거리와 출처를 row 단위로 함께 볼 때 사용합니다. |
| `plots/rag_similarity_rank_bar_cve_signature_db_k6.png` | query별 top-6의 FAISS L2 거리와 cosine similarity를 함께 보여줍니다. rank 1만 유독 가까운지, top-6가 비슷한 후보군인지 판단할 수 있습니다. |
| `plots/rag_similarity_heatmap_cve_signature_db_k6.png` | 20개 query와 rank 1-6의 cosine similarity heatmap입니다. 어떤 CVE query가 RAG DB와 전반적으로 잘 붙는지 빠르게 확인합니다. |
| `source_distribution/rag_retrieved_source_distribution_cve_signature_db_k6.csv` / `.png` | 검색된 top-6 예시가 `Open Source Project`, `OWASP Tutorial`, `Juliet` 중 어디에서 왔는지 집계합니다. rank instance 기준과 unique retrieved sample 기준을 함께 봅니다. |
| `source_distribution/rag_cve_reference_rate_cve_signature_db_k6.csv` / `.png` | `Open Source Project` 계열을 `CVE/Open Source Project` 참조로 묶어 CVE/OSS 참조율을 보여줍니다. RAG DB에 실제 프로젝트/CVE 계열 샘플을 많이 넣었을 때 검색 결과도 그쪽으로 쏠리는지 설명할 때 사용합니다. |

현재 `signature_db_Vul_data.csv` 메타데이터에는 직접적인 `CVE-YYYY-NNNN` ID가 거의 없으므로, `rag_cve_reference_rate_*`는 정확한 CVE ID 매칭률이 아니라 `CVE/Open Source Project` 계열 참조율로 해석해야 합니다.
즉, Juliet/OWASP tutorial이 아닌 실제 오픈소스 프로젝트 signature를 얼마나 자주 검색했는지를 보여주는 지표입니다.

t-SNE 시각화는 검색된 예시, CVE query, 배경 signature sample의 임베딩 분포를 2차원으로 확인하는 sanity check입니다.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cvd HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python llm_cvd/analysis/visualize_rag_tsne.py \
  --inspection-csv llm_cvd/analysis/similarity_rag_result/retrieval/rag_retrieval_inspection_cve_signature_db_k6.csv \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --rag-dataset-root dataset/signature_db_Vul_data.csv \
  --index-dir llm_cvd/indexes \
  --index-name signature_db_Vul_data_codebert \
  --corpus-label "Signature DB" \
  --output-csv llm_cvd/analysis/similarity_rag_result/tsne_overview/rag_tsne_cve_signature_db_k6_points.csv \
  --output-html llm_cvd/analysis/similarity_rag_result/tsne_overview/rag_tsne_cve_signature_db_k6.html \
  --output-png llm_cvd/analysis/similarity_rag_result/tsne_overview/rag_tsne_cve_signature_db_k6.png
```

`rag_tsne_cve_signature_db_k6_points.csv`는 t-SNE 좌표와 점 종류를 저장하고, HTML/PNG는 이를 시각화합니다.
t-SNE는 고차원 embedding의 전역 거리를 보존하지 않으므로, 최종 유사도 판단은 `rag_similarity_metrics_*`의 L2 거리, cosine similarity, percentile과 함께 해석해야 합니다.

query별로 top-6 검색 결과가 query 주변에서 어떻게 배치되는지 더 자세히 보려면 local panel 시각화를 생성합니다.
각 패널은 CVE query 1개, retrieved top-6, query 기준 가까운 background signature sample 30개를 함께 그립니다.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cvd HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python llm_cvd/analysis/visualize_query_local_rag.py \
  --metrics-csv llm_cvd/analysis/similarity_rag_result/metrics/rag_similarity_metrics_cve_signature_db_k6.csv \
  --rag-dataset-root dataset/signature_db_Vul_data.csv \
  --target-dataset-csv dataset/cve_Real_Vul_data.csv \
  --index-dir llm_cvd/indexes \
  --index-name signature_db_Vul_data_codebert \
  --output-dir llm_cvd/analysis/similarity_rag_result/query_local_panels
```

`query_local_panels/query_001_local_rag.png`부터 `query_020_local_rag.png`는 query별 local t-SNE 패널입니다.
`query_local_rag_index.html`은 20개 패널과 rank별 top-6 요약 테이블을 한 번에 보여주고, `query_local_rag_points.csv`는 query/retrieved/background 각 점의 좌표와 거리, cosine, 출처 정보를 저장합니다.
`query_local_rag_contact_sheet.png`는 20개 패널을 한 장으로 훑어보기 위한 요약 이미지입니다.

## 결과 분석

반복 실행 결과가 있으면 `analysis/analyze_results.py`는 먼저 `repeat_id`별 지표를 계산한 뒤, 반복별 지표의 평균과 표준편차를 요약합니다.

```bash
python analysis/analyze_results.py results/juliet-real_k6.jsonl \
  --pricing pricing_config.json \
  --per-repeat-output results/juliet-real_k6_by_repeat.csv \
  --output results/juliet-real_k6_summary.csv
```

```bash
python analysis/analyze_results.py results/juliet-real_zeroshot.jsonl \
  --pricing pricing_config.json \
  --per-repeat-output results/juliet-real_zeroshot_by_repeat.csv \
  --output results/juliet-real_zeroshot_summary.csv
```

`pricing_config.json`의 가격값은 기본적으로 `0.0`입니다.
비용 추정이 필요하면 제공자별 1M input/output token 가격을 채운 뒤 분석을 실행하세요.
