# LLM-CVD 분석 스크립트

이 폴더에는 RAG 검색 결과 점검, 정성 평가 CSV/HTML 생성, t-SNE 시각화, 반복 실행 결과 요약 스크립트가 있습니다.

아래 예시는 `llm_cvd` 디렉터리에서 실행한다고 가정합니다.

## 파일 구성

| 파일 | 설명 |
| --- | --- |
| `analyze_results.py` | Accuracy, Precision, Recall, F1, 지연시간, 토큰 평균, 추정 비용을 계산합니다. |
| `inspect_rag_retrieval.py` | 각 대상 샘플에 대해 검색된 top-k RAG 예시를 납작한 CSV로 저장합니다. |
| `prepare_rag_qualitative_review.py` | RAG 검색 결과 CSV를 정성 평가용 컬럼과 빈 메모 필드가 있는 리뷰 CSV로 변환합니다. |
| `prepare_rag_excel_review.py` | 정성 평가 결과를 엑셀 검토용 컬럼과 per-query 요약 CSV로 재구성합니다. |
| `visualize_rag_tsne.py` | CVE query, 검색된 Juliet 예시, 배경 샘플 임베딩을 t-SNE CSV/HTML/PNG로 시각화합니다. |

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
