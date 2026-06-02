# Signature DB RAG Similarity Results

이 폴더는 `dataset/cve_Real_Vul_data.csv` 20개 query를 `signature_db_Vul_data_codebert` RAG DB에 검색한 뒤 만든 분석 산출물을 용도별로 나눠 보관합니다.

| 폴더 | 내용 |
| --- | --- |
| `retrieval/` | CVE query별 top-6 RAG 검색 원본 inspection CSV |
| `metrics/` | L2 거리, cosine similarity, DB 내 거리 percentile, 출처/참조 계열을 합친 상세 CSV |
| `plots/` | query-rank 유사도 bar chart와 heatmap |
| `source_distribution/` | retrieved top-6의 출처 분포와 CVE/Open Source Project 참조율 |
| `tsne_overview/` | 전체 query/retrieved/background embedding t-SNE overview |
| `query_local_panels/` | query별 local t-SNE 패널, HTML 인덱스, contact sheet |

`tsne_overview/`와 `query_local_panels/`의 t-SNE 좌표는 시각적 sanity check입니다.
최종 유사도 해석은 `metrics/`의 FAISS L2 거리, cosine similarity, distance percentile과 함께 봐야 합니다.
