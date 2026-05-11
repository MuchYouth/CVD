# 데이터셋 README

이 디렉터리는 취약점 탐지 실험에 사용하는 CSV 데이터셋과 Juliet 데이터 분할 스크립트를 포함한다.
데이터셋은 크게 RAG 지식베이스 구축용, API 평가용, 기존 딥러닝 모델 비교용으로 구분한다.

## 공통 컬럼

대부분의 CSV 파일은 다음 컬럼을 사용한다.

`file_name`, `unique_id`, `target`, `vulnerable_line_numbers`, `project`, `source_signature_path`, `commit_hash`, `dataset_type`, `processed_func`

일부 real-world 확장 데이터셋은 `source_signature_path`가 없고, `VulnPatchDS_Vul_data.csv`는 `patched_code_source` 컬럼을 추가로 포함한다.

`target`은 취약 여부를 나타내는 라벨이다.

- `1`: 취약 코드
- `0`: 비취약 코드

`dataset_type`은 실험 분할을 나타낸다.

- `train_val`: 학습 또는 RAG 지식베이스 구축에 사용하는 데이터
- `test`: 평가에 사용하는 데이터

## 데이터셋별 역할

| 파일 | 샘플 수 | 라벨 분포 | 분할 | 역할 |
| --- | ---: | --- | --- | --- |
| `juliet_Real_Vul_data.csv` | 875 | 취약 486 / 비취약 389 | train_val 701 / test 174 | Juliet 기반 전체 데이터셋. RAG 지식베이스 구축의 원본 corpus로 사용한다. |
| `juliet_Real_Vul_train_val.csv` | 701 | 취약 382 / 비취약 319 | train_val | `juliet_Real_Vul_data.csv`에서 학습/검증 구간만 분리한 데이터셋. RAG 구축 시 test 데이터 누수를 막기 위한 기본 corpus로 사용한다. |
| `juliet_Real_Vul_test_val.csv` | 174 | 취약 104 / 비취약 70 | test | Juliet 전체 데이터셋에서 평가 구간만 분리한 데이터셋. RAG 구축 데이터와 분리된 in-domain 평가에 사용한다. |
| `juliet_Pair_Real_Vul_data.csv` | 80 | 취약 40 / 비취약 40 | test | Juliet/SARD 계열의 취약/비취약 pairwise 평가셋. 같은 계열의 코드에서 취약 코드와 대응되는 안전 코드를 구분하는지 확인한다. |
| `cve_Real_Vul_data.csv` | 20 | 취약 10 / 비취약 10 | test | 실제 CVE 및 오픈소스 프로젝트 기반 평가셋. Juliet 기반 RAG 지식이 real-world CVE 코드에도 전이되는지 확인한다. |
| `VulnPatchDS_Vul_data.csv` | 8,730 | 취약 4,365 / 비취약 4,365 | test | VulnPatchDS 기반 balanced 평가셋. 기존 딥러닝 모델 또는 API 기반 탐지 성능을 비교 평가하는 데 사용한다. |
| `Real_Vul_data.csv` | 1,987 | 취약 28 / 비취약 1,959 | test | FFmpeg 중심의 real-world 불균형 평가셋. 실제 프로젝트 코드에서 낮은 취약 비율 조건의 탐지 성능을 확인한다. |
| `Extended_Real_Vul_data_train.csv` | 146,552 | 취약 5,335 / 비취약 141,217 | train_val | Chrome, Linux, QEMU 등 여러 프로젝트를 포함한 대규모 real-world 학습/검증 데이터셋. 기존 모델 학습 또는 확장 실험에 사용한다. |
| `Extended_Real_Vul_data_test.csv` | 73,655 | 취약 366 / 비취약 73,289 | test | 대규모 real-world 테스트 데이터셋. 확장 학습 데이터로 학습한 모델의 일반화 성능을 평가한다. |

## 실험별 사용 구분

### RAG 구축

- 기본 corpus: `juliet_Real_Vul_train_val.csv`
- 전체 Juliet corpus가 필요한 분석: `juliet_Real_Vul_data.csv`

RAG 구축에서는 평가 데이터가 검색 지식베이스에 들어가지 않도록 `train_val` 구간을 우선 사용한다.
`juliet_Real_Vul_data.csv`는 전체 원본 데이터셋으로 보관하며, 분할이 필요할 때 기준 파일로 사용한다.

### API 테스트

- In-domain pairwise 평가: `juliet_Pair_Real_Vul_data.csv`
- Juliet in-domain test 평가: `juliet_Real_Vul_test_val.csv`
- Real-world CVE 평가: `cve_Real_Vul_data.csv`
- VulnPatchDS 기반 balanced 평가: `VulnPatchDS_Vul_data.csv`
- FFmpeg real-world 불균형 평가: `Real_Vul_data.csv`

API 테스트에서는 RAG가 검색한 Juliet 패턴이 같은 계열의 pairwise 예제와 실제 CVE/프로젝트 코드에서 얼마나 잘 작동하는지 확인한다.

### 기존 모델 및 확장 실험

- 대규모 학습/검증: `Extended_Real_Vul_data_train.csv`
- 대규모 테스트: `Extended_Real_Vul_data_test.csv`
- balanced 테스트 비교: `VulnPatchDS_Vul_data.csv`

Extended 데이터셋은 취약 샘플보다 비취약 샘플이 훨씬 많은 real-world 분포를 반영한다.
따라서 accuracy만으로 평가하기보다 취약 클래스 기준 precision, recall, F1 등을 함께 확인하는 것이 좋다.

## 보조 스크립트

### `split_juliet_real_vul_by_dataset_type.py`

`juliet_Real_Vul_data.csv`의 `dataset_type` 값을 기준으로 데이터를 분리하는 스크립트다.

- `train_val` 행: `juliet_Real_Vul_train_val.csv`
- `test` 행: 기본 출력값은 `juliet_Real_Vul_test.csv`

현재 디렉터리에는 test 분할 파일이 `juliet_Real_Vul_test_val.csv` 이름으로 보관되어 있으므로,
스크립트 실행 시 필요한 경우 `--test-output dataset/juliet_Real_Vul_test_val.csv` 옵션을 지정한다.

예시:

```bash
python3 dataset/split_juliet_real_vul_by_dataset_type.py \
  --input dataset/juliet_Real_Vul_data.csv \
  --train-output dataset/juliet_Real_Vul_train_val.csv \
  --test-output dataset/juliet_Real_Vul_test_val.csv
```
