# LineGate AOI

> AI-Assisted First-Board Quality Gate for SMT Changeovers

LineGate AOI는 SMT 제품 또는 배치 전환 직후 제작되는 첫 생산품(First Board)의 R0805 부품을
**V1 상단 View와 V2·V2.1 양방향 경사 View로 검사하고, 부품 실장 상태와 납땜 상태를 종합하여
RELEASE / HOLD / REVIEW 권고를 제공하는 AI 품질 게이트 PoC**입니다.

본 프로젝트는 SolDef_AI의 Multi-view PCB 이미지 중 R0805 데이터를 활용합니다.
V1은 부품 실장 상태 검사에, V2·V2.1은 좌·우 납땜 접합부 상태 검사에 사용하며,
V3는 본 모델 학습 범위에서 제외했습니다.

## System Architecture

```text
V1 Top View · 부품 실장 상태
  └─ Superb AI Platform / Deployment
     └─ RF-DETR Small Object Detection
        ├─ good_BBBox
        └─ no_good_BBBox

V2 Left View · 좌측 납땜
V2.1 Right View · 우측 납땜
  └─ Local FastAPI
     └─ YOLO11s-seg Instance Segmentation
        ├─ good
        ├─ exc_solder
        ├─ poor_solder
        └─ spike

V1 + V2 + V2.1 검사 결과
  └─ LineGate Decision Logic
     └─ RELEASE / HOLD / REVIEW
        └─ Human Approval
```

## Main Features

- **V1 실장 상태 검사**
  - Superb AI에서 학습·배포한 **RF-DETR Small Object Detection**
  - 부품 전체 영역을 Bounding Box로 검출
  - `good_BBBox` / `no_good_BBBox` 판별
  - BBox Overlay 및 Confidence 표시

- **V2 / V2.1 납땜 상태 검사**
  - **YOLO11s-seg Instance Segmentation**
  - `good`, `exc_solder`, `poor_solder`, `spike` 4개 클래스
  - Polygon Mask Overlay 및 Confidence 표시

- **3각도 통합 검사**
  - V1 실장 상태 + V2 좌측 납땜 + V2.1 우측 납땜 결과를 모두 확인
  - 특정 View에서 결함이 검출되더라도 나머지 View 검사를 계속 수행해 결함 근거를 구체화

- **최종 로트 판정**
  - RELEASE / HOLD / REVIEW
  - 판정 사유 및 View별 근거 제공
  - 작업자용 권장 조치 안내

- **Human Approval**
  - AI 권고 승인
  - 예외 RELEASE 승인
  - HOLD 유지
  - REWORK 후 재검
  - 추가 검사 요청
  - 검사자 의견 및 변경 사유 기록
  - 승인 기록 JSON 다운로드

## Final Models

### V1 — RF-DETR Small Object Detection

V1은 초기에는 Polygon 기반 Instance Segmentation으로 학습했으나,
부품 외곽의 정밀 분할보다 **부품 단위의 정상/이상 판별**이 핵심 목적이라는 점을 반영해
최종적으로 Bounding Box 기반 Object Detection으로 전환했습니다.

최종 V1 학습 데이터는 다음과 같습니다.

| Class | Annotation |
|---|---:|
| `good_BBBox` | 129 |
| `no_good_BBBox` | 139 |
| **Total** | **268** |

최종 모델은 **RF-DETR Small**이며, 별도 Test Set 27장 기준 성능은 다음과 같습니다.

| Metric | Result |
|---|---:|
| Precision | 81.5% |
| Recall | 81.5% |
| F1-score | 81.5% |
| mAP | 0.658 |
| mAP@50 | 0.746 |
| Evaluation Confidence Threshold | 0.56 |

클래스별 성능은 다음과 같습니다.

| Class | Precision | Recall | F1-score | Test Instance |
|---|---:|---:|---:|---:|
| `no_good_BBBox` | 90.9% | 71.4% | 80.0% | 14 |
| `good_BBBox` | 75.0% | 92.3% | 82.8% | 13 |

> 서비스 코드에서는 Superb AI Deployment가 반환하는 클래스명 접미사 변형을 처리한 뒤,
> 내부 판정 로직에서 `good` / `no_good`으로 정규화하여 사용합니다.
> 학습·보고서 기준 공식 클래스명은 `good_BBBox` / `no_good_BBBox`입니다.

### V2 / V2.1 — YOLO11s-seg Instance Segmentation

V2와 V2.1은 동일한 클래스 체계를 공유하므로 하나의 데이터셋으로 통합해
**YOLO11s-seg** 모델을 학습했습니다.

최종 학습에는 데이터 유효성 검토 후 **260장**을 사용했고,
총 **627개 Polygon 객체**가 포함됐습니다.

| Class | Polygon Objects |
|---|---:|
| `good` | 113 |
| `exc_solder` | 238 |
| `poor_solder` | 118 |
| `spike` | 158 |
| **Total** | **627** |

데이터는 Seed 42를 고정해 Train / Validation / Test = **80 / 10 / 10**으로 분할했습니다.

| Split | Images |
|---|---:|
| Train | 208 |
| Validation | 26 |
| Test | 26 |
| **Total** | **260** |

최종 YOLO11s-seg 모델의 Test Set 기준 Mask 성능은 다음과 같습니다.

| Metric | Result |
|---|---:|
| Mask Precision | 0.810 |
| Mask Recall | 0.820 |
| Mask mAP50 | 0.861 |
| Mask mAP50-95 | 0.689 |

View별 내부 Test Set 성능은 다음과 같습니다.

| View | Images | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|
| V2 | 11 | 93.22% | 83.81% | 90.61% | 70.65% |
| V2.1 | 15 | 91.63% | 86.24% | 95.25% | 79.53% |

## AI Classes

### V1 — 부품 실장 상태

| Official Class | Service Internal Label | Meaning |
|---|---|---|
| `good_BBBox` | `good` | 부품이 패드에 정상적으로 실장된 상태 |
| `no_good_BBBox` | `no_good` | 부품 실장 상태가 정상 기준에서 벗어난 상태 |

V1은 위치 이상 원인을 세부 클래스로 분리하지 않고,
부품 본체와 패드 사이의 상대적인 위치·정렬 상태를 기준으로 정상/이상을 판별합니다.

### V2 / V2.1 — 납땜 접합부 상태

| Class | Meaning |
|---|---|
| `good` | 허용 가능한 정상 납땜 접합부 |
| `exc_solder` | 납땜량이 과도하게 형성된 상태 |
| `poor_solder` | 납땜량이 부족하거나 접합 상태가 불충분한 상태 |
| `spike` | 납땜부에 뾰족한 돌출이 형성된 상태 |

## Decision Policy

LineGate는 V1만 먼저 판정하고 검사를 중단하지 않습니다.
**V1, V2, V2.1 세 View 검사를 모두 수행한 뒤 결과를 종합**합니다.

| Case | V1 실장 상태 | V2·V2.1 납땜 상태 | Final Decision |
|---|---|---|---|
| 1 | 정상 | 모두 정상 | **RELEASE** |
| 2 | 이상 | 모두 정상 | **HOLD** |
| 3 | 정상 | 하나 이상 이상 | **HOLD** |
| 4 | 이상 | 하나 이상 이상 | **HOLD** |
| 5 | 저신뢰도 / 결과 불확실 | 어느 View든 해당 | **REVIEW** |
| 6 | 필수 View 누락 | 어느 View든 해당 | **REVIEW** |

- **RELEASE**: 전체 3각도 검사 결과가 정상
- **HOLD**: 실장 이상 또는 납땜 이상 검출
- **REVIEW**: 저신뢰도, 판정 불확실 또는 필수 View 누락

AI 결과는 최종 품질 결정을 자동 확정하는 것이 아니라,
작업자가 Bounding Box / Polygon Mask Overlay와 판정 근거를 검토할 수 있도록 지원합니다.

## Recommended Actions

- **V1 위치 이상**
  - 부품 실장 위치
  - 장착 프로그램
  - Setup 재확인

- **poor_solder**
  - 납땜 부족 여부 재검
  - 보강 가능성 검토

- **exc_solder**
  - 과납 및 인접부 접촉 가능성 점검

- **spike**
  - 돌출 형상 및 주변 간섭 가능성 점검

## Project Structure

```text
linegate-aoi/
├─ backend/
│  ├─ main.py
│  ├─ decision_engine.py
│  └─ __init__.py
├─ frontend/
│  ├─ app.py
│  └─ __init__.py
├─ models/
│  └─ best_yolo11s.pt
├─ demo_images/
   ├─ case1_v1-hold_v2-hold_v21-hold/
   └─ case2_v1-release_v2-hold_v21-hold/
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## Requirements

- Windows 10 / 11
- Anaconda 또는 Miniconda
- **Python 3.12**
- Superb AI Tenant API Key

## Initial Setup

프로젝트 폴더에서 Anaconda Prompt를 실행합니다.

```bat
conda create -n linegate312 python=3.12 -y
conda activate linegate312
python -m pip install -r requirements.txt
```

Python 버전을 확인합니다.

```bat
python --version
```

`Python 3.12.x`가 나오면 정상입니다.

## Superb AI API Key

API Key를 소스코드에 직접 저장하지 않습니다.

Streamlit을 실행할 Prompt에서 다음과 같이 환경변수를 설정합니다.

```bat
set SUPERB_AI_API_KEY=YOUR_API_KEY
```

> 실제 API Key를 GitHub 저장소, README, 스크린샷 등에 업로드하지 마세요.

## Run

현재 LineGate는 **FastAPI 서버와 Streamlit 웹 화면을 동시에 실행**합니다.
Anaconda Prompt 창 2개를 사용합니다.

### Prompt 1 — V2 / V2.1 FastAPI

```bat
cd /d "YOUR_PATH\linegate-aoi-v1.0"
conda activate linegate312
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

이 창은 종료하지 않습니다.

### Prompt 2 — V1 Superb AI + Streamlit

```bat
cd /d "YOUR_PATH\linegate-aoi-v1.0"
conda activate linegate312
set SUPERB_AI_API_KEY=YOUR_API_KEY
python -m streamlit run frontend/app.py
```

브라우저에서 Streamlit 화면이 열리면 실행 완료입니다.

## Inspection Flow

1. 생산 라인 및 배치 번호 입력
2. 동일 부품의 V1 / V2 / V2.1 이미지 등록
3. **AI 검사 시작**
4. RELEASE / HOLD / REVIEW 최종 로트 판정 확인
5. 판정 사유 확인
6. V1 BBox / V2·V2.1 Polygon Mask Overlay 증거 확인
7. 권장 조치 확인
8. 작업자 최종 승인 및 의견 저장
9. 승인 기록 JSON 다운로드

## Demo Data

`demo_images/`에는 로컬 시연에 사용할 수 있는 예시 이미지 세트가 포함되어 있습니다.

## Security

- `SUPERB_AI_API_KEY`는 환경변수로만 사용합니다.
- `.env`, Python cache, 실행 중 생성되는 Overlay 및 로그는 `.gitignore` 대상입니다.
- 실제 API Key를 GitHub 저장소에 커밋하지 마세요.

## Project

**Superb AI × BDAI Vision AI Hackathon — TRACK A 제조**

Team **코알라**
