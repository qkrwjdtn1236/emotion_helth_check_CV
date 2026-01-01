# 사용자 이상탐지 행동 분석

## 프로젝트 개요

본 프로젝트는 **카메라 기반 영상 분석**과 **인공지능 기술**을 융합하여 운전자의 피로도, 스트레스, 졸음 상태를 실시간으로 감지하는 **비접촉식 운전자 감시 시스템**입니다.

- **핵심 기술**: YOLO 객체 탐지 + dlib 얼굴 특징 인식

---

## 연구 목적

운전자의 상태를 비접촉적으로 파악할 수 있는 혁신적인 운전자 감시 시스템 개발을 목표로 합니다.

### 주요 목표
- ✅ 운전자의 얼굴 표정, 눈 깜빡임, 피부 광택을 **실시간 분석**
- ✅ 피로도, 스트레스, 졸음 등 **생리적 및 심리적 상태 판별**
- ✅ 이상 감지 시 **시각·청각 경고** 및 **차량 제어 대응**
- ✅ 장기적으로 개인 **맞춤형 건강 솔루션** 제공

---

## 연구 동기

### 교통사고 현황
| 통계 항목 | 데이터 | 출처 |
|---------|--------|------|
| 운전자 요인 비율 | **약 90% 이상** | 한국 경찰청 |
| 고령층(65세+) 사고 비율 (2024) | **21.6%** (역대 최고) | 교통사고 통계 |
| 고령 운전자 사고 증가율 (2020→2024) | **약 30% 증가** | 한국 경찰청 |
| 85세 이상 운전자 치사율 | **60대의 약 3배** | 2023년 통계 |

### 문제점
- **고령화 사회 진입**: 고령 운전자 비중 급증
- **인지 능력 저하**: 시각 능력 감소, 반응 속도 둔화
- **만성질환 위험**: 고혈압, 당뇨병, 뇌졸중 등 돌발 상황 발생
- **기존 시스템 한계**: 운전 중 순간적인 건강 상태 변화 미감지

### 참고 사례
- **일본 ADAS 시스템**: 사포카S를 통해 모든 차량 대비 **41.6% 낮은 사고율** 달성
- **2024년 대구 사건**: 저혈당 쇼크로 인한 9중 추돌 사고 발생

---

## 기술 스택

### 핵심 라이브러리
```
- YOLO (YOLOv8/v11): 실시간 객체 탐지
- dlib: 얼굴 특징점 추출 (68개 랜드마크)
- OpenCV: 영상처리 및 색공간 변환
- PyTorch: 딥러닝 프레임워크
- NumPy & Pandas: 데이터 분석
```

### 개발 환경
- **Language**: Python 3.8+
- **Framework**: PyTorch, OpenCV
- **GPU**: NVIDIA CUDA 지원
- **Platform**: Linux, Windows, macOS

---

## 연구 내용

### 이론적 배경

운전자의 얼굴 영상 분석을 통해 생리적·심리적 상태를 추정하는 비접촉식 생체정보 분석입니다.

**핵심 알고리즘**: CNN(Convolutional Neural Network)
- 카메라가 인식한 얼굴의 세부 특징 분석
- 다층 신경망을 통한 상태 정량화
- 피로도, 스트레스, 졸음 상태 분류

### 선행 연구

| 연구자 | 제목 | 주요 성과 | 한계 |
|-------|------|---------|------|
| 석창훈 (2017) | 스마트 영상센서를 이용한 헬스케어 모니터링 | ICA + FFT를 통한 심박수/호흡수 추정 | 의료 환경 중심, 운전 환경 미적용 |
| 기존 차량 시스템 | 기계적 데이터 활용 | 속도, 차선 이탈, 핸들 조작 감지 | 운전자 생리적 상태 미감지 |

---

## 시스템 구조

### 1️⃣ dlib 기반 얼굴 특징 추출

```
입력 영상 
  ↓
[dlib 얼굴 감지기]
  ↓
68개 특징점 추출 (눈, 코, 입, 턱, 윤곽)
  ↓
Eye Aspect Ratio (EAR) / Mouth Aspect Ratio (MAR) 계산
  ↓
RGB → HSV 색공간 변환
  ↓
[피부 특성 분석]
  ↓
Saturation / Value 값 추출 (피로도 지표)
```

**주요 지표**
- **Eye Aspect Ratio (EAR)**: 눈 감김 여부 판정
- **Saturation/Value**: 피부 광택 및 밝기 분석
- **윤곽 변화**: 입꼬리, 이마 주름 추적

### 2️⃣ YOLO 기반 객체 탐지

```
입력 영상
  ↓
[YOLO 모델]
  ↓
실시간 이진 상태 분류
  │
  ├─ 눈 상태: 열림 / 감김
  ├─ 입 상태: 열림 / 닫힘
  ├─ 얼굴 탐지: 위치 및 신뢰도
  └─ 휴대폰 감지: 사용 여부
```

**YOLO의 장점**
- 임계값 설정 불필요 (자동 학습)
- 개인차 및 환경 변화에 강건
- **실시간 처리**: 대시캠 환경 지원

---

## 데이터 분석

### 모니터링 항목 (7가지)

| 항목 | 정의 | 활용 | 정확도 |
|------|------|------|--------|
| **Closed Eye** (눈 감김) | 양쪽 눈이 0.5초 이상 닫혀있음 | 졸음 감지 | 68% |
| **Opened Eye** (눈 열림) | 눈이 완전히 열린 상태 | 각성도 평가 | 92% |
| **Closed Mouth** (입 닫힘) | 입이 닫혀있는 정상 상태 | 안정적 운전 | 95% |
| **Opened Mouth** (입 열림) | 하품, 말하기, 음식 섭취 | 주의력 저하 감지 | 96% |
| **Face** (얼굴) | 얼굴 영역 탐지 | 기본 참조 | 100% |
| **Phone** (핸드폰) | 휴대폰 사용 여부 | 주의력 분산 | 92% |
| **기타** | 안전벨트, 고개 방향 | 안전 행동 | - |

### 성능 지표

```
전체 평균 mAP@0.5 = 0.872 (87.2%)
클래스별 정확도 = 93~95%

혼동 행렬 분석:
- 얼굴 탐지: 100% (가장 우수)
- 입 닫힘: 95% (매우 안정적)
- 입 열림: 96% (우수한 성능)
- 눈 감김: 68% (개선 필요)
- 휴대폰: 92% (양호)
```

### 성능 분석

#### ✅ 우수한 성능 (90% 이상)
- **입 닫힘** (95%): 운전자의 정상 상태로 지속시간이 길어 모델 학습 용이
- **얼굴 탐지** (100%): 명확한 윤곽과 색상 특징, 큰 영역 차지
- **입 열림** (96%): 명확하게 벌어지는 특징
- **눈 열림** (92%): 상대적으로 안정적인 특징

#### ⚠️ 개선 필요 (90% 미만)
- **눈 감김** (68%): 짧은 지속시간, 작은 영역, 조명에 민감
  - 개선 방안: 학습 데이터 증강, 연속 프레임 분석
- **휴대폰** (92%): 작은 크기, 다양한 형태, 가려짐
  - 개선 방안: 손-휴대폰 관계 학습 강화

---

## 성과 및 결과

### 주요 성과

✨ **비접촉식 생체신호 분석 성공**
- 센서 착용 불편함 제거
- 카메라만으로 피로도 측정 가능

📊 **종합적 상태 평가**
- 7가지 항목 실시간 분석
- 단순 피로 감지 → 주의력, 안전 의식, 집중도 통합 평가

⚡ **고정확도 인식**
- 평균 93~95% 정확도
- mAP@0.5 = 0.872
- 실제 차량 환경 적용 가능

### 기존 기술과의 비교

| 기술 | 장점 | 단점 |
|------|------|------|
| **접촉식 센서** (ECG, PPG) | 매우 정확 | 불편함, 지속 측정 어려움 |
| **기존 ADAS 시스템** | 기계적 데이터 분석 | 순간적 건강 변화 미감지 |
| **본 연구 (DMS)** | 비접촉식, 종합적 분석, 실시간 처리 | 조명 변화에 영향 받음 |

---

## 파일 구조

```
DMS-Project/
├── README.md
├── requirements.txt
├── src/
│   ├── dlib_face_detection.py      # 얼굴 특징점 추출
│   ├── yolo_object_detection.py    # YOLO 기반 상태 분류
│   ├── skin_analysis.py             # 피부 특성 분석
│   └── main.py                      # 통합 시스템
├── data/
│   ├── training/                    # 학습 데이터
│   └── validation/                  # 검증 데이터
├── models/
│   ├── yolo_dms.pt                  # 학습된 YOLO 모델
│   └── config.yaml                  # 모델 설정
├── results/
│   ├── confusion_matrix.png
│   ├── performance_metrics.json
│   └── logs/
└── docs/
    ├── METHODOLOGY.md               # 연구 방법론
    └── TECHNICAL_DETAILS.md         # 기술 상세 내용
```

---

## 설치 및 실행

### 요구사항
```bash
Python 3.8+
CUDA 11.0+ (GPU 사용 시)
```

### 설치
```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/DMS-Project.git
cd DMS-Project

# 2. 환경 설정
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. 패키지 설치
pip install -r requirements.txt
```

### 실행
```bash
# 웹캠으로 실시간 모니터링
python src/main.py --input webcam

# 비디오 파일 분석
python src/main.py --input video.mp4 --output result.mp4

# 이미지 분석
python src/main.py --input image.jpg
```

---

## 모델 결과
![image]('/train8/train_batch34202.jpg')
![image]('/train8/val_batch2_labels.jpg')

## 성과 수치

| 지표 | 값 | 평가 |
|------|-----|------|
| **전체 mAP@0.5** | 0.872 | ✅ 우수 |
| **평균 정확도** | 93~95% | ✅ 우수 |
| **실시간 처리 속도** | 30+ FPS | ✅ 우수 |
| **환경 적응성** | 제한적 | ⚠️ 개선 필요 |
| **연령대 포함도** | 미확인 | ⚠️ 개선 필요 |

---

## 참고문헌

1. King, D. E. (2009). "Dlib-ml: A Machine Learning Toolkit." *Journal of Machine Learning Research (JMLR)*, 10, 1755-1758.

2. 대한민국 경찰청 (2023). "2023년 교통사고 분석 통계." 경찰청 교통관리실.

3. Shorten, C., & Khoshgoftaar, T. M. (2019). "A Survey on Image Data Augmentation for Deep Learning." *Journal of Big Data*, 6(1), 1-48.

4. Ultralytics. (2024). "YOLOv11: An Overview of the Key Architectural Enhancements." Ultralytics Official Documentation and Research Repository.

5. Sun, L., Lu, Z., Jiang, H., & Wang, Z. (2018). "Multimodal Fusion for Multimedia Analysis: A Survey." *Proceedings of the IEEE*, 106(8), 1423-1453.

6. Ayyoob, H. A., Veerappa, A., & Gopakumar, R. (2019). "A Real-Time Driver Monitoring System using Deep Learning." *Journal of Ambient Intelligence and Humanized Computing*, 10(11), 4395-4411.

7. Poh, M. Z., McDuff, D. J., & Picard, R. W. (2011). "Non-Contact, Automated Cardiac Pulse Measurements at a Distance and Intended Applications in Cars." *International Conference on Affective Computing and Intelligent Interaction*.

8. 한국소비자원 (2024). "고령운전자 인전실태조사 결과" 안전감시국 생활안전팀

---
