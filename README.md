# AGV-SafeNav

**Reliability-aware Camera–LiDAR Fusion for Safe Autonomous Navigation**  
카메라 기반 주행 가능 영역 인식에서 시작해, ROS 2/Nav2 기반 AGV 자율주행, Camera–LiDAR 센서 융합, 센서 열화 대응, 동적 장애물 추적, 실시간성 검증으로 확장하는 프로젝트입니다.

> **현재 단계:** Phase 1 — ADE20K를 이용한 **Binary Floor Segmentation Baseline** 데이터 파이프라인 구축.  
> **현재 학습 환경:** Windows, NVIDIA GeForce RTX 5060 Ti 1개, Python 3.11.  
> **실증 예정 플랫폼:** JETANK / Jetson Nano (Ubuntu 18.04.5, JetPack 4.5, L4T R32.5.0). 현재 LiDAR와 AGV 실기 사용이 제한되어 있으므로 PC에서 Baseline을 먼저 개발합니다.

## 1. 프로젝트 개요

### 목표

사용자가 목적지를 지정하면 AGV가 카메라와 추후 추가할 LiDAR를 활용해 주행 가능 영역 및 장애물을 인식하고, 위험도를 고려하여 목적지까지 주행하는 시스템을 개발합니다.

주요 개발 목표는 다음과 같습니다.

1. **Vision AI:** RGB 영상에서 주행 가능 영역(Drivable Area) 분할
2. **Sensor Fusion:** Camera Semantic Grid와 LiDAR Occupancy Grid를 결합한 Costmap 구성
3. **Navigation:** ROS 2/Nav2를 활용한 위치 추정, 경로계획 및 제어
4. **Robustness:** 카메라 가림·조명 변화·LiDAR 누락·데이터 지연에 대응하는 Adaptive Fusion 및 Safety Supervisor
5. **Dynamic Obstacle:** 이동 장애물 추적과 충돌 위험 판단
6. **Validation:** 이벤트 기반 로그 수집, End-to-end Latency 및 안전성 평가

**현재 ADE20K Baseline의 `floor` 라벨은 실제 AGV의 `drivable` 라벨과 같지 않습니다.** 처음에는 *Floor/Non-floor* 분할 문제를 풀고, JETANK 실증 환경의 데이터를 확보한 이후 실제 주행 가능성에 맞는 라벨로 파인튜닝·검증합니다. 카메라 Segmentation만으로 절대 거리나 로봇 위치가 확보되는 것은 아닙니다.

## 2. 개발 로드맵

| Phase | 목표 | 상태 |
|---|---|---|
| 1 | ADE20K Binary Floor Segmentation 학습·검증, 이후 Camera 기반 주행 Baseline | **진행 중** |
| 2 | Camera–LiDAR Semantic/Occupancy Costmap 및 Adaptive Fusion | 계획 |
| 3 | LiDAR Clustering, Kalman Filter, 충돌 위험 판단 및 재계획 | 계획 |
| 4 | 센서 불일치·안전 정지·지연 발생 시 Event-triggered Logging | 계획 |
| 5 | ROS 2 QoS, 데이터 신선도, P95/P99 Latency, Deadline Miss 및 연산 최적화 | 계획 |

약 6주 개발을 목표로 Phase 3까지 우선 완성하고, Phase 4~5는 시간과 하드웨어 상황에 맞춰 확장합니다. Latency 계측과 안전 정지 설계는 초기부터 반영합니다.

## 3. 현재 Baseline 파이프라인

```text
ADE20K RGB images + semantic annotations
                  |
                  v
      Binary floor mask conversion
         0: non-floor
         1: floor
       255: ignore
                  |
                  v
         PyTorch Dataset/Loader
                  |
                  v
       LR-ASPP + MobileNetV3
             (planned)
                  |
                  v
    Floor IoU / Precision / Recall
        + False Drivable Rate
                  |
                  v
     JETANK target-data fine-tuning
             (future)
                  |
                  v
       BEV Grid / ROS 2 / Nav2
             (future)
```

## 4. 프로젝트 구조

아래는 **권장 디렉터리 구조**입니다. 실제 파일이 아직 없다면 필요한 단계에서 생성합니다.

```text
AD/                              # 저장소 루트 (원하는 이름 사용 가능)
├── configs/                     # 학습/데이터 설정 (예정)
├── data/                        # Git에 포함되는 Python 데이터 로더
│   ├── __init__.py
│   └── ade20k_dataset.py
├── data_preprocess/
│   ├── __init__.py
│   ├── prepare_ade20k.py        # ADE20K -> binary floor mask
│   └── verify_dataset.py        # 이미지/마스크 정합성 검사
├── datasets/                    # Git 제외: 원본·변환 데이터
│   ├── ADEChallengeData2016/
│   │   ├── images/
│   │   │   ├── training/
│   │   │   └── validation/
│   │   └── annotations/
│   │       ├── training/
│   │       └── validation/
│   └── floor_binary/
│       └── masks/
│           ├── training/
│           └── validation/
├── models/                      # LR-ASPP 모델 코드 (예정)
├── tools/
│   ├── __init__.py
│   ├── test_dataloader.py
│   ├── train.py                 # 예정
│   ├── evaluate.py              # 예정
│   └── inference.py             # 예정
├── ros2_ws/                     # 향후 ROS 2 워크스페이스
├── deployment/                  # 향후 Jetson 배포
├── outputs/                     # Git 제외: 모델·로그·시각화
├── requirements.txt
├── .gitignore
└── README.md
```

> `datasets/` 전체를 `.gitignore`로 제외할 경우 `ade20k_dataset.py`는 그 안에 두지 말고 `data/`에 배치하세요.

## 5. 설치 — Windows + RTX 5060 Ti

권장 환경은 **Python 3.11**과 RTX 5060 Ti(`sm_120`)를 지원하는 **CUDA 12.8 PyTorch 빌드**입니다. NVIDIA 드라이버가 호환되는지 확인하세요.

### 5.1 Conda 환경 생성

```powershell
conda create -n agv_safenav python=3.11 -y
conda activate agv_safenav
```

### 5.2 PyTorch 설치

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 5.3 프로젝트 의존성 설치

```powershell
python -m pip install -r requirements.txt
```

데이터 전처리만 수행한다면 `numpy`, `Pillow`, `tqdm`만으로 시작할 수 있습니다. 학습 단계에서는 `torch`, `torchvision` 등이 필요합니다.

### 5.4 GPU 확인

```powershell
python -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('Available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Capability:',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"
python -c "import torch; x=torch.randn(256,256,device='cuda'); print((x@x).mean().item())"
```

CUDA 인식뿐 아니라 실제 GPU 연산이 성공하는지 확인합니다.

### requirements.txt 예시

> `torch`, `torchvision`은 위 CUDA 전용 명령으로 별도 설치합니다.

```text
numpy>=1.26.4,<2.0
Pillow>=10.0.0
opencv-python>=4.8.0
tqdm>=4.66.0
torchmetrics>=1.4.0
timm>=1.0.0
albumentations>=1.4.0,<2.0
PyYAML>=6.0
matplotlib>=3.8.0
pandas>=2.0.0
scikit-learn>=1.4.0
tensorboard>=2.16.0
```

## 6. 데이터셋 — ADE20K

사용 데이터: **ADEChallengeData2016**, ADE20K Scene Parsing Benchmark.

- Training: **20,210**장
- Validation: **2,000**장
- 원본 Semantic Annotation: **150개 클래스**
- 공식 페이지: https://sceneparsing.csail.mit.edu/
- 다운로드: http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

다운로드 후 `datasets/ADEChallengeData2016/` 아래에 원본을 압축 해제합니다. 파일이 `ADEChallengeData2016/ADEChallengeData2016/`로 이중 중첩되지 않았는지 확인하세요.

### 6.1 Binary Floor 라벨 규칙

`ADEChallengeData2016` 원본 Annotation PNG 기준:

| 원본 Pixel ID | 변환된 Binary Mask | 뜻 |
|---|---:|---|
| `0` | `255` | Ignore / Unlabeled |
| `4` | `1` | Floor |
| 나머지 | `0` | Non-floor |

**주의:** 0-based 150-class 학습 라벨에서는 Floor가 `3`일 수 있습니다. **이 README와 전처리 코드는 원본 ADEChallengeData2016 PNG 클래스 ID(`4`)를 전제로 합니다.** 변환 전 실제 PNG 값을 확인해야 합니다. PNG가 인덱스 방식일 때 RGB 변환으로 클래스 ID를 훼손하지 마세요.

### 6.2 전처리

저장소 루트에서 실행합니다.

```powershell
python -m data_preprocess.prepare_ade20k
```

변환된 Mask는 `datasets/floor_binary/masks/{training,validation}/`에 저장되며 원본 RGB는 복사하지 않습니다.

### 6.3 데이터 검증

```powershell
python -m data_preprocess.verify_dataset
```

검증할 항목:

- 원본 이미지마다 대응하는 Mask 존재
- RGB와 Mask의 해상도 일치
- Mask 픽셀 값이 `{0, 1, 255}`에 포함
- Training/Validation 각각의 샘플 수 및 Floor 포함 이미지 수

### 6.4 Dataset Loader 및 Batch 확인

```powershell
python -m tools.test_dataloader
```

테스트 예시 (`batch_size=4`, `image_size=(240,320)`):

```text
Dataset size: 20210
Image shape: torch.Size([4, 3, 240, 320])
Mask shape: torch.Size([4, 240, 320])
DataLoader test passed!
```

Mask 값은 배치에 따라 `0`, `1`, `255`가 모두 나타나지 않을 수도 있습니다. 이미지 Resize는 **BILINEAR**, Mask Resize는 **NEAREST**를 사용해야 클래스 ID를 유지할 수 있습니다. Windows에서는 처음에 `num_workers=0`으로 검증합니다.

### `No module named 'data'` 오류

`python tools/test_dataloader.py` 대신 **저장소 루트에서** 실행합니다.

```powershell
cd C:\path\to\AD
python -m tools.test_dataloader
```

`data/__init__.py`, `tools/__init__.py`를 추가하면 패키지 구조를 명확히 할 수 있습니다. 프로젝트 내부 Import는 루트 기준으로 작성합니다.

## 7. 학습 계획 (미구현)

다음 단계에서 LR-ASPP + MobileNetV3 기반 2-class Segmentation을 구현합니다. **아래 설정은 초기 제안이며 아직 실험 결과가 아닙니다.**

| 설정 | 초기값 |
|---|---|
| Model | LR-ASPP + MobileNetV3-Large |
| Classes | 2 (`non-floor`, `floor`) |
| Input | RGB 240 × 320 (H × W) |
| Pretraining | ImageNet backbone weight 검토 |
| Loss | CrossEntropyLoss(`ignore_index=255`), 추후 Dice 비교 |
| Optimizer | AdamW |
| LR | 1e-4 |
| Batch size | 8부터 시도 |
| Epoch | 50 |
| Checkpoint | 최고 Validation Floor IoU 기준 |

평가 지표: Floor IoU, mIoU, Precision, Recall, False Drivable Rate(실제 명칭은 Floor False Positive 지표로 시작), 추론 지연 및 P95/P99 Latency.

> `tools/train.py`, `tools/evaluate.py`, `tools/inference.py`는 현재 계획된 파일입니다. 구현 전에는 `python -m tools.train` 등을 실행할 수 없습니다.

## 8. 실제 JETANK 실증으로 확장

### 8.1 Target-domain Fine-tuning

1. JETANK 실제 카메라 위치·각도를 고정하여 실내 RGB 영상 수집
2. 조명·그림자·장애물·바닥 재질별 샘플 확보
3. 실제 **Drivable/Non-drivable/Ignore** 정책으로 재라벨링
4. Fine-tuning 전 Zero-shot Transfer 성능 측정
5. Target Train/Validation/Test를 **촬영 세션·장소 단위로 분할**
6. Fine-tuning 후 독립 Test Set에서 재평가

### 8.2 BEV 및 ROS 2 / Nav2

```text
Camera -> Floor/Drivable Mask -> Camera Calibration/IPM -> BEV Semantic Grid
                                                            |
LiDAR (future) -> Occupancy Grid -----------------------------+--> Fusion Costmap
                                                                    |
Localization / TF / Goal Pose --------------------------------------+--> Nav2
                                                                          |
                                                                   Safety Gate
                                                                          |
                                                                     AGV Motor
```

지면 평면 IPM은 카메라 Intrinsic/Extrinsic 및 평면 가정을 필요로 합니다. 2D LiDAR는 스캔 평면 밖 장애물을 완전히 관측하지 못합니다. 목적지 주행을 위해서는 별도의 Odometry/Localization도 필요합니다.

### 8.3 실시간성 및 안전성

- 이미지 처리와 Navigation/Control 실행 주기 분리
- Sensor Timestamp 및 Data Age 검사
- Queue 누적 방지와 Latest-frame 처리 검토
- Safety Supervisor에서 감속/정지 명령 중재
- 명령 수신 중단 시 Motor Watchdog Stop
- End-to-end Latency, P95/P99, Deadline Miss Rate 실측
- Event-triggered rosbag2 기록 및 실패 케이스 재평가

Jetson Nano의 Ubuntu 18.04 / L4T R32.5.0과 개발 PC의 최신 PyTorch 환경은 서로 다릅니다. 배포 시 ONNX/TensorRT 버전·연산자·카메라 드라이버·ROS 2 호환성을 별도로 검증해야 하며, PC의 학습 환경을 그대로 복사해 실행할 수 있다고 가정하지 않습니다. ROS 2 Foxy는 지원 종료된 배포판이므로 장기 유지보수 및 최신 Nav2 활용에는 호환되는 다른 컴퓨팅 환경을 고려합니다.

## 9. 권장 Git 관리

데이터 원본·모델 가중치·실험 출력은 Git에 포함하지 않습니다.

```gitignore
# Raw and processed datasets
datasets/

# Training artifacts
outputs/
runs/
checkpoints/
*.pt
*.pth
*.onnx
*.engine

# Python
__pycache__/
*.pyc
.venv/
.env

# IDE
.vscode/
.idea/

# ROS 2 build outputs
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
```

```powershell
git add README.md requirements.txt .gitignore data data_preprocess tools
git commit -m "Document ADE20K binary floor baseline"
git push origin main
```

실제로 존재하는 파일만 `git add`하세요. 데이터셋의 재배포 가능 여부와 사용 조건은 공식 배포 안내를 확인합니다.

## 10. 현재 진행 상태 / 다음 작업

- [x] 프로젝트 목표 및 단계별 아키텍처 정의
- [x] ADE20K → Binary Floor Mask 전처리 코드 구성
- [ ] 전체 데이터셋 검증 스크립트 실행 결과 기록
- [ ] Dataset/DataLoader 테스트 성공 결과 기록
- [ ] RGB / GT Mask / Overlay 시각화
- [ ] LR-ASPP + MobileNetV3 모델 구현 및 GPU Forward Pass
- [ ] Training / Validation / Checkpoint 저장
- [ ] Evaluation 및 오류 분석
- [ ] 실제 JETANK 데이터 확보와 Fine-tuning
- [ ] Camera BEV Costmap 및 ROS 2/Nav2 통합
- [ ] Camera–LiDAR Adaptive Fusion 및 실시간성 검증

> **상태 표기 원칙:** 코드 작성·데이터 전처리 수행 사실과, 정상 실행·성능 검증 완료 사실은 구분합니다. 위 체크박스는 결과 로그를 확인한 뒤 업데이트하세요.

## References

- ADE20K Scene Parsing Benchmark: https://sceneparsing.csail.mit.edu/
- ADE20K dataset information: https://ade20k.csail.mit.edu/
- PyTorch: https://pytorch.org/
- Torchvision segmentation models: https://pytorch.org/vision/stable/models.html
- ROS 2 documentation: https://docs.ros.org/
- Navigation2: https://docs.nav2.org/
