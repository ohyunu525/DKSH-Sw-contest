# 개발 컴퓨터 요구사항 및 작업 분담

이 문서는 동일 저장소를 여러 컴퓨터에서 안전하게 작업하기 위한 기준이다. 컴퓨터 이름이 아니라
실제 RAM, GPU, 저장공간과 수행할 작업을 기준으로 등급을 선택한다.

## 1. 프로젝트 고정 버전

| 항목 | 고정 기준 |
|---|---|
| 운영체제 | Windows 10/11 64-bit |
| Unity | `6000.5.3f1` (`ProjectVersion.txt` 기준) |
| Isaac Sim | standalone `4.5.0` |
| Isaac Lab | 저장소 submodule commit `bc7c9f5c7c2a9f6fd6c69a6bdbfaace19b4204be` |
| Isaac Python | Isaac Sim 4.5 번들 Python 3.10 |
| ROS2 | Docker의 ROS2 Jazzy / Ubuntu 24.04 기반 이미지 |
| Unity–ROS | ROS-TCP-Connector `v0.7.0`, ROS-TCP-Endpoint `ROS2v0.7.0` |

Unity, Isaac Lab 또는 ROS 패키지를 각 컴퓨터에서 임의의 최신 버전으로 바꾸지 않는다. 버전 변경은
한 컴퓨터에서 호환성 검증 후 저장소 전체에 반영한다.

## 2. 컴퓨터 등급

### A — Isaac/RL 워크스테이션

지정 장비: **RTX 3070 8GB VRAM, RAM 32GB** 컴퓨터.

NVIDIA의 Isaac Sim 4.5 공식 최소 요구사항은 RTX 3070, 8GB VRAM, RAM 32GB, 4코어 CPU와
50GB SSD이다. 따라서 이 장비는 Isaac Sim 4.5를 실행할 수 있지만 공식 최소 경계에 해당하며,
대규모 환경이나 고해상도 렌더링에 충분한 여유가 있다고 간주하지 않는다.

프로젝트 요구사항:

- RTX 3070 8GB 이상
- RAM 32GB 이상
- CPU 4코어는 공식 최소값이며, 프로젝트에서는 8코어 이상 권장
- SSD 여유 공간 최소 50GB, 프로젝트 권장 100GB 이상
- Windows NVIDIA 드라이버 최소 537.58
- 설치 전 Isaac Sim 4.5 Compatibility Checker 통과
- 오프라인 asset pack은 약 86GB가 추가되므로 필요한 경우에만 설치

담당 작업:

- Isaac Sim 4.5 및 Isaac Lab 실행
- 6족/8족 보행 학습, 평가, 정책 export
- URDF → USD 재생성
- 험지·외란·Domain Randomization 실험
- Unity와 ROS2 통합의 최종 성능 시험

운영 규칙:

- 최초 검증은 `-NumEnvs 4`로 시작한다.
- 4개 환경이 안정적일 때만 8 → 16 → 32 순으로 증가시킨다.
- 이 저장소에서는 RTX 3070으로 32개 환경을 검증했지만, 장면과 관측 차원이 바뀌면 다시
  4개부터 확인한다.
- 학습은 가능하면 headless로 실행한다.
- Isaac Sim 실행 중에는 Unity Editor와 전체 Nav2 stack을 종료한다.
- GUI playback 또는 USD 변환 중에도 다른 GPU 작업을 동시에 실행하지 않는다.

공식 기준: [Isaac Sim 4.5 요구사항](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html),
[Isaac Sim 4.5 다운로드 크기](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html),
[Isaac Lab 2.1.0 binary 설치](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html)

### B — Unity/ROS 통합 개발기

현재 확인된 장비: **Intel Core Ultra 5 125H, RAM 16GB, Intel Arc 내장 GPU** 컴퓨터.

권장 기준:

- RAM 16GB 이상
- 최근 4코어 이상 x64 CPU
- Unity 프로젝트와 Docker 이미지를 위한 SSD 여유 공간 30GB 이상
- Docker Desktop WSL2 backend
- Unity `6000.5.3f1`

담당 작업:

- Unity 재난 환경, LiDAR, Occupancy Grid, Frontier, 최소 A* 개발
- Unity standalone 탐색 검증
- Unity + headless ROS2/Nav2 연동
- C# 편집 및 Unity Edit Mode 테스트
- ROS2 설정, launch, Frontier 후보 검증

제한:

- Isaac Sim/Isaac Lab은 설치하거나 실행하지 않는다.
- Unity와 headless ROS2/Nav2까지만 동시에 실행한다.
- ROS 컨테이너는 사용하지 않을 때 `.\ros2\manage.ps1 Stop`으로 정지한다.
- ROS 이미지 빌드는 C: 여유 공간이 25GB 미만이면 실행하지 않는다. 관리 스크립트도 이를 차단한다.

현재 측정값은 `ros2/README.md`에 기록한다. 측정 당시 전체 headless SLAM/Nav2 stack은 약
307MiB RAM을 사용했지만, 실제 SLAM 지도가 커질 때의 최대 사용량으로 해석하지 않는다.

### C — 경량 개발기

판정 기준 예시:

- RAM 8~15GB
- NVIDIA RTX GPU 없음
- SSD 여유 공간 10~29GB

허용 작업:

- 소스 코드와 문서 편집
- Git branch 병합 및 코드 리뷰
- Isaac Sim을 import하지 않는 Python 단위 테스트
- Unity를 실행하지 않는 정적 C# 검토
- 설정/YAML/JSON 검증

조건부 작업:

- RAM 12GB 이상이고 Unity가 안정적일 때만 Unity Editor 하나를 단독 실행한다.
- 이 경우 Game View 해상도와 품질을 낮추고 Docker Desktop은 종료한다.

금지 작업:

- Isaac Sim/Isaac Lab 실행
- ROS Docker 이미지 로컬 빌드
- Unity와 Docker 동시 실행
- 대규모 로그, checkpoint, Library 폴더 복사

실행 검증이 필요하면 변경을 Git으로 전달하고 A 또는 B 등급 컴퓨터에서 수행한다.

### D — 검토 전용 컴퓨터

다음 중 하나에 해당하면 검토 전용으로 취급한다.

- RAM 8GB 미만
- 저장공간 여유 10GB 미만
- Unity 6를 안정적으로 실행할 수 없는 GPU/드라이버

문서, 이슈, 작은 소스 수정과 코드 리뷰만 수행한다. Unity, Docker, Isaac Sim은 실행하지 않으며
빌드·학습·시뮬레이션 결과는 A/B 등급 컴퓨터에서 생성한다.

## 3. 동시 실행 허용표

| 작업 조합 | A 등급 | B 등급 | C 등급 | D 등급 |
|---|---:|---:|---:|---:|
| Unity standalone | 가능 | 가능 | 조건부 단독 | 불가 |
| Unity + headless ROS2/Nav2 | 가능 | 가능 | 불가 | 불가 |
| Isaac Lab headless 학습 | 가능, 단독 권장 | 불가 | 불가 | 불가 |
| Isaac Sim GUI/playback | 가능, 단독 | 불가 | 불가 | 불가 |
| Isaac Sim + Unity + Nav2 동시 실행 | 금지 | 금지 | 금지 | 금지 |
| 소스/문서/정적 테스트 | 가능 | 가능 | 가능 | 가능 |

## 4. 새 컴퓨터 공통 준비

1. Git으로 저장소를 clone한다.
2. 저장소 루트에서 `git submodule update --init --recursive`를 실행한다.
3. Unity Hub에서 정확히 `6000.5.3f1`을 설치하고 `DKSH SW Project`를 연다.
4. Unity가 `Packages/manifest.json`과 `packages-lock.json`의 패키지를 복원하고 컴파일을 마칠 때까지
   기다린다.
5. `Library`, `Temp`, `Obj`, `Logs`, `UserSettings`, `.venv`, `IsaacLab/_isaac_sim`은 컴퓨터 사이에
   복사하지 않는다.
6. Unity asset을 추가할 때 `.meta` 파일도 함께 Git에 저장한다.

Windows의 긴 경로 문제로 submodule checkout이 실패하는 컴퓨터에서는 Git의 long path 지원을
활성화한 뒤 다시 clone한다. 개인 경로나 드라이버 경로는 프로젝트 파일에 고정하지 않고 실행 인자로
전달한다.

## 5. A 등급 컴퓨터 설치 및 검증

Isaac Sim standalone 4.5.0을 설치한 뒤 기본 경로를 사용하거나 실제 설치 경로를 전달한다.

```powershell
git submodule update --init --recursive
.\setup_isaaclab.ps1
.\run_isaaclab.ps1 -Mode isaac-smoke -NumEnvs 4
.\run_isaaclab.ps1 -Mode smoke -RobotModel cad6 -NumEnvs 4 -Steps 64
```

기본 경로가 아닌 경우:

```powershell
.\setup_isaaclab.ps1 -IsaacSimPath "D:\isaac-sim-standalone-4.5.0-windows-x86_64"
```

스모크 테스트가 통과한 뒤에만 학습 환경 수를 증가시킨다.

```powershell
.\run_isaaclab.ps1 -Mode train -RobotModel cad6 -Environment rough -NumEnvs 8 -MaxIterations 1000
```

## 6. B 등급 컴퓨터 ROS2 검증

Docker Desktop을 시작한 후:

```powershell
.\ros2\manage.ps1 Build
.\ros2\manage.ps1 Start
.\ros2\manage.ps1 Validate
```

작업 종료 후:

```powershell
.\ros2\manage.ps1 Stop
```

Docker image와 WSL 데이터는 Git으로 공유하지 않는다. 다른 컴퓨터에서는 동일 Dockerfile로 다시
빌드한다.

## 7. 컴퓨터 간 결과 전달 규칙

- Unity: `Assets`, `Packages`, `ProjectSettings`와 `.meta`만 Git으로 전달한다.
- Isaac: 환경 코드, URDF/USD 원본, 설정과 선별된 정책 파일만 전달한다.
- `logs/`, `outputs/`, Unity `Library/`와 전체 Isaac cache는 전달하지 않는다.
- 대형 학습 로그와 중간 checkpoint는 Git에 넣지 않고 별도 저장소에 보관한다.
- 최종 정책을 저장소에 추가할 때는 로봇 모델, 환경 preset, seed, 학습 iteration과 평가 결과를 함께
  기록한다.
- ROS2: Docker image가 아니라 `ros2/Dockerfile`, compose, launch와 YAML 설정을 전달한다.
- 컴퓨터별 절대 경로, GPU 번호와 개인 설정을 공용 설정 파일에 저장하지 않는다.

## 8. 최소 인수 기준

| 변경 종류 | 최종 검증 컴퓨터 | 필수 확인 |
|---|---|---|
| 문서/설정만 변경 | C 이상 | JSON/YAML/링크와 diff 확인 |
| Unity Frontier/A* 변경 | B 이상 | Unity 컴파일 + Frontier Verification |
| Unity–ROS 메시지/TF 변경 | B 이상 | Endpoint health + `/scan`·`/odom`·`/tf`·`/cmd_vel` 왕복 |
| Isaac 환경/보상 변경 | A | 4 env smoke + 유한 step evaluate |
| 6족 정책 또는 USD 변경 | A | cad6 smoke + playback/evaluate + artifact 기록 |
| 전체 시스템 변경 | A와 B | 각 전용 검증 후 Unity–ROS 폐루프 검사 |
