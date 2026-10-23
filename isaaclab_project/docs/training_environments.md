# 로봇 학습 환경

현재 실행기는 `run_cad6.ps1`이며 MG996R hip 6개 + RC920DMG 12개의 Onshape CAD6를 사용합니다.
정책은 18개 관절 절대각을 직접 출력하고 새 학습은 기본 `revised_v3` behavior를 사용합니다.
[직접 각도 학습 안내](joint_angle_training.md)에서 출력 계약·새 체크포인트·평가 기준을 확인하세요.
과거 MG90S task·실행기와 해당 v0 gait 학습 경로는 제거됐습니다.

실행기의 `-Environment`로 환경을 선택하며 기본값은 `flat`입니다.
아래 명령은 현재 혼합 서보 CAD6 구성에 해당합니다.

| 선택값 | 환경 | 난이도를 높이면 |
| --- | --- | --- |
| `flat` | 평지에서 무작위 방향·거리의 목표로 이동 | 장애물 없음 |
| `narrow` | 로봇 폭에 맞춘 좁은 통로와 낮은 천장 통과 | 벽 사이와 천장 아래 여유 공간 감소 |
| `vibrating` | 지속적으로 움직이는 바닥에서 균형을 잡으며 통과 | 진동 강도 증가 |
| `falling_debris` | 물리 충돌이 있는 낙하물이 반복해서 떨어지는 위험구역 통과 | 낙하물 회피 부담 증가 |
| `rough` | 통로를 가로지르는 세 개의 요철·단차 통과 | 단차 높이 증가 |
| `mixed` | 좁은 통로, 진동 바닥, 낙하물을 함께 사용 | 통로 폭·진동·낙하물 난이도가 함께 증가 |
| `stairs` | 세 연속 상승 단과 위쪽 목표 플랫폼 | 로봇 높이의 2.5~12.5%로 단 높이 증가 |

비평지 환경은 시작 지점에서 반대편 목표로 이동하는 통과 과제입니다. 측벽과 영역 이탈 판정으로
장애물을 통로 밖으로 돌아서 피하는 행동을 제한합니다. `mixed`는 낙하물이 로봇까지 내려오도록
천장을 열어 둡니다. 좁은 틈의 기준은 몸통만이 아니라 다리를 포함한 로봇 폭입니다.

## 화면에서 확인

`check -Gui`는 체크포인트 없이 환경 생성·스텝·리셋을 화면에서 확인합니다.
진동과 낙하물은 시뮬레이션이 진행되는 동안 작동하며 학습된 정책의 보행 검증은 아닙니다.

```powershell
.\run_cad6.ps1 -Mode check -Gui -Environment narrow -Difficulty 0.8 -NumEnvs 1
.\run_cad6.ps1 -Mode check -Gui -Environment vibrating -NumEnvs 1
.\run_cad6.ps1 -Mode check -Gui -Environment falling_debris -NumEnvs 1 -Seed 42
.\run_cad6.ps1 -Mode check -Gui -Environment rough -NumEnvs 1
.\run_cad6.ps1 -Mode check -Gui -Environment mixed -NumEnvs 1
```

기본 검사는 1000스텝(시뮬레이션 시간 20초)입니다. `-Steps`로 스텝 수를 지정할 수 있습니다.

## 학습·평가·재생

`-Difficulty` 범위는 0~1이고 기본값은 0.5입니다. 0도 해당 환경의 장애물이 있는 쉬운 설정이며,
장애물 없이 학습하려면 `flat`을 선택합니다. 난이도는 실행할 때 고정하며 자동으로 상승하지 않습니다.
`-Seed`는 0~2147483647 범위의 정수입니다. 학습 기본값은 42, 독립 평가 기본값은 73입니다.

```powershell
# 평지에서 먼저 학습하고 독립 시드로 순전진을 평가
.\run_cad6.ps1 -Mode train -Environment flat -Seed 42 -NumEnvs 32 -MaxIterations 1000 -AllowProvisionalCad
.\run_cad6.ps1 -Mode evaluate -Environment flat -Seed 73 -NumEnvs 4 -Steps 12000 -Checkpoint '<평지 checkpoint>'

# 같은 관측·행동 계약의 평지 정책에서 복합 환경으로 추가 학습
.\run_cad6.ps1 -Mode train -Environment mixed -Difficulty 0.6 -NumEnvs 32 -Checkpoint '<평지 checkpoint>' -AllowProvisionalCad

# 해당 환경에서 학습한 체크포인트를 평가하고 재생
.\run_cad6.ps1 -Mode evaluate -Environment mixed -Difficulty 0.6 -NumEnvs 4 -Steps 12000 -Checkpoint '<복합 환경 checkpoint>'
.\run_cad6.ps1 -Mode play -Environment mixed -Difficulty 0.6 -NumEnvs 1 -Checkpoint '<복합 환경 checkpoint>'
```

## 체크포인트와 관측 호환성

| 로봇 | 평지 로그 폴더 | 비평지 로그 폴더 | 정책 / 가치함수 관측 | 행동 |
| --- | --- | --- | --- | --- |
| 혼합 서보 CAD6 | `dksh_mg996r_hip_cad6_jointangles_v2` | `dksh_mg996r_hip_cad6_jointangles_v2_obstacles` | 140 / 140 | 18개 절대각 |

로그 폴더는 모두 `logs/rsl_rl/` 아래에 생성됩니다. 같은 로봇 모델의 모든 환경은 동일한
정책 입력 차원을 사용합니다. 환경별 동작 성능은 각각 평가해야 합니다.

정책 입력은 몸통·목표·관절·적용 목표·발 접촉 등 92개 값과 지형 48개 값입니다.
지형 입력 중 L2 장애물 인코더가 만드는 부분은 다음 32개입니다.

| 개수 | 관측값 |
| --- | --- |
| 16 | L2 프레임을 수평 16개 구역으로 묶은 최소 장애물 거리 ÷ 30 m. 미검출은 1.0 |
| 16 | 각 구역의 유효 측정 여부. 검출 1.0, 미검출 0.0 |

나머지 16개는 주변 높이 9개, 바닥 속도 3개, 최근접 낙하물 상태 4개이며 정책과 가치함수가
모두 사용합니다. 높이는 로봇에 대한 상대 높이이므로 평지에서도 0으로 채우지 않습니다.
현재 시뮬레이터에서 제공하는 값이며 실기에는 같은 정보를 만드는 추정기가 필요합니다.

LiDAR 설정은 Unitree L2의 30 m 최대 거리(@90% 반사율), 15 m(@10%), 0.05 m 근거리 한계,
360°×96° NEGA FOV, 수평 5.55 Hz/수직 216 Hz 스캔, 128,000 samples/s와 64,000 points/s 유효 출력,
내장 IMU 1 kHz 샘플링/500 Hz 출력, 2 cm 정확도 및 4.5 mm 거리 분해능을 기록합니다.
시뮬레이터 입력에는 수평 5.55 Hz sample-and-hold, ±2 cm 균등 오차와 4.5 mm 양자화를 적용합니다.
고도 범위는 -6°~90°이며 아래쪽으로 확장된 6° 영역의 실제 최대 거리는 더 짧을 수 있습니다.
수치와 좌표계는 [Unitree L2 사용자 설명서](https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf)와
[Unitree SDK2 좌표계 설명](https://github.com/unitreerobotics/unilidar_sdk2)을 기준으로 했습니다.

L2 본체의 공칭 질량 230 g과 75×75×65 mm 외형은 베이스의 질량·무게중심·관성에 합산하고,
같은 크기의 충돌 형상을 베이스에 부착해 천장과 낙하물이 센서 본체를 통과하지 않게 합니다.
포인트클라우드 좌표 원점은 매뉴얼과 같이 본체 바닥 중심을 사용합니다. 실제 브래킷 치수가 아직
없으므로 베이스 좌표계의 `(0, 0, 0.030 m)` 장착 위치는 임시값이며 실측 후 교체해야 합니다.

병렬 학습 비용을 제한하기 위해 전체 점군을 정책에 직접 넣지 않습니다. 시뮬레이터는 각 수평
구역에서 방위각 5개 × 고도각 5개(-6°, 0°, 30°, 60°, 90°) 광선을 쏴 최근접 거리를 사용합니다.
고정 구조물은 AABB, 낙하물은 구로 계산합니다. 실제 점군은 같은 구역에서 모든 전처리된 점의
최근접 거리를 사용합니다. 이 광선 근사에는 L2 고유의 비균일 스캔 패턴·음각 영역의 거리 감소·
반사율별 누락·재질·날씨 효과가 없습니다. 명시된 15 m 저반사율 범위, 점 출력률, 수직 스캔 주파수,
IMU 주파수는 기기 메타데이터이며 현재 학습 입력 생성에 직접 적용되지 않습니다.

과거 L1 탐색·MG90S·v0/v1 체크포인트는 현재 관측·행동·모터 계약과 호환되지 않습니다.
현재 작업에서 생성한 체크포인트와 metadata를 함께 사용하며 보상·접촉 behavior도 확인합니다.
파일의 이름이나 관측 차원만으로 호환성을 판단하지 않습니다.

Unity `LidarScene`과 새 센서의 기본값도 L2 기준입니다. 360×32 광선, 5.55 Hz로 약 63,936 rays/s를
사용하고 고도 -6°~90°에 수평 고도 0°를 포함합니다. ROS 브리지는 수평 광선을 선택해 `/scan`을
발행합니다. Unity 메뉴 `DKSH/Spiderbot/Sensors/Run L2 Verification`에서 시야·수평 투영·양자화와
저장된 씬 설정을 확인할 수 있습니다.

## 실제 L2 점군 입력

`scripts/l2_pointcloud_features.py`의 `encode_l2_obstacles(points_lidar)`는 **완성된 한 프레임**의
장애물 점 `(x, y, z)`를 받아 지형 입력의 첫 32개 값(거리 16개, 유효값 16개)을 반환합니다.
좌표는 L2 본체 바닥 중심 원점, 센서 +X는 케이블 반대 방향, +Y는 +X에서 반시계 90°, +Z는 위입니다.
구역 0은 +X 중심이고 번호가 +Y 방향으로 증가하며 각 구역 폭은 22.5°입니다. 바로 세운 L2의
NEGA 고도 -6~90°와 거리 `[0.05, 30)` m 안의 최근접 점을 4.5 mm로 양자화해 30 m로 나눕니다.
점이 없으면 `(거리, 유효값)=(1, 0)`입니다. 장착 방향과 실제 점군의 고도 분포는 실측으로 확인해야 합니다.

호출 전에 로봇 자체와 바닥의 점을 제거하고 센서 좌표계로 변환해야 합니다. 특히 바닥 제거에는
로봇 자세와 실제 바닥 형상이 필요하며 이 모듈은 이를 추정하지 않습니다. 포인트를 여러 프레임에
누적하거나 패킷 일부만 넣으면 학습 관측과 달라집니다. 정책 50 Hz 사이에는 가장 최근 완성된
L2 프레임을 유지합니다. 이 인코더만으로 140개 관측이 완성되지는 않습니다.
나머지 몸통·관절·적용 목표각과 속도·발 접촉·높이·위험 정보는 학습과 같은 순서와 단위로
별도 공급해야 합니다. 현재 시뮬레이터는 L2 축이 몸체 축과 나란하다고 가정합니다.
`lidar_mount_position_b=(0, 0, 0.030)`은 임시 장착값이므로 위치와 축 정렬을 실측 후 수정합니다.

```powershell
# points.json: 한 프레임에서 로봇/바닥을 제거한 [[x,y,z], ...] (미터, L2 좌표계)
python .\isaaclab_project\scripts\l2_pointcloud_features.py .\points.json
```

실제 L2 로그와 동일 장면의 시뮬레이션 출력을 비교해 구역별 거리·유효율·스캔 지연을 보정해야
실기 입력 일치를 확인할 수 있습니다. 현재는 기하 근사와 입력 형식까지만 구현됐으며 실측 검증은
아직 하지 않았습니다.

하드웨어 launch는 같은 인코더를 사용하는 ROS2 `l2_policy_features` 실행 파일을 함께 시작하고,
39개 수직 스캔을 묶은 `/unilidar/cloud` 프레임마다 `/policy/l2_features` (`Float32MultiArray`, 32개)를
발행합니다. 공통 패키지 이름 `dksh_l1_features`와 이전 L1 오프라인 인코더는 호환용으로 유지합니다.
프레임 ID가 `unilidar_lidar`가 아니면 좌표계 혼용을 막기 위해 해당 프레임을 거부합니다. L2는
음각 구간도 사용하므로 수평 장착 상태에서도 들어오는 바닥점과 로봇 자체의 점은 장착 실측 후
별도 전처리로 제거해야 합니다. 원시 점군을 그대로 정책에 적용하는 실기 검증은 아직 완료되지 않았습니다.

2026-10-10 당시 센서 코드의 로컬 검증 기록: Python 단위 테스트 103개 중 102개 통과, CUDA 비교 1개는 NVIDIA GPU가
없어 생략했습니다. Unity 6000.5.3f1에서 C# 컴파일과 L2 음각 광선·수평 장애물 검출·양자화·씬
설정 검증 및 기존 Frontier 검증을 통과했습니다. UDP/UART Compose 설정과 PowerShell 문법도
검사했습니다. 이 PC는 RAM 16 GB·Intel Arc 내장 GPU이며 Docker daemon이 실행 중이지 않고
C: 여유 공간이 이미지 빌드 기준 25 GB 미만이어서 이미지 빌드는 수행하지 않았습니다.
이 기록은 현재 절대 관절각 정책의 보행 검증 결과가 아닙니다. Isaac Sim 실행, Jazzy 이미지 빌드 및
실제 L2 점군 수신은 각 구성에서 별도로 검증해야 합니다.

## 스모크 검사와 Python 실행

스모크 검사는 학습 없이 환경 생성, 스텝 진행, 리셋 등을 확인합니다.

```powershell
.\run_cad6.ps1 -Mode check -Environment narrow -NumEnvs 4 -Steps 100
.\run_cad6.ps1 -Mode check -Environment vibrating -NumEnvs 4 -Steps 100
.\run_cad6.ps1 -Mode check -Environment falling_debris -NumEnvs 4 -Steps 200 -Seed 42
.\run_cad6.ps1 -Mode check -Environment mixed -Difficulty 1 -NumEnvs 4 -Steps 200
```

Python 스크립트를 직접 호출할 때는 `--environment`, `--difficulty`, `--seed`를 사용합니다.
현재 직접 각도 실행기는 `joint_angle_runner.py`입니다.

```powershell
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\joint_angle_runner.py --mode=train --task=Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2 --environment=flat --seed=42 --num_envs=32 --headless --allow_provisional_cad
```

같은 시드를 지정하면 같은 무작위 환경 설정으로 실행할 수 있습니다.

## CAD6 계단 학습

현재 계단도 동일한 140개 관측·18개 절대각 계약을 사용합니다. 평지 순전진을 먼저 평가한 뒤
`run_cad6.ps1 -Environment stairs -Difficulty 0`으로 시작하며, 구체적인 전환·평가 명령은
[직접 각도 학습 안내](joint_angle_training.md)를 따릅니다.
[과거 계단 센서 학습 기록](stair_sensor_training.md)의 240개 관측·20~120 mm curriculum과
평가 수치는 제거된 v0 작업의 역사 기록이며 현재 계단 정책의 검증으로 사용하지 않습니다.
