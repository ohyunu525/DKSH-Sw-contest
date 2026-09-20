# 로봇 학습 환경

Isaac Lab 실행기의 `-Environment`로 환경을 선택합니다. 지정하지 않으면 기존 `flat` 환경을 사용합니다.
모든 환경은 기본 로봇과 `-RobotModel cad8`, `-RobotModel cad6`에서 선택할 수 있습니다.

| 선택값 | 환경 | 난이도를 높이면 |
| --- | --- | --- |
| `flat` | 평지에서 무작위 방향·거리의 목표로 이동 | 장애물 없음 |
| `narrow` | 로봇 폭에 맞춘 좁은 통로와 낮은 천장 통과 | 벽 사이와 천장 아래 여유 공간 감소 |
| `vibrating` | 지속적으로 움직이는 바닥에서 균형을 잡으며 통과 | 진동 강도 증가 |
| `falling_debris` | 물리 충돌이 있는 낙하물이 반복해서 떨어지는 위험구역 통과 | 낙하물 회피 부담 증가 |
| `rough` | 통로를 가로지르는 세 개의 요철·단차 통과 | 단차 높이 증가 |
| `mixed` | 좁은 통로, 진동 바닥, 낙하물을 함께 사용 | 통로 폭·진동·낙하물 난이도가 함께 증가 |

비평지 환경은 시작 지점에서 반대편 목표로 이동하는 통과 과제입니다. 측벽과 영역 이탈 판정으로
장애물을 통로 밖으로 돌아서 피하는 행동을 제한합니다. `mixed`는 낙하물이 로봇까지 내려오도록
천장을 열어 둡니다. 좁은 틈의 기준은 몸통만이 아니라 다리를 포함한 로봇 폭입니다.

## 화면에서 확인

`preview`는 체크포인트 없이 기본 자세의 로봇과 선택한 환경을 보여 줍니다. 진동과 낙하물은
시뮬레이션이 진행되는 동안 작동합니다. 이 모드의 로봇은 학습된 정책으로 걷지 않습니다.

```powershell
.\run_isaaclab.ps1 -Mode preview -Environment narrow -Difficulty 0.8 -NumEnvs 1
.\run_isaaclab.ps1 -Mode preview -Environment vibrating -NumEnvs 1
.\run_isaaclab.ps1 -Mode preview -Environment falling_debris -NumEnvs 1 -Seed 42
.\run_isaaclab.ps1 -Mode preview -Environment rough -RobotModel cad6 -NumEnvs 1
.\run_isaaclab.ps1 -Mode preview -Environment mixed -NumEnvs 1
```

기본 미리보기는 3000스텝(시뮬레이션 시간 60초)입니다. `-Steps`로 스텝 수를 지정할 수 있으며,
창을 닫으면 미리보기가 끝납니다.

## 학습·평가·재생

`-Difficulty` 범위는 0~1이고 기본값은 0.5입니다. 0도 해당 환경의 장애물이 있는 쉬운 설정이며,
장애물 없이 학습하려면 `flat`을 선택합니다. 난이도는 실행할 때 고정하며 자동으로 상승하지 않습니다.
`-Seed`는 0~2147483647 범위의 정수이고, 생략 시 기존 설정의 시드를 유지합니다.

```powershell
# 쉬운 낙하물 환경부터 학습
.\run_isaaclab.ps1 -Mode train -Environment falling_debris -Difficulty 0.2 -Seed 42 -NumEnvs 32 -MaxIterations 1000

# 좁은 틈이 포함된 복합 환경 학습
.\run_isaaclab.ps1 -Mode train -Environment mixed -Difficulty 0.6 -RobotModel cad6 -NumEnvs 32

# 같은 로봇의 최신 비평지 체크포인트를 평가하고 재생
.\run_isaaclab.ps1 -Mode evaluate -Environment mixed -RobotModel cad6 -Difficulty 0.6 -Steps 1000
.\run_isaaclab.ps1 -Mode play -Environment narrow -RobotModel cad6 -NumEnvs 1

# 특정 체크포인트 선택
.\run_isaaclab.ps1 -Mode play -Environment vibrating -NumEnvs 1 -Checkpoint "C:\path\to\model_1000.pt"
```

## 체크포인트와 관측 호환성

| 로봇 | 평지 로그 폴더 | 비평지 로그 폴더 | 정책 / 가치함수 관측 | 행동 |
| --- | --- | --- | --- | --- |
| `baseline` | `dksh_spider_navigation_l1v2` | `dksh_spider_navigation_l1v2_obstacles` | 116 / 132 | 24 |
| `cad8` | `dksh_spider_cad8_navigation_l1v2` | `dksh_spider_cad8_navigation_l1v2_obstacles` | 116 / 132 | 24 |
| `cad6` | `dksh_spider_cad6_navigation_l1v2` | `dksh_spider_cad6_navigation_l1v2_obstacles` | 98 / 114 | 18 |

로그 폴더는 모두 `logs/rsl_rl/` 아래에 생성됩니다. 같은 로봇 모델의 모든 환경은 동일한
정책 입력 차원을 사용합니다. 환경별 동작 성능은 각각 평가해야 합니다.

정책 입력은 기존 로봇 상태(기본/CAD8 84개, CAD6 66개) 뒤에 다음 32개 값을 붙입니다.

| 개수 | 관측값 |
| --- | --- |
| 16 | L1 RM 프레임을 수평 16개 구역으로 묶은 최소 장애물 거리 ÷ 30 m. 미검출은 1.0 |
| 16 | 각 구역의 유효 측정 여부. 검출 1.0, 미검출 0.0 |

가치함수에는 정책 입력 뒤에 시뮬레이터 전용 정보 16개(주변 높이 9개, 바닥 속도 3개,
최근접 낙하물 상태 4개)를 추가합니다. 평지에서는 이 값이 0입니다. 이 값은 정책에
들어가지 않으므로 실제 로봇이 별도로 측정할 필요가 없습니다.

LiDAR 설정은 L1 RM의 30 m 최대 거리(@90% 반사율), 15 m(@10%), 0.05 m 근거리 한계,
360°×90° FOV, 수평 11 Hz/수직 180 Hz 스캔, 43,200 samples/s와 21,600 points/s 유효 출력,
내장 IMU 1 kHz 샘플링/250 Hz 출력, ±2 cm 정확도 및 8 mm 거리 분해능을 기록합니다.
시뮬레이터 입력에는 수평 11 Hz sample-and-hold, ±2 cm 균등 오차와 8 mm 양자화를 적용합니다.
수치와 좌표계는 [Unitree L1 사용자 설명서](https://oss-global-cdn.unitree.com/static/52b72f707b304d229d4321eea223738f.pdf)와
[Unitree SDK 좌표계 설명](https://github.com/unitreerobotics/unilidar_sdk)을 기준으로 했습니다.

L1 본체의 실측 질량 230 g과 75×75×65 mm 외형은 베이스의 질량·무게중심·관성에 합산하고,
같은 크기의 충돌 형상을 베이스에 부착해 천장과 낙하물이 센서 본체를 통과하지 않게 합니다.
포인트클라우드 좌표 원점은 매뉴얼과 같이 본체 바닥 중심을 사용합니다. 실제 브래킷 치수가 아직
없으므로 베이스 좌표계의 `(0, 0, 0.030 m)` 장착 위치는 임시값이며 실측 후 교체해야 합니다.

병렬 학습 비용을 제한하기 위해 전체 점군을 정책에 직접 넣지 않습니다. 시뮬레이터는 각 수평
구역에서 방위각 5개 × 고도각 3개 광선을 쏴 최근접 거리를 사용합니다. 고정 구조물은 AABB,
낙하물은 구로 계산합니다. 실제 점군은 같은 구역에서 모든 전처리된 점의 최근접 거리를 사용합니다.
이 광선 근사에는 L1 고유의 비균일 스캔 패턴·반사율별 누락·재질·날씨 효과가 없습니다.
명시된 15 m 저반사율 범위, 점 출력률, 수직 스캔 주파수, IMU 주파수는 기기 메타데이터이며
현재 학습 입력 생성에 직접 적용되지 않습니다.

이전 체크포인트는 입력 차원이나 입력 의미가 달라 새 환경과 호환되지 않습니다. 같은 116차원인
이전 비평지 정책도 내부 값 순서가 달라 사용할 수 없습니다. 새 `*_l1v2*` 폴더에서 다시 학습하고,
명시적 `-Checkpoint`에도 새 정책만 지정해야 합니다. 제공된 `balance_baseline.pt`는 자동 사용하지 않습니다.

## 실제 L1 RM 점군 입력

`scripts/l1_pointcloud_features.py`의 `encode_l1_obstacles(points_lidar)`는 **완성된 한 프레임**의
장애물 점 `(x, y, z)`를 받아 정책 입력 끝의 32개 값(거리 16개, 유효값 16개)을 반환합니다.
좌표는 L1 본체 바닥 중심 원점, 센서 +X는 케이블 반대 방향, +Y는 +X에서 반시계 90°, +Z는 위입니다.
구역 0은 +X 중심이고 번호가 +Y 방향으로 증가하며 각 구역 폭은 22.5°입니다. 바로 세운 L1의
상반구 고도 0~90°와 거리 `[0.05, 30)` m 안의 최근접 점을 8 mm로 양자화해 30 m로 나눕니다. 점이 없으면
`(거리, 유효값)=(1, 0)`입니다.
고도 0~90°는 설명서의 "센서 위쪽 반구"를 적용한 모델 가정이며, 장착 방향과 실제 점군의
고도 분포로 확인해야 합니다.

호출 전에 로봇 자체와 바닥의 점을 제거하고 센서 좌표계로 변환해야 합니다. 특히 바닥 제거에는
로봇 자세와 실제 바닥 형상이 필요하며 이 모듈은 이를 추정하지 않습니다. 포인트를 여러 프레임에
누적하거나 패킷 일부만 넣으면 학습 관측과 달라집니다. 정책 50 Hz 사이에는 가장 최근 완성된
L1 프레임을 유지합니다. 나머지 정책 입력(속도·자세·관절·목표·직전 행동)은 동일한 순서와 단위로
별도 공급해야 합니다. 현재 시뮬레이터는 L1 축이 몸체 축과 나란하다고 가정합니다.
`lidar_mount_position_b=(0, 0, 0.030)`은 임시 장착값이므로 위치와 축 정렬을 실측 후 수정합니다.

```powershell
# points.json: 한 프레임에서 로봇/바닥을 제거한 [[x,y,z], ...] (미터, L1 좌표계)
python .\isaaclab_project\scripts\l1_pointcloud_features.py .\points.json
```

실제 L1 로그와 동일 장면의 시뮬레이션 출력을 비교해 구역별 거리·유효율·스캔 지연을 보정해야
실기 입력 일치를 확인할 수 있습니다. 현재는 기하 근사와 입력 형식까지만 구현됐으며 실측 검증은
아직 하지 않았습니다.

## 스모크 검사와 Python 실행

스모크 검사는 학습 없이 환경 생성, 스텝 진행, 리셋 등을 확인합니다.

```powershell
.\run_isaaclab.ps1 -Mode smoke -Environment narrow -NumEnvs 4 -Steps 100
.\run_isaaclab.ps1 -Mode smoke -Environment vibrating -NumEnvs 4 -Steps 100
.\run_isaaclab.ps1 -Mode smoke -Environment falling_debris -NumEnvs 4 -Steps 200 -Seed 42
.\run_isaaclab.ps1 -Mode smoke -Environment mixed -Difficulty 1 -NumEnvs 4 -Steps 200
```

Python 스크립트를 직접 호출할 때는 `--environment`, `--difficulty`, `--seed`를 사용합니다.
학습·재생 래퍼는 `env.environment_preset=...`, `env.environment_difficulty=...` 표기도 지원합니다.

```powershell
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\train.py --task=Isaac-DKSH-Spider-Navigation-Direct-v0 --environment=mixed --difficulty=0.5 --seed=42 --num_envs=32 --headless
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\preview_env.py --environment=falling_debris --num_envs=1
```

같은 시드를 지정하면 같은 무작위 환경 설정으로 실행할 수 있습니다.
