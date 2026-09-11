# 로봇 학습 환경

Isaac Lab 실행기의 `-Environment`로 환경을 선택합니다. 지정하지 않으면 기존 `flat` 환경을 사용합니다.
모든 환경은 기본 로봇과 `-RobotModel cad8`, `-RobotModel cad6`에서 선택할 수 있습니다.

| 선택값 | 환경 | 난이도를 높이면 |
| --- | --- | --- |
| `flat` | 기존 평지에서 무작위 방향·거리의 목표로 이동 | 기존 평지 동작 유지 |
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

| 로봇 | 평지 로그 폴더 | 비평지 로그 폴더 | 평지 / 비평지 관측 | 행동 |
| --- | --- | --- | --- | --- |
| `baseline` | `dksh_spider_navigation` | `dksh_spider_navigation_obstacles` | 84 / 116 | 24 |
| `cad8` | `dksh_spider_cad8_navigation` | `dksh_spider_cad8_navigation_obstacles` | 84 / 116 | 24 |
| `cad6` | `dksh_spider_cad6_navigation` | `dksh_spider_cad6_navigation_obstacles` | 66 / 98 | 18 |

로그 폴더는 모두 `logs/rsl_rl/` 아래에 생성됩니다. 비평지 환경은 주변 장애물과 위험 상태를
표현하는 공통 32차원 관측을 추가합니다. 같은 로봇 모델의 비평지 환경끼리는 입력·출력 크기가
같아 체크포인트를 공유할 수 있지만, 새 환경에서의 보행 성능은 별도로 평가해야 합니다.

추가되는 32개 관측값은 다음과 같습니다.

| 개수 | 관측값 |
| --- | --- |
| 16 | 로봇 루트 높이에서 수평 방향으로 계산한 장애물 경계 상자(AABB)까지의 거리 |
| 9 | 로봇 주변 지점의 지형 높이 |
| 3 | 움직이는 바닥의 3축 속도 |
| 4 | 가장 가까운 낙하물의 상대 위치 3축과 수직 속도 |

이 관측은 시뮬레이터의 실제 형상과 상태를 직접 사용하는 방식입니다. 거리 계산은 형상의
축 정렬 경계 상자를 사용하며, 센서 잡음이나 카메라 가림 현상은 모델링하지 않습니다.

기존 평지 체크포인트는 비평지 관측 크기와 맞지 않으므로 비평지 학습은 새 정책으로 시작합니다.
자동 체크포인트 선택도 별도의 `*_obstacles` 폴더만 검색합니다. 제공된 `balance_baseline.pt`는
`baseline`의 `flat`에서만 자동으로 사용합니다. 비평지 체크포인트가 아직 없으면 `preview`로 환경을
먼저 확인하거나 `train`으로 학습합니다. `-Checkpoint`를 직접 지정할 때도 환경과 로봇이 맞아야 합니다.

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
