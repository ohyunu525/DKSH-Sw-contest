# CORE: Raspberry Pi 보행 정책 입력·추론

`CORE.py`는 센서/ESP32가 없어도 **가상 상태 → 68개 관측값 → ONNX 정책의 18개 출력 → 학습과 같은 행동 보정 → 화면/JSONL 기록**을 한 번 실행합니다. 모터 명령은 만들거나 전송하지 않습니다.

대응하는 학습 작업은 6족 MG90S Walk와 Sprint입니다. 이 저수준 보행 정책의
관측에는 LiDAR·카메라·목표 위치가 들어 있지 않습니다. LiDAR는 ROS2 SLAM/Nav2가 `/cmd_vel`을
생성하는 상위 계층에서 사용하고, 보행 정책은 그 속도 명령을 `command` 관측으로 받는 구조입니다.
학습 저장소의 목표 탐색 정책은 LiDAR를 포함한 98개 관측값과 다른 관절 행동 규칙을 사용합니다.
그 정책 파일을 이 68개 관측값 보행 코드에 연결할 수 없습니다. L1 RM 장치의 질량과 장착 위치는
보행 학습의 **물리 설정**에 반영되지만, LiDAR 점군 자체는 이 보행 정책의 입력에 들어가지 않습니다.

## 먼저 가상 입력 확인

Pi에서 `RSPI/scripts` 폴더로 이동한 뒤:

```bash
python3 CORE.py --state sample_state.json --log core_log.jsonl
```

`--state`를 생략하면 내장 가상 상태를 사용합니다. 모델 없이 실행한 결과는 `status: observation_only`, `action_raw: null`입니다. `observation` 배열은 68개입니다. `--log`를 지정하면 실행마다 JSON 한 줄을 추가합니다.

`--profile walk`는 기본 보행 작업의 학습 범위인 **0.003~0.006 m/s**를 검사합니다.
빠른 보행 작업은 별도 정책이므로 `--profile sprint`를 지정하며 학습 범위는 **0.012~0.050 m/s**입니다.
가상 상태를 쓰는 빠른 보행 형식 확인은 `python3 CORE.py --profile sprint`로 할 수 있습니다.
`sample_state.json`의 0.0045 m/s는 walk 전용 예시입니다.
코드와 학습 설정의 일치 여부는 `python3 -m unittest test_CORE.py`로 확인할 수 있습니다.

## 학습 모델 연결

6족 Walk 학습이 완료된 체크포인트를 Isaac Lab의 `play.py`로 열면 해당 학습 폴더의 `exported/policy.onnx`가 생성됩니다. 예를 들어 저장소 루트의 Windows PowerShell에서:

```powershell
.\run_mg90s.ps1 -Mode play -NumEnvs 1 -Checkpoint '.\logs\rsl_rl\dksh_mg90s_cad6_walk\<run>\model_2001.pt'
```

위 경로는 예시입니다. 실제로 완료된 실행 폴더의 체크포인트를 사용하세요. `policy.onnx`와 같은 실행 폴더의 `run_metadata.json`을 Pi의 같은 디렉터리로 복사합니다. 이 ONNX에는 학습 때 사용한 관측 정규화가 포함됩니다.

Pi에서:

```bash
python3 -m pip install -r requirements-core.txt
python3 CORE.py --state sample_state.json --model ./policy.onnx --log core_log.jsonl
```

ONNX Runtime의 CPU 패키지는 Linux ARM64를 지원하므로, Raspberry Pi에서는 64비트 OS와 해당 Python용 패키지를 사용하세요. 32비트 OS의 패키지 설치 가능 여부는 별도 확인이 필요합니다. [ONNX Runtime Python 설치 문서](https://onnxruntime.ai/docs/get-started/with-python.html)

모델이 있으면 `status: inferred`와 `action_raw` 18개가 표시됩니다. 프로그램은 ONNX의 입출력 차원과 `run_metadata.json`의 작업 ID, 6족, 68관측, 18행동, 학습 완료 상태, L1 RM을 포함한 학습 질량 약 2.54362 kg을 검사합니다. 따라서 LiDAR 질량이 반영되기 전의 같은 작업 모델도 거부합니다. Sprint 모델은 `--profile sprint`와 함께 실행해야 합니다. 다른 위치의 메타데이터는 `--metadata /path/to/run_metadata.json`으로 지정할 수 있습니다. 현재 저장소에 포함된 공개 모델 파일은 8족용이므로 이 프로그램의 6족 추론에 사용할 수 없습니다.

추론 후 표시하는 `action_clipped`는 모델 출력을 `[-1,1]`로 제한한 값입니다.
`filtered_actions_next = filtered_actions + 0.2 × (action_clipped − filtered_actions)`이고,
`joint_residual_rad = 0.035 × filtered_actions_next`입니다. 이는 학습 환경의
`_pre_physics_step()`와 같은 **보정분** 계산이며, 기준 보행 관절값을 합친 최종 목표는 아닙니다.
다음 정책 호출의 관측값에는 `filtered_actions_next`를 넣어야 합니다.

## 관측값이란?

관측값(`observation`)은 **지금 로봇이 어떤 상태인지 AI 정책에 알려 주는 숫자 묶음**입니다. 카메라 화면을 그대로 넣는 것이 아니라, 학습할 때 정해 둔 항목을 정해 둔 순서로 이어 붙인 길이 68의 배열입니다. 정책은 이 배열을 입력으로 받아 관절 보정 행동 18개를 출력합니다.

예를 들어 가상 입력에서 `projected_gravity_b=[0,0,-1]`은 몸체가 똑바로 서 있다는 뜻이고, `command=[0.0045,0,0]`은 초당 0.0045 m로 앞으로 가라는 목표입니다. 현재 관절 각도가 기본 각도와 같으면 해당 관절의 위치 관측값은 0입니다. 마지막 `sin(phase), cos(phase)`는 보행 주기의 현재 위치를 나타냅니다.

관측값은 센서의 원본 데이터와 다릅니다. 아래 표의 값들을 실제 센서나 상태 추정기로 얻은 뒤, 학습 때와 같은 순서·단위로 변환해야 합니다.

## 상태 JSON 형식

`sample_state.json`을 복사해 각 필드를 실제 값으로 바꿀 수 있습니다. 순서는 저장소의 `MG90SWalkEnv._get_observations()`와 같습니다.

| 필드 | 길이 | 의미 |
| --- | ---: | --- |
| `root_lin_vel_b` | 3 | 몸체 좌표계 선속도, m/s |
| `root_ang_vel_b` | 3 | 몸체 좌표계 각속도, rad/s |
| `projected_gravity_b` | 3 | 몸체 좌표계 중력 방향, 직립 시 `[0,0,-1]` |
| `command` | 3 | 몸체 기준 `[전진속도, 0, 0]`; 학습 범위 밖의 속도, 횡이동, 회전은 거부 |
| `joint_pos` | 18 | 관절 현재 각도, rad; 관측에는 기본 각도를 뺀 값 사용 |
| `default_joint_pos` | 18 | 학습 환경의 기본 관절 각도, rad; 관측 배열에는 별도 항목으로 들어가지 않음 |
| `joint_vel` | 18 | 관절 각속도, rad/s; 관측 시 0.1배 |
| `filtered_actions` | 18 | 직전 단계까지 평활화된 AI 행동 |
| `phase` | 1 | 보행 위상, 0 이상 1 미만; 관측에는 `sin(2πphase), cos(2πphase)` 두 값 사용 |

관측 배열의 실제 묶음은 **몸체 선속도 3 + 각속도 3 + 중력 방향 3 + 목표 명령 3 + 관절 위치 차이 18 + 관절 속도 18 + 평활화 행동 18 + 보행 위상 2 = 68개**입니다.

18개 관절 배열은 **학습 환경의 `robot.joint_names` 순서**와 일치해야 합니다. 가상 예제의 순서는 설명용이며 실제 Isaac Lab 관절 순서를 검증한 값은 아닙니다. 현재 코드는 한 번의 추론만 하므로 다음 호출의 `filtered_actions`와 보행 위상은 호출자가 이어서 공급해야 합니다. 정지, 후진, 횡이동, 회전 명령은 이 학습 정책이 다루지 않으므로 별도 제어 상태로 처리해야 합니다. IMU만으로 몸체 선속도와 관절 위치/속도를 알 수 없으므로, 실제 센서 연결 전까지 가상 입력을 실측값으로 해석하지 마세요.

출력 `action_raw`는 정규화된 관절 **보정 행동**입니다. 최종 관절 목표나 서보 PWM이 아니며, 실제 구동에는 기준 보행, 관절 한계·목표 변화율 제한, 이상 상태 정지, 통신 검사 등이 더 필요합니다.
