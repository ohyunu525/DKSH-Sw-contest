# Upgraded 원본 Gait 기반 CAD 모방학습

`Upgraded-CAD6-Imitation-v0`는 `HexapodRobot2.zip`의 `hexapod_joy_walk.py:Gait`에서
발끝 궤적과 tripod 보행 계산을 가져와 현재 CAD FK/IK로 연결한다. 별도 RL Teacher를 학습하지 않는다.
스크립트 보행으로 물리 시범 수집 → BC → 같은 actor를 PPO로 개선하는 흐름이다.
기존 single-foot crawl 시범 및 모델과 관측·제어 계약이 다르므로 **새로 수집하고 BC를 학습해야 한다**.
기존 PMTG 모델도 이 환경에 바로 불러올 수 없다.

## 원본에서 가져온 것과 검증 범위

- `HexapodRobot.zip`의 SDK·계단 코드와 `HexapodRobot2.zip`의 Body·Gait 코드를 함께 검사한다.
  두 ZIP을 합치면 Python 모듈 7개의 로컬 import가 연결된다. 검사 자체는 AST와 해시만 읽고 압축을 풀지 않는다.
- `Upgraded_original_gait.py`는 원본 `swing_profile`, 착지 목표, 지지 발 이동, 보폭 예산과 속도 램프를
  배치 Torch 연산으로 옮긴 계산 코어다. 원본 Python 모듈·`main()`을 import하거나 실행하지 않는다.
- `Upgraded_gait_reference.py`가 CAD IK와 현재 제어 한계를 적용하고 접지에 맞춰 보행 진행을 조절한다.
  원본 도 단위·180도 영점·펌웨어 미러링을 CAD 관절각에 직접 복사하지 않는다.
- 원본은 200°/s의 느슨한 governor를 사용하고 SDK 보간을 사실상 끈다. 이 환경은 기존 **1.5 rad/s 제한과
  0.35 smoothing**을 유지한다. 제한을 통과할 수 있도록 기준 궤적의 진행 속도를 조절한다.
- 원본 `Grounder`의 실제 시간 대기·IMU 응답 탐색 대신 시뮬레이션 시간과 엄격한 말단 접촉 신호를 사용한다.
  원본 하드웨어 프로그램 전체, IMU 보정·지형 지도·계단 접기·하중 분배 계획을 그대로 실행하는 것은 아니다.
- 착지 확인 뒤에는 원본 `_feet_plane`/`_recenter` 원리에 따라 여섯 발의 높이 평면을 추정하고,
  CAD IK와 제어 제한 안에서 몸 높이·기울기 기준을 되돌린다. 착지마다 처진 높이를 새 기준으로 받아들여
  계속 낮아지는 현상을 줄이기 위한 절차이며, 실제 모터의 하중 지지 성능은 별도 검증이 필요하다.

이식 범위는 평지 tripod 시범이다. 시리얼·Bluetooth 연결과 펌웨어 동작은 실행하지 않으며 `CLAUDE.md`는 제외한다.
기본값은 주기 2.4초, duty 0.6, 발 들기 20 mm, 최대 보폭 30 mm, 착지 접근 여유 4 mm,
목표 이동 속도 15 mm/s다. 실제 주기는 접촉 확인과 관절 제한에 따라 길어질 수 있다.
게인·모터 토크·CAD 충돌 기준은 기존 물리 모델을 유지한다. 이전 하중 추종 실패가 해결됐다는 증거는 아직 없다.
CPU에서 궤적과 IK 계산이 통과해도 접촉 역학에서 보행할 수 있다는 보장은 없다. GPU에서 정상 시범을 얻지
못하면 실패 기록을 남기고 BC 학습을 시작하지 않는다.

## 관측과 행동

- 행동: 18개 관절의 정규화된 절대 목표각. 50 Hz. 기존 1.5 rad/s 제한과 0.35 필터가 적용된다.
- actor: IMU, 이동 명령, 필터 적용 명령각·명령속도, 이전 행동, 이진 접지, 기준 Gait 내부 상태,
  L2 거리·유효값. Gait 상태 95개를 포함한 프레임 196개 × 4 = **784개**이며 계약에도 기록한다.
- critic: actor 관측과 기존 물리 상태 140개를 합친 **924개**.
- 발 접지는 시뮬레이터의 엄격한 말단 접촉을 이진 스위치로 근사한다. 원본의 IMU 착지 추정과 동일한
  감도·지연이 검증된 것은 아니다. 실제 관절각, 지형 정답, 접촉 힘은 actor에 넣지 않는다.
- 기준 Gait는 학습 후에도 위상·상태 입력을 제공하고, 신경망이 18개 관절 명령을 출력한다.
  완전히 독립적인 보행 타이밍을 신경망이 생성하는 구조는 아니다.

## 실행

CPU 모드는 현재 Python의 `torch`, `numpy`를 사용한다. 수집·평가·PPO는 기존 Isaac Sim 4.5 /
Isaac Lab 2.1과 NVIDIA GPU 환경에서 실행한다. 설치된 의존성을 이 스크립트가 변경하지 않는다.

```powershell
# 두 압축의 구조·누락 모듈·해시 확인: 시뮬레이터 불필요
.\Upgraded_run_imitation.ps1 -Mode inspect

# 기본 4개 환경 × 120초. 새 디렉터리를 지정한다.
.\Upgraded_run_imitation.ps1 -Mode collect -NumEnvs 4 -Steps 6000 -Seed 42 `
  -Output outputs/Upgraded_gait_demo_flat_42 -AllowProvisionalCad

# 정상적으로 완료된 시범이 2개 이상 있을 때만 실행 가능
.\Upgraded_run_imitation.ps1 -Mode bc -Dataset outputs/Upgraded_gait_demo_flat_42 `
  -Output outputs/Upgraded_gait_bc_flat.pt -Epochs 100

# BC 오차가 낮아도 물리 보행 성공을 뜻하지 않는다. 별도 시드로 실제 rollout 평가
.\Upgraded_run_imitation.ps1 -Mode evaluate -Checkpoint outputs/Upgraded_gait_bc_flat.pt `
  -Seed 211 -Output outputs/Upgraded_gait_bc_eval_211.json -AllowProvisionalCad

# 같은 actor를 PPO로 미세 조정. critic은 새로 학습한다.
.\Upgraded_run_imitation.ps1 -Mode train -Checkpoint outputs/Upgraded_gait_bc_flat.pt `
  -Output outputs/Upgraded_gait_ppo_flat -MaxIterations 1000 -AllowProvisionalCad
```

다른 실행에는 새 `Upgraded...` 출력 이름을 사용한다. 검사 보고서도 기본값에 시각을 붙이며 기존 출력을 덮어쓰지 않는다.
ZIP이 다른 위치에 있으면 `-Archive <첫 ZIP 경로> -SupplementArchive <두 번째 ZIP 경로>`를 지정한다.
PPO 재개와 평가는 `Upgraded_model_*.pt`도 받는다. `-Environment stairs -Difficulty 0`은 동일 관측의
추가 실험을 위한 옵션이며, 현재 시범이 160 mm 계단을 오를 수 있다는 뜻은 아니다. 평지 시범 검증부터 한다.

## 데이터와 진단

`Upgraded_episode_*.npz`에는 **행동 직전 관측과 그 관측에 대응하는 실제 환경 입력 명령**을 저장한다.
정답은 smoothing 적용 전의 정규화 절대 목표각이다. 제한 후 명령을 정답으로 저장하면 BC 실행 때 필터가
다시 적용되므로 그러한 이중 필터링을 하지 않는다. 물리 엔진이 실행하는 명령에는 항상 기존 필터가 적용된다.
자동 리셋 이전의 지표로 에피소드 전체를 검사한다. 금지 접촉·낙상·경계 이탈·기준 동작 오류가 하나라도
있으면 전체를 버린다. 10초 이상, 순전진 20 mm 이상인 완료 에피소드만 사용한다. 시간 초과 에피소드도
이 조건을 만족하면 느린 보행 시범으로 인정하며, 목표 도달 성공과 구분한다. 미완료 에피소드는 버린다.

`Upgraded_manifest.json`은 채택/거부 이유·두 ZIP 해시·실행 코어/어댑터/URDF/USD 계약·시범 파일 해시를 기록한다.
ZIP 내부 Gait 해시가 검토하여 이식한 원본 해시와 다르면 시뮬레이션 실행을 거부한다. 다른 ZIP을 지정해도
실행 코어가 자동으로 바뀌지는 않으며, 새 원본을 사용하려면 별도로 이식하고 검증해야 한다.
`Upgraded_collection_report.json`에는 기존 평가 지표와 실제 최대 추종 오차·출력 토크·기준 동작 오류를 남긴다.
정상 시범이 2개 미만이면 수집 결과와 실패 기록을 남기고 실패를 반환한다. 가짜 성공 시범으로 대체하지 않는다.
메모리는 환경별 한 에피소드로 제한하고, 채택한 에피소드는 별도 파일로 저장한다.

BC는 프레임을 섞기 전에 **에피소드 단위로 학습/검증 데이터를 분리**한다. 최저 검증 MSE의 가중치를
원자적으로 저장한다. 검증 MSE와 물리 rollout 결과는 서로 다른 보고서다. 기본 로딩 상한은 20만 샘플이며,
대규모 수집 시 Python runner의 `--maximum_samples`를 메모리에 맞춰 명시적으로 설정한다.

BC와 PPO actor는 모두 `[128, 128, 64]` ELU이고 관측의 추가 통계 정규화를 사용하지 않는다.
PPO를 시작할 때 actor 가중치를 엄격히 복사하고 같은 입력의 출력 일치를 검사한다. critic은 새로 학습하며
탐색 표준편차는 0.06에서 시작하고 0.02–0.15 범위로 제한한다. 첫 PPO 업데이트 전에
`Upgraded_model_bc_initial.pt`를 저장하고 이전 BC 파일은 보존한다. 이후 `Upgraded_model_*.pt`로 재개하거나
평가할 수 있다. 기존 데이터·모델의 제어 계약이 다르면 로딩을 거부한다.

```powershell
python -m unittest discover -s isaaclab_project/tests -p 'Upgraded_test_*.py' -v
```

CPU 검사는 원본 궤적 계산·CAD IK·데이터 정렬·실패 배제·에피소드 분리·BC 저장/불러오기·PPO actor 전달·관측 계약을 검증한다.
접촉 역학, 실제 계단 보행과 실기 전달은 이 검사로 검증되지 않는다.
