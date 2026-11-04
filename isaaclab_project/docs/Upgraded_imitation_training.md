# Upgraded HexapodRobot 모방학습

`Upgraded-CAD6-Imitation-v0`는 `HexapodRobot.zip`의 **한 발씩 들기 → 옮기기 → 접지 확인 → 몸 이동**을
현재 CAD6에 맞춰 재구성한 환경이다. 별도 RL Teacher를 학습하지 않는다. 코드로 만든 시범을 BC로
학습한 뒤 같은 actor를 PPO로 미세 조정한다. 기존 PMTG와 관측·행동 계약이 다르므로 모델을 섞을 수 없다.

## 원본에서 가져온 것과 검증 범위

- `hexapod_stair.py`: 한 번에 한 발을 옮기고 나머지 다리의 지지를 확인하는 원칙.
- `hexapod_stair_seq.py`: 앞·중간·뒤 다리 순서, 접지가 풀린 뒤 다시 눌렸는지 확인하는 절차.
- `hexapod_sdk.py`: PC가 아는 관절각은 실제 측정각이 아닌 **명령각**이라는 센서 구분.

압축에는 `hexapod_body.py`, `hexapod_joy_walk.py`가 없다. 원본은 그대로 실행할 수 없으며 이 환경은
원본의 전체 계단 제어기를 복제하지 않는다. 원본의 도 단위·180도 영점·펌웨어 미러링·서보 한계는
현재 URDF의 라디안 좌표로 직접 복사하지 않았다. 프로젝트의 CAD FK/IK를 사용한다.
원본의 지형 지도, IMU 기반 높이 추정, 큰 계단 접기 동작, 하중 분배·토크 계획은 아직 이식하지 않았다.
`hexapod_stair_crawl.py`의 다리 끌기도 현 프로젝트의 미끄러짐·비말단 접촉 기준과 맞지 않아 사용하지 않는다.
ZIP은 AST와 해시만 읽으며 Bluetooth/serial 코드를 import하거나 실행하지 않는다. `CLAUDE.md`는 제외한다.

기본 시범은 보폭 30 mm, 들기 25 mm의 느린 평지 보행이다. 접지 해제/착지는 60 ms 동안 확인한다.
발이 들리지 않음, 지지 상실, 착지 실패, 기울기 한계, IK 실패는 해당 에피소드를 종료시킨다.
5개 **명령 발 위치**의 지지 다각형 여유도 확인하지만, 실제 무게중심·동적 안정성 보장은 아니다.
게인·모터 토크·CAD 충돌 기준은 기존 물리 모델을 유지한다. 이전에 관찰된 하중 추종 실패가 이 방식으로
해결됐다는 증거는 아직 없다. GPU에서 정상 시범을 얻지 못하면 학습 데이터가 생성되지 않는 것이 정상이다.

## 관측과 행동

- 행동: 18개 관절의 정규화된 절대 목표각. 50 Hz. 기존 1.5 rad/s 제한과 0.35 필터가 적용된다.
- actor: IMU 6, 이동 명령 3, 명령각 18, 명령속도 18, 이전 행동 18, 접지 신호 6,
  동작 단계/다리/시간 13, L2 거리·유효값 32. 한 프레임 114개 × 4 = **456개**.
- critic: actor 관측과 기존 물리 상태 140개를 합친 **596개**.
- 발 접지는 시뮬레이터의 엄격한 말단 접촉을 이진 스위치로 근사한다. 압축의 실제 발 버튼과 동일한
  감도·지연·장착 위치가 검증된 것은 아니다. 실제 관절각, 지형 정답, 접촉 힘은 actor에 넣지 않는다.
- 시퀀서가 단계와 이동할 다리를 결정하고, 신경망이 관절 명령을 출력한다. 완전히 자유로운 보행 타이밍을
  학습하는 구조는 아니다. 원본 상태기계가 학습 후에도 위상 입력을 제공한다.

## 실행

CPU 모드는 현재 Python의 `torch`, `numpy`를 사용한다. 수집·평가·PPO는 기존 Isaac Sim 4.5 /
Isaac Lab 2.1과 NVIDIA GPU 환경에서 실행한다. 설치된 의존성을 이 스크립트가 변경하지 않는다.

```powershell
# 압축 구조와 누락 모듈 확인: 시뮬레이터 불필요
.\Upgraded_run_imitation.ps1 -Mode inspect

# 기본 4개 환경 × 120초. 새 디렉터리를 지정한다.
.\Upgraded_run_imitation.ps1 -Mode collect -NumEnvs 4 -Steps 6000 -Seed 42 `
  -Output outputs/Upgraded_demo_flat_42 -AllowProvisionalCad

# 정상적으로 완료된 시범이 2개 이상 있을 때만 실행 가능
.\Upgraded_run_imitation.ps1 -Mode bc -Dataset outputs/Upgraded_demo_flat_42 `
  -Output outputs/Upgraded_bc_flat.pt -Epochs 100

# BC 오차가 낮아도 물리 보행 성공을 뜻하지 않는다. 별도 시드로 실제 rollout 평가
.\Upgraded_run_imitation.ps1 -Mode evaluate -Checkpoint outputs/Upgraded_bc_flat.pt `
  -Seed 211 -Output outputs/Upgraded_bc_eval_211.json -AllowProvisionalCad

# 같은 actor를 PPO로 미세 조정. critic은 새로 학습한다.
.\Upgraded_run_imitation.ps1 -Mode train -Checkpoint outputs/Upgraded_bc_flat.pt `
  -Output outputs/Upgraded_ppo_flat -MaxIterations 1000 -AllowProvisionalCad
```

다른 실행에는 새 `Upgraded...` 출력 이름을 사용한다. 시범·모델을 묵시적으로 덮어쓰지 않는다.
PPO 재개와 평가는 `Upgraded_model_*.pt`도 받는다. `-Environment stairs -Difficulty 0`은 동일 관측의
추가 실험을 위한 옵션이며, 현재 시범이 160 mm 계단을 오를 수 있다는 뜻은 아니다. 평지 시범 검증부터 한다.

## 데이터와 진단

`Upgraded_episode_*.npz`에는 **행동 직전 관측과 그 관측에 대응하는 실제 입력 명령**을 저장한다.
자동 리셋 이전의 지표로 에피소드 전체를 검사한다. 금지 접촉·낙상·경계 이탈·기준 동작 오류가 하나라도
있으면 전체를 버린다. 10초 이상, 순전진 20 mm 이상인 완료 에피소드만 사용한다. 시간 초과 에피소드도
이 조건을 만족하면 느린 보행 시범으로 인정하며, 목표 도달 성공과 구분한다. 미완료 에피소드는 버린다.

`Upgraded_manifest.json`은 채택/거부 이유·소스/URDF/USD 계약·시범 파일 해시를 기록한다.
`Upgraded_collection_report.json`에는 기존 평가 지표와 실제 최대 추종 오차·출력 토크·기준 동작 오류를 남긴다.
정상 시범이 2개 미만이면 수집 결과와 실패 기록을 남기고 실패를 반환한다. 가짜 성공 시범으로 대체하지 않는다.
메모리는 환경별 한 에피소드로 제한하고, 채택한 에피소드는 별도 파일로 저장한다.

BC는 프레임을 섞기 전에 **에피소드 단위로 학습/검증 데이터를 분리**한다. 최저 검증 MSE의 가중치를
원자적으로 저장한다. 검증 MSE와 물리 rollout 결과는 서로 다른 보고서다. 기본 로딩 상한은 20만 샘플이며,
대규모 수집 시 Python runner의 `--maximum_samples`를 메모리에 맞춰 명시적으로 설정한다.

```powershell
python -m unittest discover -s isaaclab_project/tests -p 'Upgraded_test_*.py' -v
```

CPU 검사는 시퀀스·CAD IK·데이터 정렬·실패 배제·에피소드 분리·BC 저장/불러오기·관측 계약을 검증한다.
접촉 역학, 실제 계단 보행과 실기 전달은 이 검사로 검증되지 않는다.
