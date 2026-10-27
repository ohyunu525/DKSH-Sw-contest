# 15cm 디딤판 센서 기반 계단 학습 — 2026-10-27

사용자가 확인한 목표는 계단 한 단의 **앞뒤 깊이 15cm, 높이 15cm**다.
계단 좌우 폭은 기존 1.2m 가정이다. 3단이며 맨 위에 0.6m 플랫폼이 있다.
처음 실행한 `staged_v1`은 6, 10, 20, 40, 60, 120mm로 시작했으며,
목표가 확인된 뒤 `staged_15cm_v2`에 6, 10, 20, 40, 60, 80, 100, 120, 150mm를 추가했다.
이 실험 단계들은 실물 등반 가능 높이를 뜻하지 않는다. 각 환경은 6mm에서 시작하고 한 높이에서 연속 3회 전체 계단을
통과해야 높아진다. 타임아웃은 성공으로 취급하지 않는다.

시뮬레이션 모델은 잠정 Onshape CAD6, MG996R 6개와 RC920DMG 12개,
L2/BNO086, 발끝 캡 버튼 6개 및 18개 절대 관절 목표각을 사용한다.
센서 근사, 무게/모터/캡 및 서보 보정은 [기존 버튼 정책 기록](button_absolute_training.md)의
미측정 가정을 유지한다. 실제 두 번째 IMU의 출력은 actor에 추가하지 않았고,
실제 관절 각도/속도나 몸체 선속도 참값도 actor 관측에 넣지 않는다.
학습 보상은 시뮬레이션 진행 거리 참값을 사용한다.

2026-10-27 fetch한 OYW `8b4ac71`의 `model_10000.pt`는 140관측·18행동,
별도 관측/행동 계약이다. 6mm 계단에서 84/84 성공, 낙상 0 기록이 있으나
발판 깊이가 0.55m인 별도 시뮬레이션이고 실제 관절 상태 등 privileged 관측을 사용한다.
본 15cm 발판의 실기 센서 계약과 일치하지 않아 해당 가중치를 복사하거나 재개하지 않는다.
대신 signed progress와 Huber 속도 오차, 초기 탐색 표준편차 0.08,
128-step rollout, gamma=0.9995, lambda=0.98, 목표각 평활화 0.35를 참고한다.
본 보상 가중치와 센서/지형은 독립적인 실험 가정이다. 행동은 여전히
정규화 출력을 절대 관절각으로 매핑하며, 기준 보행 또는 IK를 더하지 않는다.
목표각 변화율은 1.5 rad/s, 제어 주기는 20ms다.

두 staged 전략은 초기 잡음만 0.08로 지정하며 OYW의 학습 중 탐색 표준편차
최소/최대 제한은 적용하지 않는다. 학습 중 실제 std는 TensorBoard 기록에서 확인한다.
평가와 체크포인트 재개는 저장된 tread/width/strategy와 같아야 한다. 다만 동일 지형에서
`staged_v1`을 `staged_15cm_v2`로 계속 학습할 때만 `-AllowCurriculumTransfer`를 명시한다.
이는 reward/관측/행동은 유지하고 높이 단계만 확장한다. 에피소드와 높이는 6mm로 리셋되며
신경망·정규화·optimizer·탐색 표준편차는 checkpoint에서 복원한다.
기존 25cm `legacy_v1` 정책은 원래 설정으로 계속 사용 가능하다.

```powershell
.\run_stairs.ps1 -Mode train -NumEnvs 16 -MaxIterations 300 -Seed 42 `
  -TreadM 0.15 -StairWidthM 1.2 -Strategy staged_v1 -AllowProvisionalCad
.\run_stairs.ps1 -Mode evaluate -NumEnvs 4 -Level 0 -Steps 4100 -Seed 43 `
  -TreadM 0.15 -StairWidthM 1.2 -Strategy staged_v1 `
  -Checkpoint '<새 model_299.pt>' -AllowProvisionalCad

# 150mm 목표 단계 추가: 이전 관측/행동을 유지하며 명시적으로 이어 학습
.\run_stairs.ps1 -Mode train -NumEnvs 16 -MaxIterations 300 -Seed 42 `
  -TreadM 0.15 -StairWidthM 1.2 -Strategy staged_15cm_v2 -AllowCurriculumTransfer `
  -Checkpoint '<staged_v1 model_299.pt>' -AllowProvisionalCad
.\run_stairs.ps1 -Mode evaluate -NumEnvs 4 -Level 8 -Steps 4100 -Seed 43 `
  -TreadM 0.15 -StairWidthM 1.2 -Strategy staged_15cm_v2 `
  -Checkpoint '<staged_15cm_v2 model_599.pt>' -AllowProvisionalCad
```

각 높이의 콜라이더, 광선 교차, 지형 높이/목표 판정에 같은 tread/width가 전달된다.
확장된 9개 높이의 patch가 다른 병렬 환경과 겹치지 않도록 환경 간격은 24.5m로 늘린다.
관측 244개 및 동작 계약은 변하지 않고, 지형·전략은 별도의 metadata에 기록한다.
모델 학습 완료, 시뮬레이션 평가 통과, 실물 적용은 별개의 결과다.

## 사전 실행 및 검증

단위 테스트 145개와 PowerShell 구문, `git diff --check`를 통과했다.
15cm 디딤판의 3개 모서리·좌우 폭·지면 높이 일치, 전진/정지/왕복의 보상 순서,
잘못된 크기 및 다른 지형/전략의 평가 체크포인트 거부를 포함한다.

seed 42, 4→8→16개 환경에서 각각 2 PPO iteration을 실행했다. 모두 정상 종료,
6개 버튼 눌림 입력과 ONNX/TorchScript 절대각 내보내기 비교를 확인했다.
각 기록은 `logs/rsl_rl/dksh_cad6_stairs_mg996r_buttons_absolute_v2/<run>/`에 있으며
체크포인트는 각 `model_1.pt`다. 원본은 기존 `.gitignore`에 따라 로컬에 보관한다.

| 환경 | 실행 디렉터리 | checkpoint SHA-256 |
|---:|---|---|
| 4 | `2026-10-27_13-04-41_684376` | `d0678a9ebf5747e27dfdbbf10a3500bea8bcd573ad575ece3b812425e5e2668b` |
| 8 | `2026-10-27_13-05-03_181954` | `c591577d3548fc2603ee2ef4417a3c48b47647f3552b3cc1b138d28559d0622a` |
| 16 | `2026-10-27_13-05-33_561039` | `f384580504871845618256030bebb11dde5d5fa4cfa1dc8ef275038c2ab05e6e` |

4환경 실행은 완료 레코드가 유효했으나 Isaac wrapper의 종료 코드 1이 외부 호출에
남는 기존 실행기 버그로 셸에서 실패로 읽혔다. 완전한 완료 레코드와 허용된 종료
코드를 확인한 경우에만 실행기가 0을 반환하도록 수정했고 16환경 실행에서 확인했다.
모델 품질 평가 결과는 아니다.

## 후속 전략: 상태 특권 교사에서 실기 센서 학생으로 증류

[관절 피드백이 없는 6족 로봇의 조이스트 통과 연구](https://www-video.eecs.berkeley.edu/papers/zixian/hexapod_IROS_2023.pdf)는
시뮬레이션 참값을 보는 교사를 학습한 뒤 실제 탑재 센서만 보는 학생에게 지식을 전달했다.
이는 다른 로봇과 지형의 결과이지 이 로봇의 계단 성능을 증명하지 않는다.

현재 직접 학습이 6mm 단계에서 정체되면, 15cm 지형 및 동일한 모터/캡 물리에서
교사를 먼저 학습하고 동일 상태의 센서 관측·교사 절대각 목표 쌍을 모아 학생을
초기화한 뒤, 학생이 만든 상태도 추가 수집해 폐루프 오차를 검증하는 실험이 후보다.
OYW 140관측 모델은 현재 244관측 학생에 직접 연결할 수 없고, 기존 55cm 발판과
발 접촉 모델도 다르다. 교사의 관측 생성, 18관절 순서, ±90°와 ±85.5° 제한,
지형·캡 물리 및 정규화 계약을 명시적으로 맞춰야 한다.
증류는 아직 구현하거나 실행하지 않았으며 이번 직접 학습 결과와 구분한다.
