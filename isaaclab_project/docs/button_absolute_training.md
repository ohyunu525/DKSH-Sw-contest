# CAD6 발끝 버튼 + 절대 관절각 정책 — 2026-10-22

`OYW/Develop/MLForLegs`의 `d970d488be5ab0bb6e1c6ebcb218930c450aa8de`를
참고해 계단 정책을 변경했다. 해당 브랜치의 `joint_angle_control.py`에서 절대각
변환과 목표각 필터를 가져왔으며, CAD6의 관절 명명 순서를 명시적으로 매핑한다.
OYW의 관절 위치/속도·몸체 선속도 참값 관측 대신 기존 센서 관측을 유지하고 버튼을 추가했다.

## 행동

작업 ID는 `Isaac-DKSH-MG90S-CAD6-Stairs-Sensors-Direct-v1`이다.
`action_semantics=absolute_joint_angles_v1`이며 기준 wave gait, IK 보정,
보행 위상 또는 누적 각도 증분을 행동에 더하지 않는다.

PPO 내부의 정규화 출력 `a`는 각 관절의 절대 목표각으로 변환한다.

```text
desired_angle = lower + (clip(a, -1, 1) + 1) × (upper - lower) / 2
target = previous_target + clip(0.15 × (desired_angle - previous_target), -0.03, +0.03)
```

단위는 rad이며 영점은 Onshape 조립 영점이다. 현재 USD의 ±90°에 95% soft limit를
적용해 목표 범위는 약 ±85.5°다. 같은 `a`를 유지하면 같은 각도로 수렴한다.
제어 50 Hz, 목표 변화율 상한 1.5 rad/s를 사용한다. 초기 자세는 리셋용으로만 사용한다.
관절 순서는 `leg_0 hip/femur/tibia`부터 `leg_5 hip/femur/tibia`까지 18개다.

`exported/absolute_policy.onnx`와 `absolute_policy.pt`는 정규화가 포함된 모델이며
**18개 절대 목표각(rad)**을 출력한다. 출력에 기준 보행이나 기본 자세를 추가하지 않는다.
배포 제어기는 이 희망각에 위 평활화/변화율 제한을 한 번 적용해야 한다.
관측의 이전 요청 행동은 이 희망각을 관절 한계로 역정규화한 18개 값이고,
이전 명령 목표는 평활화/변화율 제한을 적용한 각도를 정규화한 값이다.
`run_metadata.json`에 관절 순서, 한계, 필터, 센서 설정을 기록한다.

## 버튼과 캡

발끝마다 1개씩 6개 버튼을 가정한다. 캡만 별도의 유한 질량 강체와 고정 관절로
만들어 tibia 옆면 충돌이 버튼 눌림으로 해석되지 않게 한다. 18개 회전 자유도는
유지되며 강체 수는 19개에서 25개로 늘어난다. 캡 접촉력은 센서 시뮬레이션에만
쓰고 actor에는 `0=해제, 1=눌림`만 제공한다.

현재 미측정 기본값:

| 설정 | 값 |
|---|---:|
| 캡 | CAD foot point를 중심으로 한 반지름 5 mm 구 |
| 캡+버튼 질량 근사 | 발당 1 g, 총 6 g |
| 눌림/해제 임계값 | 0.5 N / 0.25 N |
| 연속 상태 debounce | 40 ms |
| 접촉 샘플링 | 20 ms |

눌림과 해제 모두 연속 40 ms가 필요하며 반짝이는 입력은 누적하지 않는다.
실물 GPIO도 active-low 등의 전기 신호를 눌림=1로 바꾼 뒤 같은 debounce를 적용한다.
캡의 재질은 기존 지면/로봇 재질을 상속한다. 버튼 스트로크·스프링·변형·실제 캡
치수·질량·작동압·장착 오프셋은 아직 식별하지 않았으며 위 수치는 제조사 정격이 아니다.
다리 전체 접촉이나 지형 정답 높이로 버튼을 대신하지 않는다.

## 관측

관측 계약은 `l2_bno086_buttons_absolute_targets_v2`, 총 244개다.
기존 240개에서 보행 위상 2개를 제거하고 디지털 접촉 6개를 추가했다.

| 반열린 인덱스 | 내용 | 크기 |
|---|---|---:|
| `[0:6]` | BNO086 각속도와 중력 방향 | 6 |
| `[6:9]` | 몸체 기준 속도 명령 | 3 |
| `[9:27]` | 직전 정규화 절대각 요청 | 18 |
| `[27:45]` | 직전 실제로 명령한 목표각, 정규화 | 18 |
| `[45:51]` | 발 버튼 0~5, debounce 후 0/1 | 6 |
| `[51:243]` | L2 관측 높이 격자 96개 + 유효성 96개 | 192 |
| `[243:244]` | L2 데이터 나이, 초 | 1 |

실제 관절각/속도와 몸체 선속도를 센서 없이 채워 넣지 않는다. L2 내장 IMU와
BNO086 중 actor의 자세/각속도는 기존과 같이 BNO086 가정을 사용한다.
이번 변경은 두 IMU를 융합하거나 두 번째 IMU 6개를 새로 추가한 변경이 아니다.

## 학습과 검증

기존 68/240차원 정책과 residual 체크포인트는 재개할 수 없다.
차원뿐 아니라 작업 ID, 관측 계약, 절대각 행동 의미, 모터 계약을 모두 검사한다.
출력/관측 의미가 달라졌으므로 새 가중치로 시작한다.

```powershell
.\run_stairs.ps1 -Mode train -NumEnvs 4 -MaxIterations 2 -AllowProvisionalCad
.\run_stairs.ps1 -Mode train -NumEnvs 8 -MaxIterations 2 -AllowProvisionalCad
.\run_stairs.ps1 -Mode train -NumEnvs 16 -MaxIterations 2 -AllowProvisionalCad
.\run_stairs.ps1 -Mode train -NumEnvs 16 -MaxIterations 300 -AllowProvisionalCad
.\run_stairs.ps1 -Mode evaluate -NumEnvs 4 -Level 0 -Steps 4100 `
  -Checkpoint 'logs/rsl_rl/dksh_cad6_stairs_mg996r_buttons_absolute_v2/<run>/model_299.pt' -AllowProvisionalCad
```

결과는 `logs/rsl_rl/dksh_cad6_stairs_mg996r_buttons_absolute_v2/<run>/`에 저장한다.
완료 메타데이터에는 발별 눌림 표본 수와 원본 OYW commit도 기록한다.
export 검사는 학습 정책과 ONNX reference evaluator의 절대각 출력을 비교한다.
계단 성공은 3개 이상 버튼이 눌린 상태로 상단 자세 조건을 1초 유지해야 하며,
독립 평가의 성공률·낙상·영역 이탈·출력 토크 검사도 유지한다.

하드웨어 설정과 CAD가 여전히 provisional이므로 `hardware_deployable=false`다.
위 학습 완료는 실물 보행 또는 계단 통과의 증명이 아니다.

## 2026-10-22 실행 결과

단위 테스트 142개, Python compile, PowerShell 문법과 `git diff --check`를 통과했다.
절대각 끝점/비대칭 범위/상수 명령의 비누적성/변화율/비정상 값 유지,
관절 순서 재매핑, 버튼 눌림·해제 debounce·바운스·리셋,
센서 관측에서 참값 미사용, 이전 residual 체크포인트 거부를 검사한다.
ONNX 내보내기가 정규화 모듈을 다시 학습 모드로 켜는 회귀도 검사한다.

4→8→16환경 × 2 iteration의 런타임 확인을 통과했다.
16환경 확인에서는 6개 버튼 모두 눌림 표본이 발생했고, 질량은 2.028339386 kg,
회전 DOF 18개/강체 25개였다. 최초 export 두 번은 설치된 RSL-RL의 `alg.policy`
속성 및 export 후 정규화 모드 문제로 실패했으며 수정 후 다시 통과했다.
실패 실행은 `logs/`에 원인과 함께 남겨 두었다.

초기 학습은 2026-10-22 21:53:22~21:59:28 KST, seed 42,
16환경 × 300 iteration, 230,400 transition으로 새 가중치에서 완료했다.
완료 episode 88개 중 성공 0, 낙상 62, 시간 종료 26, 영역 이탈 0이다.
20 mm 단계에 머물렀으며 보행/등반에 수렴했다고 해석하지 않는다.
발별 눌림 표본 수는 `[182893, 111567, 93842, 21627, 138669, 145391]`이다.

체크포인트:
`logs/rsl_rl/dksh_cad6_stairs_mg996r_buttons_absolute_v2/2026-10-22_21-53-22_486358/model_299.pt`.
SHA-256: `1361a79a91dabb4fa94a4074222a09efeda0cd127374272608990e66fe851483`.
같은 폴더에 설정, 원본 코드, 메타데이터와 절대각 ONNX/TorchScript를 보관한다.
`export_verified=true`, `stair_climbing_verified=false`, `hardware_deployable=false`다.
checkpoint 및 원본 로그는 Git ignore 대상이므로 이 PC에만 보관한다.

독립 평가: 2026-10-22 21:59:43~22:01:03 KST, seed 43,
4환경 × 4,100 control step(환경당 82초), 20 mm 3단 계단.
완료 episode 7개 중 성공 0, 낙상 7, 영역 이탈 0으로 **평가 실패**다.
최대 몸체 x=0.365743637 m, 최소 지형 상대 몸체 여유 0.082027003 m,
최대 토크/시뮬레이션 상한 비율 1.0이다. 첫 계단 모서리는 x=0.40 m이며,
이 집계로 개별 발의 계단 착지 여부를 확정하지 않는다.

평가 메타데이터:
`logs/rsl_rl/dksh_cad6_stairs_mg996r_buttons_absolute_v2/2026-10-22_21-59-43_905813/run_metadata.json`.
`evaluation.passed=false`, `stair_climbing_verified=false`이며 실패 종료를 확인했다.
이 결과는 버튼/절대각 정책의 실행·학습·내보내기 검증이며,
안정적인 보행 또는 계단 통과 정책을 확보한 결과가 아니다.
