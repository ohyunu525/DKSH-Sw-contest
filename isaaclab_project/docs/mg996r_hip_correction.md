# 몸체 관절 MG996R 정정 및 계단 재학습 — 2026-10-19

사용자가 몸체–coxa 모터가 MG90S가 아니라 MG996R이라고 정정했다.
현재 6족은 MG996R 6개 + RC920DMG 12개 + Unitree L2 + BNO086 가정이다.
과거 MG90S 체크포인트와 평가는 새 하드웨어의 검증으로 간주하지 않는다.

## 직접 관절각 v2 수정 — 2026-10-23

현재 브랜치의 절대 관절각 학습은 `Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2`로
유지한다. `run_mg90s.ps1`의 기본 direct 모드는 이 작업을 선택하고, legacy 모드는
아래에 기록된 main의 v0 혼합 서보 gait 작업을 선택한다.

v2 CAD6는 main의 Onshape URDF/USD, 모든 관절 q=0 조립 자세와 ±90° 한계를 사용한다.
초기 높이는 0.1375 m, 로봇 폭은 0.56 m로 보정하고 자기충돌을 활성화한다.
관절 리셋 잡음은 없으며, 고정 보행각을 정책 출력에 더하지 않는다. 모터와 질량 가정은
아래 main 명세를 따른다. 과거 all-MG90S 형상·모터는 별도 v2 task로 선택할 수 있다.

```powershell
.\run_mg90s.ps1 -Mode check -NumEnvs 4
.\run_mg90s.ps1 -Mode train -NumEnvs 4 -MaxIterations 400 -AllowProvisionalCad
```

v2는 왕복 보상 허점, 과도한 탐색 잡음, 짧은 목표 거리, 누락된 지형 높이와 잘못된 GUI
복제 형상을 수정한다. CAD6 관측은 평지·계단 모두 140개이며, v1 체크포인트는 재사용하지 않는다.
CAD URDF 해시, 관절 순서·범위, actuator·보상·접촉·관측·제어 계약을 검사한다.
직접 각도 학습의 보상·평가 계약은 [별도 안내](joint_angle_training.md)를 따른다.

## 모델 수정

[TowerPro MG996R 제조사 페이지](https://towerpro.com.tw/product/mg996r/)의
Specification 절을 2026-10-19 확인했다. 실제 공급 전압과 실측 성능은 미확인이라
4.8 V 사양을 보수적 기준으로 사용한다. RC920DMG는 기존 5 V 가정을 유지한다.

| 항목 | 이전 MG90S 가정 | 현재 MG996R 가정 |
| --- | ---: | ---: |
| 기준 전압 | 4.8 V | 4.8 V |
| 정지 최대 토크 | 1.8 kgf·cm | 9.4 kgf·cm |
| 75% 임시 시뮬레이션 토크 상한 | 0.132390 N·m | 0.691369 N·m |
| 무부하 60° 이동 시간 | 0.10 s | 0.19 s |
| 무부하 속도 | 10.471976 rad/s | 5.511566 rad/s |
| 알려진 서보 질량 | 13.4 g | 55 g |

75%는 제조사 연속 정격이 아니다. 페이지 본문에는 0.19 s, 구형 설명 표에는
0.17 s가 혼재하므로 더 느린 0.19 s를 채택했다. 정품/변형/실제 공급 전압,
부하 속도·연속 토크·백래시·발열·전원 강하는 실측하지 않았다.
공통 관절 목표 변화율 1.5 rad/s와 PD 이득은 바꾸지 않아 모터 정정만 비교한다.
전달 관성(armature)·마찰도 기존 임시값을 유지하며 실제 MG996R의 식별 결과가 아니다.

- 몸통 6개만 MG996R DCMotor 모델로 변경했다. 다리 12개는 RC920DMG를 유지한다.
- 알려진 서보 질량 합은 다리당 0.175 kg, 6족 전체 1.05 kg이다.
  그러나 CAD 링크 질량의 모터 포함 여부가 불명확하다. 가상의 교체 질량 차이나
  서보 전체 질량을 CAD에 임의로 더하지 않았다. CAD 1.792339 kg + L2 0.230 kg의
  기존 약 2.022339 kg과 임시 관성을 유지한다. 실측 쌍둥이가 아니다.
- 정규화 URDF의 hip effort/velocity를 수정하고 공식 Isaac Lab converter로 USD를
  재생성했다. 원본 Onshape URDF·형상·조립 영점은 변경하지 않았다.
  provenance에는 정규화 URDF와 USD base/physics 레이어 hash를 기록한다.
  URDF는 LF 줄바꿈으로 고정해 Windows/Linux checkout에 따른 hash 차이를 방지한다.
- 기존 요청의 ±90° 소프트웨어 범위는 유지한다. 제조사 답변에는 일반형 159°와
  로봇형 180°가 구분되어 있어 실제 소유한 변형의 가동 범위는 미확정이다.
  현재 180° 범위는 시뮬레이션 가정이지 실물 안전 각도가 아니다.
- 실물 영점/PWM/방향·게인 보정은 미완료이며 `servo_zero_pwm_calibrated=false`다.
  임의 PWM을 만들어 전송하거나 모터를 실제 구동하지 않는다.

## 체크포인트 격리와 재현

새 actuator contract: `mg996r_4v8_rc920dmg_5v_v1`.
CAD6 checkpoint guard가 계약 누락/다른 모터 계약을 거부한다. 관측 크기가 같다고
MG90S 학습을 그대로 재개하지 않는다. 현재 재학습은 처음부터 수행한다.
기존 task ID/스크립트의 `MG90S` 문자열은 호환용 이름으로 남긴다.

새 실험 디렉터리: `logs/rsl_rl/dksh_cad6_stairs_mg996r_l2_bno086_v1/`.
관측은 기존 240개 센서 가정, 행동 18개, 마찰 0.6, 물리 5 ms/제어 20 ms다.
20 mm 3단부터 시작해 연속 3회 통과해야 높이를 올리는 기존 기준을 유지한다.
라이다·IMU 근사와 자기 차폐 미구현 등은 [계단 학습 문서](stair_sensor_training.md)를 따른다.

```powershell
.\run_stairs.ps1 -Mode train -NumEnvs 16 -MaxIterations 300 -AllowProvisionalCad
.\run_stairs.ps1 -Mode evaluate -NumEnvs 4 -Level 0 -Steps 4100 `
  -Checkpoint 'logs/rsl_rl/dksh_cad6_stairs_mg996r_l2_bno086_v1/<run>/model_299.pt' -AllowProvisionalCad
```

런타임에서 6개 hip의 actuator 토크/속도와 PhysX 토크 상한을 검사한다.
메타데이터는 실제 actuator 상한, 두 모터 모델/전압, 질량 가정, CAD 준비 상태를 기록한다.
학습 완료와 계단 통과를 분리하며 `hardware_deployable=false`를 유지한다.

## 실행 결과

아래에 초기 재학습 및 다른 seed 평가 결과를 기록한다. 체크포인트·원본 로그는 로컬
`logs/` 아래에 보관하며 Git에는 코드·설정·자산·검증 요약만 커밋한다.

2026-10-19 12:06:37~12:12:14 KST, seed 42, 16환경 × 300 iteration,
230,400 transition의 초기 PPO 실행을 완료했다. 학습 중 완료 episode 59개 중
전체 계단 통과 0개, 낙상 0개, 영역 이탈 21개, timeout 38개다.
높이 단계는 20 mm에 머물렀으며 충분히 수렴한 정책으로 간주하지 않는다.

저장 위치:
`logs/rsl_rl/dksh_cad6_stairs_mg996r_l2_bno086_v1/2026-10-19_12-06-37_155818/model_299.pt`.
SHA256: `e421465f6fc04a1a5eb59056937a5dd8494f5c2fe15762246d218dff0daa088c`.
실제 상한은 hip 0.691368818 N·m / 5.511566162 rad/s,
leg 1.397447586 N·m / 6.544984818 rad/s이며, 관절별 런타임 검사를 통과했다.
메타데이터에 `status=completed`, `stair_climbing_verified=false`를 기록했다.
학습 시작 후 URDF 텍스트를 LF로 고정해 소스 hash만 바뀌었으며 물리 USD와 모터
파라미터는 변경하지 않았다. 당시 manifest는 실행 폴더의 source snapshot에 보존했다.

검증: 단위 테스트 135개 중 134개 통과/1개 skip. 자산 hash, 새 모터 수치,
과거 MG90S checkpoint 거부, 현재 MG996R checkpoint 허용을 포함한다.
Python compile 및 PowerShell 문법 검사도 통과했다.

독립 평가: seed 43, 4환경 × 4,100 control step(82초), 20 mm 3단 계단.
완료 episode 4개 중 전체 계단 통과 0개, 낙상 0개, 영역 이탈 2개, timeout 2개다.
최대 몸체 x 전진 0.667744 m, 최소 지형 상대 몸체 여유 0.111226 m,
최대 관절 토크/상한 비율 1.0. **평가 실패**이며 높이 단계는 올리지 않았다.
결과는 `logs/rsl_rl/dksh_cad6_stairs_mg996r_l2_bno086_v1/2026-10-19_12-12-35_463021/run_metadata.json`에
`evaluation.passed=false`, `stair_climbing_verified=false`로 보관한다.

과거 MG90S 실행의 최대 x 전진 0.277652 m보다 이번 값은 커졌다. 하지만 서로 다른 날짜의
짧은 초기 학습/평가이며 완전 통제된 하드웨어 A/B 시험이 아니다. 일부 발의 단계별 착지,
최대 등반 높이 또는 모터 변경만의 인과 효과를 이 집계로 확정하지 않는다.
다음 단계는 횡방향 이탈과 실제 관절 추종/발 접촉 시점을 분리해 검사하고,
실측 영점·PWM·전원/질량 정보를 반영하는 것이다.

수정은 `Bugfix/MLForLegs-MG996RHip`에서 수행하고 계단 작업 브랜치
`Feature/MLForLegs-StairsSensors`에 병합한다. 계단 성공이 미검증이고 해당 기능이
아직 개발 브랜치에 들어가지 않았으므로 `Develop/MLForLegs`에는 함께 병합하지 않는다.
