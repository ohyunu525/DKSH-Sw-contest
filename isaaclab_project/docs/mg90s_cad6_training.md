# 6족 CAD · MG90S/RC920DMG 혼합 서보 보행 학습

> 2026-09-22 변경: body-coxa(`hip_joint`)는 MG90S를 유지하고,
> coxa-femur(`femur_joint`)와 femur-tibia(`tibia_joint`)는 RC920DMG로 교체했다.
> RC920DMG는 보수적인 5 V 사양인 19 kgf·cm, 0.16 s/60°, 60 g을 사용한다.
> 이 혼합 서보 구성으로 2026-09-22 사전 검사, 2,000 iteration 재학습,
> 4개 환경·3,000 step 평가를 통과했다.

## Nav2 속도 명령 작업 (권장)

### 속도 추종 재학습 v1 (2026-09-22)

`run_mg90s.ps1 -SpeedProfile velocity`는 이제
`Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v1`을 실행한다.
새 학습 폴더는 `dksh_mg90s_cad6_velocity_l1v2`이며, 기존 v0 체크포인트는
작업 ID 검증에서 거절된다. 기존 v0 등록과 보관 정책은 이전 결과 재현용이다.

- 명령 후보 범위: 전진 -0.025~0.050 m/s, 횡이동 ±0.025 m/s, 회전 ±0.08 rad/s.
- 보행 주기 0.6초, 최대 발 이동량 25mm, lift 5mm, 관절 목표 변화율 6.25rad/s.
- 등가 속도 0.025~0.035m/s에서 wave에서 tripod로 부드럽게 전환한다.
- 병진·회전 복합 명령이 발별 이동 한계를 넘으면 전체 twist를 동일 비율로 축소한다.
  축소된 명령을 관측·보상·발 궤적 모두에 사용한다. 각 축의 최댓값을 동시에
  보장하는 직육면체 명령 영역은 아니다. 외부 명령도 같은 투영을 적용해야 한다.
- 평면 추종 보상 배율 6.0, Gaussian 폭 0.015m/s, yaw 추종 배율 3.0.
- 사전 검사는 정지·최대 전진·최대 횡이동·최대 회전·후진·복합 명령을 확인한다.
  이는 기준 제어기 검사이며 학습된 정책의 속도 정확도를 보증하지 않는다.

```powershell
.\run_mg90s.ps1 -Mode check -SpeedProfile velocity -NumEnvs 4
.\run_mg90s.ps1 -Mode check -SpeedProfile velocity -NumEnvs 8
.\run_mg90s.ps1 -Mode check -SpeedProfile velocity -NumEnvs 16
.\run_mg90s.ps1 -Mode train -SpeedProfile velocity -NumEnvs 32 -MaxIterations 2000
```

평가에는 `evaluate_mg90s_cad6.py --task Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v1`을
사용한다. 평면 오차 0.008m/s, yaw 오차 0.08rad/s의 기존 통과 기준을 유지한다.
평가 기본 시드는 43으로 학습 시드 42와 분리하며 결과 JSON에 기록한다.
아래 학습 완료 수치는 **기존 저속 v0** 정책의 기록이며 v1 재학습 결과가 아니다.

### 기존 저속 v0 기록

실기 계층은 `L1 RM → ROS2 SLAM/Nav2 → /cmd_vel → 저수준 보행 정책`으로 고정한다.
`Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0`는 68개 관측의 `command` 세 값을
몸체 기준 `[vx, vy, wz]`로 모두 사용하며, 기존 직진 정책과 체크포인트를 공유하지 않는다.
학습 폴더는 `dksh_mg90s_cad6_velocity_l1v1`이다. 명령 범위는 전진
`-0.006~0.012 m/s`, 횡이동 `±0.006 m/s`, 회전 `±0.08 rad/s`이고 리셋의 15%는
정지 명령이다. 발 궤적은 각 발의 병진·회전 합성 이동량을 기존 12 mm 도달 범위로 제한한다.

루트 실행기의 기본 프로필은 `velocity`이며 현재 v1을 선택한다. 이전 직진 또는 sprint 정책을 다룰 때만
각각 `-SpeedProfile normal`, `-SpeedProfile sprint`를 명시한다.

```powershell
.\run_mg90s.ps1 -Mode check -SpeedProfile velocity -NumEnvs 4
.\run_mg90s.ps1 -Mode train -SpeedProfile velocity -NumEnvs 32 -MaxIterations 2000
```

물리 사전 검사는 정지·전진·횡이동·회전 기준 궤적을 각각 20초 확인한다. 장시간 학습 전
4→8→16개 환경 검사를 통과하고, 새 작업에서 생성된 `run_metadata.json`과 체크포인트만 사용한다.

2026-09-22 혼합 서보 모델에서 seed 42, 32개 환경의 2,000 iteration 학습을 완료했다.
3,072개 학습 episode에서
낙상·영역 이탈은 0회였으며 모두 시간 종료됐다. 최종 `model_1999.pt`의 별도 4환경·3,000 step
평가도 12 episode 낙상·이탈 0회, 평면 속도 오차 0.00600232 m/s, yaw 오차 0.03714342 rad/s로
통과했다. 선택 체크포인트와 ONNX/TorchScript 내보내기는
`isaaclab_project/policies/mg90s_cad6_velocity_l1v1`에 보관한다.

## 체크포인트 조사 (2026-09-16)

현재 작업 폴더의 `.pt` 160개를 텐서 차원으로 검사했다. 113개는 84관측/24행동,
47개는 86관측/24행동으로 모두 8족이다. 18행동 6족 체크포인트는 없었다.
로컬 Git 참조 전체의 `.pt` 기록도 8족 baseline과 MG90S 정책뿐이다.
`assets/spiderbot_variants/README.md`의 6족 검증은 모델 smoke test이며 학습 완료 기록이 아니다.
상세 경로·차원 목록은 `logs/checkpoint_audit.json`, 재검사 도구는 `scripts/audit_checkpoints.py`다.

따라서 이번 6족 학습은 새 가중치로 시작한다. 이후 생성된 같은 작업의 체크포인트는
정책·관측 정규화·optimizer 상태와 iteration을 복구해 이어서 학습할 수 있다.

## 모델과 조건

- 기존 직진 작업: `Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0`.
- 실제 로드 자산: `assets/spiderbot_variants/spiderbot_6leg/spiderbot_6leg.usd`.
- 6개 다리, 18관절, 19강체, 18행동, 68관측.
- CAD URDF의 짧은 링크와 조립 영점을 사용한 6족 wave gait + ±0.035 rad 관절 보정.
  기존 primitive 8족의 IK는 이 작업에 사용하지 않는다.
- body-coxa는 MG90S 4.8 V 사양(1.8 kgf·cm, 0.10 s/60°), 나머지 두 관절은
  RC920DMG 5 V 사양(19 kgf·cm = 1.8632635 N·m, 0.16 s/60° = 6.54498 rad/s)을 사용한다.
  각 관절 그룹에 DCMotor 속도-토크 포화를 따로 적용한다.
- 최대 출력은 두 모터 모두 임시로 정지 토크의 75%로 제한한다. MG90S는
  0.13239 N·m, RC920DMG는 1.39745 N·m이며 제조사 연속 토크 정격이 아니다.
  RC920DMG의 7.4 V 사양과 8.4 V 상한은 현재 모델에 적용하지 않았다.
- 알려진 다리당 서보 질량은 MG90S 13.4 g 하나와 RC920DMG 60 g 두 개로 133.4 g이다.
  이를 로봇 전체 질량으로 대체하지 않는다. CAD 링크 질량/관성 추정치
  2.31362kg에 L1 RM 0.230kg를 더한 2.54362kg를 사용한다. 실제 배터리 포함
  총중량·나머지 질량 분포는 미측정이다.
- 약 180° 범위 내 기존 ±90° 관절 제한과 95% soft limits를 유지한다.
  dead band 5μs, PWM 보정, 전원/열 특성은 실측 전이므로 아직 모델에 포함하지 않았다.
- 정책 50Hz, 물리 200Hz, 2.4초 wave 주기, lift 4mm, 직진 명령 0.003–0.006m/s.
- PPO: 48-step rollout, 32개 환경, 추가 2,000 iteration, 학습률 0.0001, seed 42.

새 작업은 기존 CAD6 navigation(66관측/18행동)과도 관측·행동 의미가 다르다.
호환 차원과 `run_metadata.json`의 작업 ID를 확인한 체크포인트만 재개/재생한다.

## 실행

아래는 기존 직진 정책을 재현하는 명령이므로 `-SpeedProfile normal`을 명시한다.

```powershell
# 장면/관측 변경 시 4 → 8 → 16 순으로 통과 후 32개 학습
.\run_mg90s.ps1 -Mode check -SpeedProfile normal -NumEnvs 4
.\run_mg90s.ps1 -Mode check -SpeedProfile normal -NumEnvs 8
.\run_mg90s.ps1 -Mode check -SpeedProfile normal -NumEnvs 16
.\run_mg90s.ps1 -Mode train -SpeedProfile normal -NumEnvs 32 -MaxIterations 2000

# 동일 CAD6 작업의 체크포인트에서 추가 학습
.\run_mg90s.ps1 -Mode train -SpeedProfile normal -NumEnvs 32 -MaxIterations 2000 -Checkpoint '.\logs\rsl_rl\dksh_mg90s_cad6_walk\<run>\model_1999.pt'

# 학습된 6족 정책 GUI
.\run_mg90s.ps1 -Mode play -SpeedProfile normal -NumEnvs 1 -Checkpoint '.\logs\rsl_rl\dksh_mg90s_cad6_walk\<run>\model_1999.pt'
```

로그/체크포인트는 `logs/rsl_rl/dksh_mg90s_cad6_walk/<run>/`에 저장된다.
`run_metadata.json`에는 모델 식별, 시작 iteration, 부모 체크포인트 해시, 완료 여부가 기록된다.
완료 표시는 모든 추가 iteration과 최종 체크포인트 저장 후에만 출력한다.
Isaac 실행 경로의 종료 코드 1은 이 완료 표시가 확인된 경우에만 정상 처리한다.

이번 본 학습 폴더는 `2026-09-16_16-45-12_507281_cad6_4v8`이다.
사전 2회 학습 폴더 `2026-09-16_16-42-43_569742_cad6_4v8/model_1.pt`에서
가중치·정규화·optimizer를 복원했고 iteration 2부터 추가 2,000회를 수행한다.
완료 시 기대 파일은 `model_2001.pt`이다. 완료 여부는 `run_metadata.json`의
`status: completed`와 최종 파일로 확인하며, 이 기록 자체는 완료 선언이 아니다.
콘솔 로그는 `logs/mg90s_cad6_launch_2026-09-16_16-45-03/`에 있다.

학습이 `completed`가 된 뒤에는 다음 평가로 60초(3,000 control step)를 확인한다.

```powershell
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\evaluate_mg90s_cad6.py --headless `
  --checkpoint '.\logs\rsl_rl\dksh_mg90s_cad6_walk\<run>\model_2001.pt' --num_envs 4 --steps 3000 `
  --output '.\logs\rsl_rl\dksh_mg90s_cad6_walk\<run>\evaluation_60s.json'
```

통과 기준은 낙상·영역 이탈 0, 평균 전진 0.0025m/s 이상, 평균 속도 오차 0.004m/s 이하,
토크 상한 준수다. 수치와 통과 여부를 모두 저장한다.

## 빠른 프로필

기존 정책은 0.003–0.006m/s만 학습했으므로 빠른 보행 정책으로 사용할 수 없다.
`Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0`는 0.012–0.050m/s 명령, 0.6초 gait 주기,
최대 25mm stride와 6.25rad/s 목표 변화율을 사용한다. 속도 상한은 기존
0.020m/s에서 2.5배 높였다. 0.050m/s × 0.6s × 5/6 = 0.025m이므로
상한에서도 발 궤적은 기존에 검증한 25mm 도달 범위를 넘지 않는다. 목표
변화율은 더 느린 RC920DMG 5 V의 6.545rad/s 무부하 속도 모델 아래로 유지한다. 이 설정은
고속 가능성을 탐색하기 위한 것이며, 실제 하드웨어의 지속 가능 속도를 보증하지 않는다. 0.025m/s 이하에서는 안정적인
wave gait를 유지하고, 0.025–0.035m/s에서는 발 목표를 smoothstep으로 섞어, 그 이상에서는 상호교대 tripod gait로 전환한다. 이 전환은 한 프레임에 바뀌지 않아 발 목표와 지지 상태의 불연속을 방지한다.

Sprint의 속도 추종 보상은 일반 보행보다 강한 6.0 배율과 0.006m/s Gaussian 폭을
사용한다. 단순 전진 보상은 30.0으로 낮춰 목표 속도를 초과하는 행동보다 명령 속도에
맞추는 행동을 우선한다. 자세·횡방향 속도·회전·행동 변화 패널티는 일반 보행과 동일하다.

빠른 기준 gait 4개 환경·20초 검사에서는 낙상 0회, 평균 전진 0.01083m/s,
최대 토크 0.13239N·m, 토크 근접 비율 3.59%를 기록했다. 8개와 16개 환경 검사 뒤
새 가중치로 학습한다. 빠른 작업의 평가/학습 스크립트에는 `--task Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0`를 전달한다.

루트 launcher에서는 다음처럼 실행한다.

```powershell
.\run_mg90s.ps1 -Mode check -SpeedProfile sprint -NumEnvs 4
.\run_mg90s.ps1 -Mode train -SpeedProfile sprint -NumEnvs 32 -MaxIterations 2000
```

## 검증과 해석

URDF 기반 순기구학 대조, IK 왕복, 전체 위상/명령 범위의 관절 한계 및 잔여 보정 여유,
한 번에 한 발 swing 검사를 통과했다. 기존 8족 gait 회귀 검사 포함 5개 테스트가 통과했다.
물리 검사에서는 로드한 USD, 18관절/19강체, 총질량, 토크와 관측 차원을 직접 확인한다.
4개 환경 정지/보행 각 20초 검사: 낙상 0, 보행 평균 전진 0.05652m.
8개 환경 검사: 낙상 0, 평균 전진 0.06479m. 최대 토크는 모두 0.13239 N·m 이하다.
16개 환경도 낙상 0, 평균 전진 0.06620m로 통과했다. 4개 환경에서 PPO 2회와
최종 체크포인트 저장/완료 판정을 확인한 뒤, 생성된 `model_1.pt`에서 32개 환경 추가 학습을 시작한다.
기존 8족 정책을 기본 CAD6 실행에 전달했을 때 시뮬레이터 시작 전에 거절되는 것도 확인했다.
이는 학습 전 기준 궤적 실행 검사다. 학습 결과는 별도 시드에서 60초 직진·속도 추종·낙상률을
평가해야 하며, 이후 회전/정지와 실물 센서/PWM 대응으로 확장한다.
