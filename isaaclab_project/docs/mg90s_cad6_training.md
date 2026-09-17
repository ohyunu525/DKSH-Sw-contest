# 6족 CAD · MG90S 보행 학습

## 체크포인트 조사 (2026-09-16)

현재 작업 폴더의 `.pt` 160개를 텐서 차원으로 검사했다. 113개는 84관측/24행동,
47개는 86관측/24행동으로 모두 8족이다. 18행동 6족 체크포인트는 없었다.
로컬 Git 참조 전체의 `.pt` 기록도 8족 baseline과 MG90S 정책뿐이다.
`assets/spiderbot_variants/README.md`의 6족 검증은 모델 smoke test이며 학습 완료 기록이 아니다.
상세 경로·차원 목록은 `logs/checkpoint_audit.json`, 재검사 도구는 `scripts/audit_checkpoints.py`다.

따라서 이번 6족 학습은 새 가중치로 시작한다. 이후 생성된 같은 작업의 체크포인트는
정책·관측 정규화·optimizer 상태와 iteration을 복구해 이어서 학습할 수 있다.

## 모델과 조건

- 작업: `Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0`.
- 실제 로드 자산: `assets/spiderbot_variants/spiderbot_6leg/spiderbot_6leg.usd`.
- 6개 다리, 18관절, 19강체, 18행동, 68관측.
- CAD URDF의 짧은 링크와 조립 영점을 사용한 6족 wave gait + ±0.035 rad 관절 보정.
  기존 primitive 8족의 IK는 이 작업에 사용하지 않는다.
- 4.8V 가정: 정지 토크 1.8 kgf·cm = 0.1765197 N·m,
  0.10 s/60° = 10.47198 rad/s 무부하 속도. DCMotor 속도-토크 포화를 적용한다.
- 최대 출력은 임시로 정지 토크의 75%인 0.13239 N·m로 제한한다.
  이는 제조사 연속 토크 정격이 아니다. 실물 전압이 확인되지 않아 6V 수치는 적용하지 않았다.
- 서보 단품 13.4g을 로봇 전체 질량으로 대체하지 않는다. CAD 링크 질량/관성 추정치
  총 2.31362kg를 유지한다. 실제 배터리 포함 총중량·질량 분포는 미측정이다.
- 약 180° 범위 내 기존 ±90° 관절 제한과 95% soft limits를 유지한다.
  dead band 5μs, PWM 보정, 전원/열 특성은 실측 전이므로 아직 모델에 포함하지 않았다.
- 정책 50Hz, 물리 200Hz, 2.4초 wave 주기, lift 4mm, 직진 명령 0.003–0.006m/s.
- PPO: 48-step rollout, 32개 환경, 추가 2,000 iteration, 학습률 0.0001, seed 42.

새 작업은 기존 CAD6 navigation(66관측/18행동)과도 관측·행동 의미가 다르다.
호환 차원과 `run_metadata.json`의 작업 ID를 확인한 체크포인트만 재개/재생한다.

## 실행

루트 PowerShell에서 `run_mg90s.ps1`의 기본 로봇은 이제 **cad6**이다.

```powershell
# 장면/관측 변경 시 4 → 8 → 16 순으로 통과 후 32개 학습
.\run_mg90s.ps1 -Mode check -NumEnvs 4
.\run_mg90s.ps1 -Mode check -NumEnvs 8
.\run_mg90s.ps1 -Mode check -NumEnvs 16
.\run_mg90s.ps1 -Mode train -NumEnvs 32 -MaxIterations 2000

# 동일 CAD6 작업의 체크포인트에서 추가 학습
.\run_mg90s.ps1 -Mode train -NumEnvs 32 -MaxIterations 2000 -Checkpoint '.\logs\rsl_rl\dksh_mg90s_cad6_walk\<run>\model_1999.pt'

# 학습된 6족 정책 GUI
.\run_mg90s.ps1 -Mode play -NumEnvs 1 -Checkpoint '.\logs\rsl_rl\dksh_mg90s_cad6_walk\<run>\model_1999.pt'
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
`Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0`는 0.012–0.020m/s, 1.5초 gait 주기,
최대 25mm stride와 6rad/s 목표 변화율을 사용한다. 0.025m/s 후보는 발 목표 일부가
도달 범위를 벗어나므로 제외했다. 0.020m/s는 18개 관절의 soft limit과 ±0.035rad
정책 보정 여유 안에 있고, MG90S의 10.47rad/s 무부하 속도보다 낮은 기준 궤적 속도를 사용한다.

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
