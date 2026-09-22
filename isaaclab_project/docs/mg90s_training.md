# MG90S 예비 보행 학습

> 2026-09-22 이후 실행 구성은 body-coxa에 MG90S, coxa-femur와
> femur-tibia에 RC920DMG를 사용하는 혼합 서보 모델이다. 아래 내용과 결과는
> 세 관절 모두 MG90S였던 과거 8족 실험 기록이며 새 구성의 검증 결과가 아니다.

> 2026-09-16 정정: 실물은 **6족**이다. 이 문서와 기존 정책은 8족 예비 실험 기록이다.
> 현재 기본 실행은 [6족 MG90S 학습](mg90s_cad6_training.md)을 따른다.
> 아래 과거 실행을 재현할 때만 `run_mg90s.ps1`에 `-RobotModel legacy8`을 명시한다.

## 범위와 가정

`Isaac-DKSH-MG90S-Walk-Direct-v0`는 기존 primitive 8족 기하를 사용하는 별도 실험이다.
CAD 6족/8족, 기존 MG996R 환경 및 이전 체크포인트는 그대로 보관한다.
실물 공급 전압·총중량·링크 질량 배분은 아직 확인되지 않았다.
`DEVELOPMENT_REQUIREMENTS.md`는 개발 PC 사양과 운영 기준을 정하며, 이 실물 값들은 명시하지 않는다.

- 공급 전압 가정: 4.8 V.
- 사용자가 제공한 1.8 kgf·cm → 정지 토크 0.1765197 N·m.
- 0.10 s/60° → 무부하 속도 10.47198 rad/s.
- 서보 단품 질량: 13.4 g. 24개는 321.6 g이며 로봇 총중량이 아니다.
- 기존 USD 링크 질량 2.91816 kg 및 관성을 유지하고 L1 RM 0.230 kg 페이로드를 합산한다.
  기존 링크의 100 g을 서보 13.4 g으로 치환하거나,
  근거 없이 MG996R 55 g을 빼서 총중량을 낮추지 않는다.
- 출력 토크 상한은 사양 정지 토크의 75%인 0.13239 N·m로 임시 제한한다.
  **이 75% 값은 제조사의 연속 토크 등급이 아니다.**
- DCMotor의 속도-토크 포화 모델을 적용한다. 정지 토크와 무부하 속도를 동시에 출력하지 않는다.
- PD, armature, 지면 마찰은 실측되지 않은 시뮬레이션 가정이다.
- 회전 범위는 기존 ±90° joint limits 및 95% soft limits를 사용한다.
  실물 장착 영점, PWM-각도 보정, 충돌 여유, dead band, 전원 강하와 열 특성은 별도 측정해야 한다.

제조사 페이지는 사용자가 제공한 표와 일부 값(상위 전압 토크, dead band, 치수)이 다르다.
이번 실행은 두 자료의 공통 4.8 V 토크·속도를 사용하며, 상위 전압을 실물에 인가하는 절차는 포함하지 않는다.
근거: [TowerPro MG90S](https://towerpro.com.tw/product/mg90s-3/),
[Isaac Lab 2.1 actuator](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/api/lab/isaaclab.actuators.html).

## 학습 구조

다리를 몸 아래에 모은 자세에서 시작해, 역기구학으로 한 다리씩 들어 올리는 wave gait를 만든다.
정책은 이 기준 궤적에 작은 관절각 보정을 더한다. 목표는 평지의 느린 직진 속도 추종이다.

- 행동: 24차원, 보정 범위 ±0.035 rad, 지수 평활화와 목표각 변화율 제한.
- 관측: 86차원(몸체 선/각속도, 중력 방향, 속도 명령, 관절 상태, 이전 필터 행동, 보행 위상).
- 물리 200 Hz, 정책 50 Hz; 1초 settling 뒤 1초 동안 gait 진폭 증가.
- 명령 속도: 0.004–0.010 m/s; 보행 주기 2초, 발 높이 6 mm.
- 학습: 32개 환경, 48-step rollout, 2,000회 PPO, 고정 학습률 0.0001, 탐색 표준편차 0.15.
- 새 관측·행동 의미와 모터 모델을 사용하므로 이전 84차원 MG996R 정책에서 resume하지 않는다.
- 보상은 속도 추종 및 전진, 측방 이동·기울기·관절 보정 비용을 포함한다.

현재 정책 관측은 시뮬레이터 상태이다. 실물 MG90S의 관절 피드백 확보 또는 추정기가 필요하다.

## 실행과 확인

프로젝트 루트 PowerShell:

```powershell
# 4개 환경에서 정지 20초 + 기준 보행 20초, 토크/낙상/변위 검사
.\run_mg90s.ps1 -Mode check -NumEnvs 4

# DEVELOPMENT_REQUIREMENTS.md에 따라 앞 단계 통과 후 순차 검사
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\check_mg90s.py --headless --num_envs=8 --output=logs/mg90s_preflight_8.json
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\check_mg90s.py --headless --num_envs=16 --output=logs/mg90s_preflight_16.json

# 창 없이 학습 (GUI는 -Gui 추가)
.\run_mg90s.ps1 -Mode train -NumEnvs 32 -MaxIterations 2000

# 반드시 MG90S 실험에서 나온 체크포인트를 사용
.\run_mg90s.ps1 -Mode play -NumEnvs 1 -Checkpoint 'C:\path\to\model_1999.pt'
```

실험 로그: `logs/rsl_rl/dksh_mg90s_walk/<timestamp>_4v8_mass_unmeasured/`.
사전 검사: `logs/mg90s_preflight.json`. 정책 학습 성공률이 아닌 기준 제어기의 물리 실행 검사다.
사전 검사 통과 기준: 두 구간 모두 낙상·영역 이탈 0회, 토크 상한 준수, 보행 구간 평균 전진 5 mm 초과.
새 환경의 `Episode_Count/*`는 각 종료 시점의 실제 개수이며, 종료 없는 스텝에 이전 값을 반복하지 않는다.
최종 평가에서는 누적 종료 개수로 낙상률을 계산해야 한다.
장면이나 관측 차원이 바뀌면 다시 4개부터 검사한다. Isaac 실행 중 Unity Editor와 전체 Nav2 stack은
동시에 실행하지 않는다.

## 구현 검증 기록 (2026-09-15)

- wave gait 단위 테스트 3개 통과: IK/FK 일치, 관절 범위·연속성, 한 번에 한 발만 swing, 초기 자세 확인.
- 4개 환경 물리 검사 통과: 정지/보행 각각 20초, 낙상·영역 이탈 0회.
  기준 보행 평균 X 변위 0.08288 m, 최소 몸체 높이 0.21879 m, 최대 토크 0.13239 N·m.
  이는 학습 전 기준 제어기 결과이며, 학습된 정책의 성능이나 실물 가능성을 증명하지 않는다.
- 이어서 8개, 16개 환경도 순차 통과했다. 각 검사 정지/보행 20초에서 낙상·영역 이탈 0회,
  보행 평균 X 변위는 각각 0.09833 m, 0.09913 m였으며 최대 토크는 모두 0.13239 N·m였다.
  원시 결과는 `logs/mg90s_preflight_8.json`, `logs/mg90s_preflight_16.json`에 있다.
  이 PC의 Kit 4.5는 검사 종료 때 exit code 1을 반환하므로, 물리 검사 결과와
  `MG90S_PREFLIGHT_PASS` 표시를 함께 확인했다. 일반 학습 오류는 이 방식으로 무시하지 않는다.
- 전체 Python 테스트 실행에서는 기존 `test_environment_cli`의 seed 전달 기대값과 구현이
  다른 하위 검사 2건이 실패했다. 해당 CLI/기존 테스트는 이번 변경에서 수정하지 않았다.
- 최초 32개 예비 학습은 4 → 8 → 16 → 32 증설 규칙을 적용하기 위해 중단했으며,
  해당 로그와 체크포인트는 삭제하지 않았다. 최종 실행은 중간 환경 수의 검사 후 새 실험으로 시작한다.

## 선택된 정책 (2026-09-15)

학습은 32개 환경, seed 42, PPO 2,000 iteration(3,072,000 transition)으로 완료했다.
선택된 최종 체크포인트는 `isaaclab_project/policies/mg90s_8leg_4v8/model_1999.pt`이며,
SHA-256은 `EF1B58921216314A1DDB6353A6DCA3110C80202876632C53E7DF11FE9F998316`이다.

마지막 학습 로그는 명령 속도 0.0061 m/s, 전진 속도 0.0060 m/s, 속도 오차 0.0028 m/s,
upright 1.0000, 토크 제한 근접 비율 0.1636을 기록했다. 이는 학습 롤아웃의 지표이며,
독립 시드·장기 평가 또는 실물 결과가 아니다.

## 다음 단계

1. 공급 전압·실물 다리 수·전체 및 링크별 중량을 확인해 가정 교체.
2. 독립 시드에서 60초 연속 직진, 명령 속도 오차·낙상·토크 포화·발 미끄럼 측정.
3. 회전·정지 명령 추가, 속도 범위를 점진적으로 확대.
4. 전원/마찰/지연/질량 오차를 모델링하고 실물 관측·PWM 경로를 맞춤.
5. 평지 검증을 통과하면 LiDAR/ROS 경로 추종, 이후 경사와 장애물로 확장.

학습 시작은 보행 완성 또는 실물 작동 검증을 뜻하지 않는다.
