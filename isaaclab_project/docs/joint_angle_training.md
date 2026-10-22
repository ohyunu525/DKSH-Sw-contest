# 관절각 직접 출력 학습

`run_isaaclab.ps1`과 `run_mg90s.ps1`의 기본 제어 방식은 이제 `-ControlMode direct`다.
정책이 각 관절의 **절대 목표각**을 선택한다. 기본 gait, IK 발 궤적, 고정 보행각에 대한
±보정이나 이전 각도에 더하는 증분을 사용하지 않는다. `legacy`는 기존 v0 정책 재생용이다.

## 출력 계약

정책 출력은 6족에서 18개, 8족에서 24개다. 순서는
`leg_0_hip`, `leg_0_femur`, `leg_0_tibia`, `leg_1_hip`, ... 이며 실제 PhysX 관절 순서와
별도로 이름을 대조해 적용한다. 정규화 출력 `a`는 다음과 같이 rad 단위 목표각으로 변환한다.

```text
target = lower + (clamp(a, -1, 1) + 1) / 2 * (upper - lower)
```

현재 Onshape CAD6는 세 관절 모두 `-pi/2~+pi/2 rad` 전체를 선택한다. `a=-1`이면
-90°, `a=0`이면 조립 영점 0°, `a=1`이면 +90°다. 같은 출력을 유지하면 같은 각도로
수렴한다. 기본 자세를 더하지 않으며 main URDF의 물리 관절 한계와 교집합을 취한다.
과거 CAD8/all-MG90S CAD 모델은 기존 `-0.40~1.40 rad` 무릎 작업 범위를 유지한다.

50 Hz 제어에서 목표각은 저역 필터와 변화 속도 제한을 거친다. 기존 기본 모델의 상한은
MG996R 3 rad/s, MG90S 2 rad/s이며, 혼합 CAD6 task는 전체 관절에 1.5 rad/s를 적용한다.
이는 정책의 출력 범위를 좁히는 잔여 보정이 아니라
실제 모터에 전달하는 명령의 급격한 변화를 제한하는 장치다. 정책은 현재 적용된 목표각을
관측하므로 필터 상태를 알 수 있다. `env.joint_angle_targets_rad`로 적용 목표각을 읽을 수 있다.
관절 작업 범위는 PhysX 관절 한계에도 적용한다. 현재 Onshape CAD6의 자기충돌은
main 설정에 따라 비활성화하며 과거 CAD8/all-MG90S task는 기존 설정을 유지한다.

초기 자세는 리셋과 새 정책의 마지막 층 bias 초기화에만 사용한다. 마지막 층은 학습 가능하며
이후 어떤 고정 보행도 덧붙이지 않는다. 각도 범위·속도·보상 배율은
`joint_angle_env_cfg.py`에 모아 두었다. CAD와 primitive의 무릎 영점이 달라 범위도 구분한다.
범위는 시뮬레이션 작업 범위이며 실물 PWM 보정값이 아니다. 현재 CAD6 혼합 서보 task는
MG996R hip과 RC920DMG leg actuator를 적용한다. 가정의 출처와 한계는
[MG996R hip 보정 문서](mg996r_hip_correction.md)에 기록한다.

## 학습 의도

- 목표까지 **처음 줄인 거리**에 보상한다. 앞뒤로 흔들며 같은 구간을 반복해도 추가 보상은 없다.
- 목표 방향 속도를 추종하고, 관절 끝 범위·관절 속도·토크·실제 기계적 일률 `sum(abs(torque * velocity))`에 비용을 준다.
- 몸통·hip·femur로 바닥을 끌거나 접지한 발이 미끄러지면 비용을 준다. 발 미끄럼은 회전하는 링크 중심 대신 tibia 끝점 속도로 계산한다.
- 도착 판정은 목표 반경, 목표 지면 위의 몸통 높이, 바른 자세, 최소 세 발 지지, 몸통 비접촉을 0.20초 유지해야 한다. 계단 아래에서 XY 좌표만 맞추거나 넘어져 도달하면 성공이 아니다.

관측은 6족 평지 74 / 비평지 106개, 8족 평지 94 / 비평지 126개다.
관절각·속도·적용 목표각, 접지, 지면 여유 높이, 목표 지면 높이와 기존 속도·중력·목표 방향을
포함한다. 비평지에서는 main의 L2 거리 16개와 유효값 16개를 추가한다.
main이 반환하는 별도 critic 전용 16개 값은 v1 정책에 넣지 않는다.

## 실행과 난이도 단계

실제 Isaac 실행은 [개발 컴퓨터 기준](../../DEVELOPMENT_REQUIREMENTS.md)의 A 등급 장비에서
Unity를 종료한 뒤 수행한다. 이 변경의 로컬 CPU 단위 검사는 PhysX나 학습 완료 검증이 아니다.
처음에는 4개 환경 smoke, 이후 8 → 16 → 32개 순으로 확인한다.

```powershell
# MG996R hip / RC920DMG leg CAD6: 새 직접 각도 환경의 물리 실행 확인
.\run_mg90s.ps1 -Mode check -Environment stairs -Difficulty 0 -NumEnvs 4

# 가장 낮은 계단부터 새 가중치로 학습
.\run_mg90s.ps1 -Mode train -Environment stairs -Difficulty 0 -NumEnvs 4 -MaxIterations 4000 -AllowProvisionalCad

# 이전 단계의 새 v1 모델로 난이도를 높여 추가 학습
.\run_mg90s.ps1 -Mode train -Environment stairs -Difficulty 0.3 -NumEnvs 4 -MaxIterations 2000 -Checkpoint '<이전 단계의 model_N.pt>' -AllowProvisionalCad

# 별도 시드로 충분한 길이의 유한 평가 / 재생
.\run_mg90s.ps1 -Mode evaluate -Environment stairs -Difficulty 0.3 -NumEnvs 4 -Steps 12000 -Seed 73 -Checkpoint '<model_N.pt>' -Output 'logs/joint_angle_stairs_evaluation.json'
.\run_mg90s.ps1 -Mode play -Environment stairs -Difficulty 0.3 -NumEnvs 1 -Checkpoint '<model_N.pt>'

# 기본 8족 all-MG996R 모델은 공통 실행기에서 별도 선택
.\run_isaaclab.ps1 -Mode train -RobotModel baseline -ServoModel mg996r -Environment stairs -Difficulty 0 -NumEnvs 4
```

`stairs`는 세 개의 연속 상승 단과 위쪽 목표 플랫폼이다. CAD 기준 단 높이는 난이도 0에서
6 mm, 1에서 30 mm다. 먼저 난이도 0에서 별도 평가를 통과한 뒤 0.3 → 0.6 → 1로 올린다.
난이도는 실행 시 고정된다. 통과 여부를 확인하지 않고 자동으로 높이지 않는다.
기존 `rough`는 떨어져 있는 세 요철로 유지한다.

Python을 직접 사용할 때는 다음 진입점을 사용한다.

```powershell
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\joint_angle_runner.py --mode=train --task=Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v1 --environment=stairs --difficulty=0 --num_envs=4 --max_iterations=4000 --allow_provisional_cad --headless
```

## 체크포인트와 평가

v1 작업 ID는 `Isaac-DKSH-{Spider|MG90S}[-CAD6|-CAD8]-JointAngles-v1`과
`Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v1`이다.
혼합 CAD6 로그는 `logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v1[_obstacles]/<run>/`에,
MG90S 보존 실험은 해당 MG90S 전용 경로에 저장한다.
`run_metadata.json`은 작업 ID, 출력 의미, 관절 순서·각도 범위, 관측 수, 시드, 부모 체크포인트,
학습 시작·완료 상태와 최종 체크포인트 해시를 기록한다. **모델 파일과 metadata를 함께 보관한다.**

v0의 68관측 CAD6 gait 보정 모델이나 66관측 navigation 모델은 새 제어에 사용할 수 없다.
같은 18행동이라도 의미가 다르므로 새 가중치로 학습한다. 평지와 비평지도 입력 수가 달라
직접 재개할 수 없다. 같은 v1 로봇·서보의 비평지 환경과 난이도끼리는 재개할 수 있다.
관절 순서나 작업 범위가 변경된 모델도 거절한다. Onshape CAD6는 정규화 URDF의 SHA-256을
검사하고, 비평지 task는 `unitree_l2_ranges16_valid16_v1` 관측 계약을 검사한다.
main 병합 전 CAD/지형 관측으로 학습한 v1 모델은 차원이 같아도 새 가중치로 학습한다.

`DKSH_JOINT_ANGLE_TRAINING_COMPLETE`는 요청한 추가 iteration과 최종 저장이 완료되어야 출력된다.
평가는 성공·낙상·영역 이탈·시간초과 횟수, 목표 방향 속도, 관절 한계 비용, 몸통 접촉,
기계적 에너지와 새 이동 거리당 에너지를 보고한다. `locomotion_pass`는 최소 환경 수만큼의
목표 도달, 완료 에피소드 성공률 80% 이상, 낙상·영역 이탈 0을 모두 요구한다.
`DKSH_JOINT_ANGLE_EVAL_COMPLETE`는 평가 실행 완료만 뜻하며 보행 통과 여부는 JSON 필드로 확인한다.
에너지 수치는 배터리 전력이나 서보 손실을 포함하지 않으며 같은 조건의 정책 비교에 사용한다.

기존 모델을 재생하려면 명시적으로 `-ControlMode legacy`를 전달한다.
기존 Sprint gait도 `-ControlMode legacy -SpeedProfile sprint`에서만 사용한다.

```powershell
python -m unittest discover -s .\isaaclab_project\tests -v
```
