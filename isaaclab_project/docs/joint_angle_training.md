# 관절각 직접 출력 학습 v2

정책이 관절의 **절대 목표각**을 직접 선택한다. 기본 gait·IK 각도나 ±1° 보정을 더하지 않는다.
두 실행기의 기본값은 `-ControlMode direct`다. Onshape CAD6는 18개 출력으로 각 관절의
-90°~+90° 전체를 선택하며, 과거 CAD8/all-MG90S 모델은 해당 형상의 작업 범위를 유지한다.

## 출력과 관측

출력 순서는 leg_0_hip, leg_0_femur, leg_0_tibia, leg_1_hip, …이다.
PhysX 관절 순서와 이름을 대조하며 다음 식으로 rad 단위 목표각을 만든다.

```text
target = lower + (clamp(action, -1, 1) + 1) / 2 * (upper - lower)
```

혼합 CAD6는 50 Hz에서 목표각 변화 1.5 rad/s, 필터 계수 0.35를 적용한다.
출력 범위를 좁히지 않으므로 같은 출력을 유지하면 선택한 절대각까지 움직인다.
env.joint_angle_targets_rad는 실제 적용 목표각을 반환한다. 초기 자세는 리셋과 마지막 층
초기화에만 쓰며 학습 중 출력에 더하지 않는다.

관측은 **CAD6 140개 / 8족 166개**로 평지와 계단에서 동일하다.
몸통 속도·중력·목표, 실제 관절각·속도·적용 목표각·목표각 속도, 발 접촉,
지면 여유·목표 지면 높이와 지형 48개를 포함한다. 지형은 L2 거리 16개, 유효값 16개,
주변 높이 9개, 바닥 속도 3개, 낙하물 정보 4개다. 낮은 계단을 L2가 놓쳐도 높이 격자로
인식한다. 높이·낙하물 정보는 시뮬레이터에서 제공하므로 실기에는 같은 정보를 만드는 추정기가 필요하다.

## 실패 원인에 따른 수정

- 왕복에도 양의 보상을 주던 Gaussian 속도 보상을 제거했다. 거리 변화의 부호를 유지하고
  전진 보상만 주기당 목표 속도로 제한한다. 후진 비용은 자르거나 자세로 약화하지 않는다.
  속도 오차·정지 비용은 볼록 함수이므로 같은 자리를 흔들며 이득을 얻지 않는다.
- 숨은 최고 전진 거리 상태를 제거하고 필터 가속도 비용에 필요한 목표각 속도를 관측한다.
- 초기 std는 각도 범위·주기·속도 제한으로 계산한다. 혼합 CAD6는 약 0.02118이며, 초기 평균과
  적용 목표가 같을 때 관절별 속도 포화 확률은 1%다. 이전 0.12는 약 65%였다.
  학습 중 std도 같은 상한으로 제한하고 초기·최종 값을 metadata에 남긴다.
- 학습 목표는 0.5~1.5 m, 성공 반경은 0.08 m다. 혼합 CAD6 에피소드는 240초다.
  rollout 128 step, gamma=0.9995, lambda=0.98로 긴 이동의 학습 신호를 보존한다.
- 발 접촉은 위쪽 힘과 발 기준점의 해당 지형 높이를 함께 확인한다. 수평 벽 접촉·공중 tibia 접촉은
  몸통 접촉 비용에 포함한다. 접촉점 자체를 측정하지는 않으며, 긴 발의 뒤꿈치 형상을 포함하도록
  높이 허용값 12 mm를 둔다. 미끄럼은 회전 링크의 발 기준점 속도로 계산한다.
- CAD6 폭 0.56 m, 초기 높이 0.1375 m, 리셋 관절 잡음 0과 자기충돌을 적용한다.
  q=0 최저 충돌체의 지면 여유는 약 10.9 mm다. 실측하지 않은 하드웨어 각도 한계는 만들지 않는다.
- GUI는 현재 USD의 링크 형상을 로컬 좌표 그대로 복제하고 원래 렌더 형상을 숨긴다.
  과거 CAD 메쉬를 현재 관절에 붙이던 어긋남·중복 표시를 제거했다.

관절 끝 범위·속도·토크·기계적 일률 sum(abs(torque*velocity)), 몸통 접촉, 발 미끄럼과
몸통 흔들림에 비용을 준다. 성공은 위치·플랫폼 높이·바른 자세·최소 세 발 지지·몸통 비접촉을
0.20초 유지해야 한다.

## 새로 학습하고 평가하기

Isaac 실행은 [개발 컴퓨터 기준](../../DEVELOPMENT_REQUIREMENTS.md)의 GPU 장비에서 수행한다.
잠정 CAD·서보 가정은 [혼합 서보 문서](mg996r_hip_correction.md)를 따른다.

```powershell
# 초기화·목표각 범위·유한값 검사
.\run_mg90s.ps1 -Mode check -Environment flat -NumEnvs 4

# v1을 이어 쓰지 않고 새 v2 정책 학습
.\run_mg90s.ps1 -Mode train -Environment flat -NumEnvs 4 -MaxIterations 4000 -AllowProvisionalCad

# 독립 시드, 고정 1 m 목표로 지속 보행 검증
.\run_mg90s.ps1 -Mode evaluate -Environment flat -NumEnvs 4 -Steps 12000 -Seed 73 -Checkpoint '<v2 model_N.pt>' -Output 'logs/joint_angle_flat_v2.json'

# 평지 검증 후 가장 낮은 계단에 전이
.\run_mg90s.ps1 -Mode check -Environment stairs -Difficulty 0 -NumEnvs 4
.\run_mg90s.ps1 -Mode train -Environment stairs -Difficulty 0 -NumEnvs 4 -MaxIterations 4000 -Checkpoint '<평지 v2 model_N.pt>' -AllowProvisionalCad
.\run_mg90s.ps1 -Mode evaluate -Environment stairs -Difficulty 0 -NumEnvs 4 -Steps 24000 -Seed 73 -Checkpoint '<계단 v2 model_N.pt>' -Output 'logs/joint_angle_stairs_v2.json'
.\run_mg90s.ps1 -Mode play -Environment stairs -Difficulty 0 -NumEnvs 1 -Checkpoint '<계단 v2 model_N.pt>'
```

계단은 연속 상승 3단과 위쪽 목표 플랫폼이다. CAD6 단 높이는 난이도 0에서 6 mm,
1에서 30 mm다. 별도 평가를 통과하면 0.3 → 0.6 → 1로 올린다. 난이도는 실행 중 고정된다.
평지·계단 입력은 같지만 자동 체크포인트 탐색은 환경별 폴더로 나뉜다.
전이하려면 위처럼 체크포인트를 명시한다.

직접 Python 진입점:

```powershell
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\joint_angle_runner.py --mode=train --task=Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2 --environment=stairs --difficulty=0 --num_envs=4 --max_iterations=4000 --allow_provisional_cad --headless
```

## 체크포인트와 판정

혼합 CAD6 task는 Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2다.
로그는 logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v2[_obstacles]/<run>/에 저장한다.
모델과 run_metadata.json을 함께 보관한다. 기존 v0 gait는 legacy 모드로 선택한다.
**v1 직접각 모델은 v2에서 재개·재생할 수 없으며 새 가중치로 학습한다.**
작업 ID·CAD 해시·관절 순서·한계·모터·제어·관측·보상·접촉 계약을 검사한다.

평가 기본 시드는 학습 42와 다른 73이다. 평지는 고정 1 m 전방 목표, 계단은 고정 코스다.
완료 에피소드별 거리·시간으로 지속 속도를 측정한다. locomotion_pass는 모든 환경의 1회 이상 성공,
완료 에피소드 성공률 80% 이상, 낙상·이탈 0, 목표 속도 50% 이상의 지속 속도를 요구한다.
완료 없이 짧게 실행하면 통과하지 않는다. 목표각·토크 포화, 추종 오차, 발 지지, 진행·총 이동 거리,
기계적 에너지와 거리당 에너지를 JSON으로 남긴다. 배터리 전력·서보 내부 손실은 포함하지 않는다.

DKSH_JOINT_ANGLE_TRAINING_COMPLETE는 추가 학습·최종 저장 완료,
DKSH_JOINT_ANGLE_EVAL_COMPLETE는 평가 실행 완료이며 보행 통과는 JSON에서 확인한다.
CPU 회귀·실제 USD 검사는 보상·관측·표시·계약을 검증한다. PhysX 접촉과 학습된 계단 보행 성능은
GPU에서 위 절차로 검증해야 한다.

```powershell
python -m unittest discover -s .\isaaclab_project\tests -v
```
