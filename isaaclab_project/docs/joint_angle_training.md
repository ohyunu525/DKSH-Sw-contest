# 관절각 직접 출력 학습 v2

정책이 관절의 **절대 목표각**을 직접 선택한다. 기본 gait·IK 각도나 ±1° 보정을 더하지 않는다.
혼합 서보 CAD6 실행기는 `run_cad6.ps1`이며 MG996R hip 6개와 RC920DMG 다리 관절 12개를 사용한다.
MG90S 지원과 이전 실행기는 제거했다. Onshape CAD6는 18개 출력으로 각 관절의
-90°~+90° 전체를 선택한다. 공통 실행기에는 MG996R baseline/CAD8 모델도 남아 있다.

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

- `revised_v3`와 CAD6 `revised_v4`는 실제 거리 변화에 양·음 같은 계수를 적용한다.
  이전의 전진 보상 상한은 없애 순전진 보행의 속도 출렁임을 불리하게 자르지 않는다.
  속도 비용은 실제 주기당 거리 감소량에 SmoothL1(Huber)을 적용한다. 순간 속도만 커지는
  떨림은 이득이 없으며, 큰 속도 오차는 제곱 비용 대신 선형으로 증가한다.
- 숨은 최고 전진 거리 상태를 제거하고 필터 가속도 비용에 필요한 목표각 속도를 관측한다.
- 탐색 std는 초기 0.08, 최소 0.03, 최대 0.30으로 물리 변화율 제한과 분리한다.
  과거의 0.02118은 초기 잡음의 포화 확률 1%를 계산한 참고값일 뿐, 학습 중 상한이 아니다.
  새 초기 std는 포화 1%를 보장하지 않는다. 물리 적용 목표각은 여전히 1.5 rad/s로 제한한다.
  초기·최소·최대와 실제 재개 전·후·최종 std, 해당 Adam 상태의 초기화 여부를 기록한다.
- 학습 목표는 0.5~1.5 m, 성공 반경은 0.08 m다. 혼합 CAD6 에피소드는 240초다.
  rollout 128 step, gamma=0.9995, lambda=0.98로 긴 이동의 학습 신호를 보존한다.
- 발 접촉은 위쪽 힘과 실제 발 충돌 형상의 볼록껍질 137개 꼭짓점을 함께 확인한다.
  각 꼭짓점 아래의 지면을 따로 조회하며 뒤꿈치 접지를 발끝 높이 때문에 거절하지 않는다.
  tibia 축 형상은 발 꼭짓점에 포함하지 않는다. 새 허용값은 4 mm이며 미끄럼은 선택한 접지점의
  속도로 계산한다. 실제 접촉점 재구성은 아니며 작은 지형 모서리에서 유한 표본의 한계가 있다.
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
.\run_cad6.ps1 -Mode check -Environment flat -NumEnvs 4

# v1을 이어 쓰지 않고 새 v2 정책 학습
.\run_cad6.ps1 -Mode train -Environment flat -NumEnvs 4 -MaxIterations 4000 -AllowProvisionalCad

# 독립 시드, 고정 1 m 목표로 지속 보행 검증
.\run_cad6.ps1 -Mode evaluate -Environment flat -NumEnvs 4 -Steps 12000 -Seed 73 -Checkpoint '<v2 model_N.pt>' -Output 'logs/joint_angle_flat_v2.json'

# 평지 검증 후 가장 낮은 계단에 전이
.\run_cad6.ps1 -Mode check -Environment stairs -Difficulty 0 -NumEnvs 4
.\run_cad6.ps1 -Mode train -Environment stairs -Difficulty 0 -NumEnvs 4 -MaxIterations 4000 -Checkpoint '<평지 v2 model_N.pt>' -AllowProvisionalCad
.\run_cad6.ps1 -Mode evaluate -Environment stairs -Difficulty 0 -NumEnvs 4 -Steps 24000 -Seed 73 -Checkpoint '<계단 v2 model_N.pt>' -Output 'logs/joint_angle_stairs_v2.json'
.\run_cad6.ps1 -Mode play -Environment stairs -Difficulty 0 -NumEnvs 1 -Checkpoint '<계단 v2 model_N.pt>'
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
모델과 run_metadata.json을 함께 보관한다. 공통 실행기의 MG996R baseline v0는 legacy 모드로 선택한다.
**v1 직접각 모델은 v2에서 재개·재생할 수 없으며 새 가중치로 학습한다.**
작업 ID·CAD 해시·관절 순서·한계·모터·제어·관측·보상·접촉 계약을 검사한다.

행동·관측 크기·물리 제어는 v2 그대로이므로 기존 혼합 CAD6 가중치는 명시적으로 전이할 수 있다.
`legacy_v2`는 수정 전 보상·단일 발끝 접촉을 재현하는 비교 설정이며 MG90S 모드가 아니다.
평가·재생은 옵션을 생략하면 저장된 보상·접촉 설정을 사용한다. MG996R-Hip CAD6는
새 학습에서 `revised_v7`을 사용한다. 다른 JointAngles 작업은 `revised_v3`를 기본으로 유지한다.

`revised_v4`는 CAD 발마다 가상 버튼 접촉 비트 1개를 계산한다. 발끝 지점 근접도와 tibia의
위쪽 접촉력을 함께 사용하고, 누름/해제 힘 문턱으로 히스테리시스를 준다. 6개 비트는
기존 observation의 같은 위치에 들어가므로 입력 크기는 140개로 유지된다. 실제 버튼 부품과
스트로크는 CAD에 없어 이 값은 시뮬레이션 근사이며 학습 metadata에도 그 사실을 남긴다.
`revised_v5`는 이 접촉 모델에 더해, 무릎 6개 중 절반 이상이 차체 바닥 아래에 있는 접근
시간이 전체 접근의 절반을 넘으면 페널티를 준다. 목표 속도는 0.20 m/s이며 속도 오차 비용도
강화해 진행 보상이 목표 속도를 계속 밀어 올리지 않도록 한다.
기존 설정에서 바뀐 설정으로 학습할 때는 `-AllowBehaviorTransfer`로 전이를 명시한다.

`revised_v6`는 버튼이 눌리지 않아도 전체 발 hull의 접지점에서 미끄럼 비용을 계산한다.
`revised_v7`은 이를 유지하면서, **실제 CAD tibia 말단의 마지막 20 mm**만 유효 지지로
인정한다. 말단을 제외한 무릎 쪽 및 중간 shaft 형상이 지면 8 mm 이내이거나 관통한 채 링크에
하중이 걸리면, 말단도 닿아 있더라도 잘못된 접지로 분류한다. 말단/지형 높이 허용치는 4 mm다.
각 다리의 서로 다른 좌표계로 변환한 기존 충돌 형상의 점을 사용하며 collider는 유지한다.

새 보상은 하중이 걸린 다리 중 비말단 접지 비율에 `-15/초`, 각 무릎의 해당 지형에 대한
30 mm 간격 부족 비율에 최대 `-5/초`를 적용한다. 무릎 비용은 즉시 적용하며, 이전
몸체 기준 누적 시간 벌점은 v5/v6 비교용으로 남긴다. 목표 도착은 잘못된 접지가 없어야 한다.
관측의 6개 가상 버튼 비트와 18개 절대각 출력은 유지하지만 접촉·보상 계약이 바뀌므로
기존 정책을 그대로 재생하는 것만으로 새 보행이 만들어지지는 않는다.

평가 JSON에는 `distal_support_fraction`, `non_distal_support_fraction`,
`mean_knee_ground_clearance_m`, `knee_ground_clearance_violation_fraction`이 추가된다.
앞의 두 비율은 **힘 문턱을 넘은 tibia 링크 수**를 분모로 하며, 무접촉 구간은 0이다.
ContactSensor 합력과 기하학으로 추정한 값이며 실제 접촉점이나 하중 분포의 측정치는 아니다.
목표 속도 0.20 m/s, 목표각 속도 제한 1.5 rad/s, 50 Hz 제어는 유지한다.
속도 향상은 재학습 후 순진행 속도·낙상·비말단 접지·미끄럼·포화를 함께 비교해 판정한다.

```powershell
# A 등급 NVIDIA 장비에서 새 말단 지지 정책을 4환경부터 학습
.\run_cad6.ps1 -Mode train -Environment flat -NumEnvs 4 -MaxIterations 4000 -BehaviorProfile revised_v7 -AllowProvisionalCad

# 기존 v2 가중치에서 명시적으로 전이하는 대안
.\run_cad6.ps1 -Mode train -Environment flat -NumEnvs 4 -MaxIterations 4000 -Checkpoint '<기존 model_N.pt>' -BehaviorProfile revised_v7 -AllowBehaviorTransfer -AllowProvisionalCad

# 저장된 v7 접촉/보상 설정으로 평가
.\run_cad6.ps1 -Mode evaluate -Environment flat -NumEnvs 4 -Steps 12000 -Seed 73 -Checkpoint '<새 model_N.pt>' -Output 'logs/distal_support_flat.json'
```

[Dynamic Gait 적용 검토](dynamic_gait_review.md)는 빠른 tripod부터 시작하는 별도 실험과
서보 속도·주기·동적 안정성 제약을 정리한다. 이번 수정에 CPG나 강제 gait는 추가하지 않았다.

`-InitialNoiseStd`, `-MinNoiseStd`, `-MaxNoiseStd`는 탐색 범위를 재정의한다.
재개 기본값은 기존 std를 새 범위로만 제한한다. `-ResetExplorationStd`는 std를 초기값으로
되돌리고 그 매개변수의 Adam 상태만 제거한다. 정책·가치망·관측 정규화·다른 Adam 상태는 보존한다.

## 중단된 model_350 분석과 비교 재개

원본은 `logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v2/2026-10-23_00-54-23_673348/`이다.
4환경×12,000 step 평가의 목표 0/8, 지속 속도 -0.00000159 m/s를 확인했다.
체크포인트의 실제 std는 0.00842~0.01747, 평균 0.01321이며 필터를 지난 상수 평균 탐색의
관절 목표각 표준편차는 약 0.547°다. 모든 관절각을 ±1°로 제한한 것은 아니다.
목표각의 학습 중 평균은 관절별 -0.591°~+0.281°로 조립 영점 주변에 머물렀다.

평가에서 토크 포화 0%, 목표각 속도 포화 0.00278%, 추종 오차 약 0.904°,
평균 지지발 4.496개, 몸통 접촉 0.15%, 평균 기계적 일률 약 0.0451 W였다.
모터 포화보다 좁은 탐색과 보상 설계에 따른 정체를 지지하지만 단독 원인을 확정하지는 않는다.
TensorBoard는 386개 반복(0~385)을 포함하며 학습률 1e-5가 336회 기록됐다.
원본 metadata의 중단 반복 368과 event 최종 반복 385는 서로 다르며 원본은 수정하지 않았다.
평가의 8개 완료 기록은 환경별 4개 결정적 궤적을 두 번 반복했고 평가 seed42도 학습과 같다.

CPU 반례에서는 평균 전진이 목표와 같은 +0.08/-0.04 m/s 보행의 기존 이동 보상이
-1.878/초로 정지의 -0.1804/초보다 낮았다. 수정 후 각각 +0.122/초와 -0.086/초다.
기존 학습 기록에서도 330~350회 순진행 보상 신호의 약 11.1%가 전진 상한에 의해 소실됐다.
이는 보상 계산의 결함을 입증하며, 새 정책의 실제 보행 성공을 입증하지는 않는다.

아래 두 실행은 **같은 원본 checkpoint·32환경·seed42·추가 400회**로 비교한다.
원본에 덮어쓰지 않고 각기 새 로그 폴더를 만든다. 먼저 탐색 변경만 비교할 수 있다.

```powershell
$baseline = 'logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v2/2026-10-23_00-54-23_673348/model_350.pt'

# 기존 보상·접촉을 유지하고 탐색 설정만 변경
.\run_cad6.ps1 -Mode train -Environment flat -NumEnvs 32 -Seed 42 -MaxIterations 400 -Checkpoint $baseline -BehaviorProfile legacy_v2 -ResetExplorationStd -AllowProvisionalCad

# 같은 원본에서 보상·접촉 수정도 적용
.\run_cad6.ps1 -Mode train -Environment flat -NumEnvs 32 -Seed 42 -MaxIterations 400 -Checkpoint $baseline -BehaviorProfile revised_v3 -AllowBehaviorTransfer -ResetExplorationStd -AllowProvisionalCad

# 각각의 새 model_N.pt를 별도 평가
.\run_cad6.ps1 -Mode evaluate -Environment flat -NumEnvs 4 -Steps 12000 -Seed 73 -Checkpoint '<새 model_N.pt>' -Output 'logs/joint_angle_recheck.json'
```

두 비교 실행의 std 설정은 초기 0.08/최소 0.03/최대 0.30으로 같다. 새 PPO 설정의 entropy
계수 0.001도 같으므로 첫 실행은 탐색 설정 묶음의 비교이며 std 상한 하나만의 인과 검정은 아니다.
정지에 머무르면 iteration 수만 늘리기보다 적용 목표각 이동량·실제 관절 이동량·기계적 일률·
추종 오차·지지발 수와 주기당 거리 속도를 함께 확인한다. 평지 평가 통과 뒤 계단으로 진행한다.

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
