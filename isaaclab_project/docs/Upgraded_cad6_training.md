# Upgraded CAD6 학습

`HexapodRobot.zip`을 참고한 시범 수집·BC·PPO 환경은
[Upgraded 모방학습 안내](Upgraded_imitation_training.md)와 `Upgraded_run_imitation.ps1`을 사용한다.
아래 PMTG 환경과 관측·행동 계약이 다르므로 체크포인트를 혼용하지 않는다.

`Upgraded-CAD6-PMTG-v0`는 CAD 발끝 좌표와 역기구학을 사용하는 단일 PPO 정책이다.
hip MG996R 6개와 femur/tibia RC920DMG 12개의 v12 물리·접촉 모델을 사용한다.

## 제어와 관측

정책은 다리별 다음 착지점의 XYZ 보정량 18개를 10 Hz에서 출력한다. 각 성분의 범위는
±15 mm다. 발을 들 때 현재 보정량을 저장하며, 해당 이동 구간과 다음 지지 구간 동안
착지점이 연속적으로 이어진다. 양의 높이 보정은 발 이동 중의 최고점도 높인다.

50 Hz 보행 생성기는 주기 2초, 지지 비율 0.65, 기본 들기 높이 20 mm의 tripod를 만든다.
첫 한 주기에는 이동 명령을 서서히 증가시킨다. 지지발의 목표는 명령한 몸통 이동·회전을
상쇄하도록 변환하고, 이동 구간은 위치·속도가 이어지는 Hermite 곡선을 사용한다.
초기 목표 속도는 0.03 m/s다. 현재 환경의 목표점 제어기가 평면 이동 명령을 제공하며,
순수 제어 모듈은 외부에서 주는 vx, vy, yaw-rate 명령도 처리한다.

IK는 현재 URDF의 관절 원점·축·회전과 CAD 말단점을 읽어 계산한다. 각도 범위와 1 mm
위치 오차, 발끝 높이에 대한 비말단 형상의 8 mm 여유 조건을 통과한 목표각을 채택한다.
통과하지 못한 다리는 직전 유효 목표를 유지하고, 거부 비율을 PPO 비용과 로그에 기록한다.
이 여유 검사는 발끝 기준의 국소 수평면 검사다. 실제 지형과 자기충돌은 v12 물리 접촉
판정으로 추가 검증한다. 목표각에는 기존 1.5 rad/s·0.35 필터가 적용된다.

actor는 IMU 각속도·중력 방향, 이동 명령, 필터링한 서보 목표각·목표속도,
보행 생성기 상태, L2 형식의 거리·유효값을 관측한다. 최근 네 프레임을 사용해 총 744개다.
실제 관절각·관절속도, 가상 발 접촉, 정확한 지형 높이와 물체 상태는 critic의 추가
140개 입력에 들어가며 critic은 총 884개를 받는다. 학습 시 IMU 잡음의 표준편차는
각속도 0.01 rad/s, 정규화 중력 성분 0.005다. 두 값은 초기 실험 설정으로 기록한다.
실기의 명령·IMU·L2 전처리를 같은 입력 계약에 연결하는 작업은 별도로 필요하다.

## 실행

Isaac Sim 4.5와 저장소의 Isaac Lab 환경이 설치된 NVIDIA GPU 컴퓨터에서 실행한다.
현재 잠정 CAD 실험은 `-AllowProvisionalCad`로 표시한다.

```powershell
# 4개 환경에서 관측·행동·제어의 유한값을 검사한다.
.\Upgraded_run_cad6.ps1 -Mode check -NumEnvs 4 -AllowProvisionalCad

# PPO 출력 0으로 고정 1 m 평지 보행을 평가한다.
.\Upgraded_run_cad6.ps1 -Mode baseline -NumEnvs 4 -Steps 1200 -AllowProvisionalCad

# 같은 제어 계약으로 평지 검증을 통과한 보고서가 있어야 새 학습을 시작한다.
.\Upgraded_run_cad6.ps1 -Mode train -NumEnvs 4 -MaxIterations 2000 -AllowProvisionalCad

# 학습 완료 메시지의 Upgraded_model_*.pt 경로를 지정한다.
$policy = '<Upgraded_model_*.pt>'
.\Upgraded_run_cad6.ps1 -Mode evaluate -Checkpoint $policy -NumEnvs 4 -Steps 1200 -Seed 307 -AllowProvisionalCad
.\Upgraded_run_cad6.ps1 -Mode play -Checkpoint $policy -NumEnvs 1 -AllowProvisionalCad

# 평지 정책에서 낮은 계단에 전이한다.
.\Upgraded_run_cad6.ps1 -Mode train -Checkpoint $policy -Environment stairs -Difficulty 0 -NumEnvs 4 -AllowProvisionalCad
```

기본 평지 보고서는 `outputs/Upgraded_baseline.json`에 기록한다. 기본 보행의 지속 전진,
목표 도달, 금지 접촉 판정과 IK 거부 비율을 통과해야 새 PPO 학습을 허용한다.
각 실행의 원본 로그와 실패 예외는 `outputs/Upgraded_*.log`에 남는다.
평가 보고서 옆에는 실패 시점의 접촉 진단 JSONL이 생성된다.

체크포인트·학습 메타데이터·환경 설정 파일과 실행 폴더에는 `Upgraded` 접두어를 붙인다.
체크포인트에 CAD, 소스, 관측·행동·제어 계약을 함께 저장하며 재개·평가 때 일치 여부를 검사한다.
TensorBoard가 생성하는 내부 이벤트 파일은 `Upgraded` 실행 폴더 안에 저장된다.

## 검증 범위

CPU 검사는 CAD 순기구학, IK, 지지 구간 높이, 보행 전환, 관측 분리와 체크포인트 계약을
검사한다. 실제 안정 보행, 마찰 대응, 계단 성공과 실기 전달은 GPU 물리 평가가 필요하다.
서보 지연·백래시와 센서 오차의 실측값을 확보하면 해당 분포를 반영해 재학습한다.

```powershell
python -m unittest discover -s isaaclab_project/tests -p 'Upgraded_test_*.py' -v
```
