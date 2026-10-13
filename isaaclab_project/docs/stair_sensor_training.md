# L2 + BNO086 가정 기반 CAD6 계단 학습

이 작업은 2026-09-28 Onshape 내보내기 CAD, MG90S 6개/RC920DMG 12개,
기존 토크·속도·관절 제한으로 수행하는 **예비 시뮬레이션**이다.
실물 등반 가능 높이를 보증하지 않는다. CAD 질량·모터 배치·연속 토크 검증은 미완료다.

## 관측과 센서 가정

작업: `Isaac-DKSH-MG90S-CAD6-Stairs-Sensors-Direct-v0`.
18개 위치 제어 action, 240개 관측. 평지 68개 관측 체크포인트와 호환되지 않는다.

- BNO086: 몸체 좌표 각속도 3개와 정규화 중력 방향 3개. 50 Hz 제어 소비,
  한 제어 주기(20 ms) 지연. 각속도 표준편차 0.01 rad/s, 중력 성분 0.01의
  가우시안 오차는 **가정**이며 제조사 성능 곡선이 아니다.
- L2: 뒤집어 장착한 하향 시야 -90°~+6° 가정. 360° 방위 10° 간격,
  고도 6° 간격, 612개 근사 광선. 5.55 Hz 갱신, 근거리 50 mm 제한,
  로컬 처리 거리 1.5 m. 실제 Unitree 비반복 주사나 전체 64,000 points/s를 재현하지 않는다.
- 광선과 계단 박스/바닥의 가장 가까운 교차점만 사용해 차폐를 처리한다.
  거리 잡음 표준편차 10 mm와 누락률 5%는 가정이다. 재질/반사율·로봇 자체 차폐는 미구현이다.
- 관측점만 12×8 로컬 격자로 모은다. 높이와 관측 유효성 마스크를 각각 제공한다.
  보이지 않는 셀에 계단 정답 높이를 채워 넣지 않는다.
- 나머지는 속도 명령 3개, 필터링 action 18개, **명령한** 관절 목표 18개,
  위상 sin/cos 2개, 라이다 데이터 나이 1개다. 실제 관절각/관절속도/몸체 병진속도,
  발 접촉력, 계단 높이·단계·월드 위치는 actor 입력에 없다.
- 라이다 좌표 변환에는 시뮬레이션 몸체 자세를 사용한다. 정합 오차, 실제 드라이버,
  자기장 간섭, 센서 탈락 등은 검증하지 않았다. `hardware_deployable=false`.

센서 자료: [Unitree L2 공식 사양](https://www.unitree.com/cn/mobile/L2/),
[BNO086 공식 안내](https://docs.sparkfun.com/SparkFun_VR_IMU_Breakout_BNO086_QWIIC/introduction/).
L2 내장 IMU와 별개로 BNO086 자세 출력이 있다고 가정한다. 두 IMU를 무조건 합산하지 않는다.
L2 230 g 질량은 기존 payload 모델을 그대로 적용한다. 장착 방향·브래킷·배터리 포함
실물 무게중심과 자체 차폐는 별도 확인해야 한다.

## 계단과 통과 기준

첫 모서리 x=0.40 m, 한 단 깊이 0.25 m, 폭 1.2 m, 3단과 상단 여유 0.60 m.
한 단 높이는 20/40/60/80/100/120 mm다. 120 mm는 실현 가능하다고 확인한 높이가
아니라 실험 지형의 상한이다. 충돌 Mesh와 광선 센서는 동일 박스 치수를 사용한다.
마찰은 기존 0.6, 물리 5 ms/제어 20 ms, 기존 CAD convex hull 충돌을 유지한다.

각 환경은 20 mm에서 시작한다. 몸체가 상단 충분히 안쪽(x>1.40 m)에 도달하고,
높이·자세·영역 조건을 만족한 채 1초 유지하면 예비 성공으로 기록한다.
연속 3회 성공해야 한 단계 높인다. 낙상/영역 이탈 시 한 단계 낮추고,
timeout은 성공으로 인정하지 않는다. 정책에 이 판정의 정답 지형을 제공하지 않는다.
현재 판정은 몸체 중심과 유지 시간 기반이며 각 발의 안정 접촉/슬립을 실기 수준으로
검증하는 기준은 아니다. 몸체 지면 여유/넘어짐은 지형 상대 좌표로 판정한다.

보상은 실제 전진 변위, 상단 유지 성공, 자세·횡방향·급격한 action 변경·토크 포화
페널티다. 몸체 z만 올리거나 지형 높이만 맞추는 보상은 없다.
관절 목표 residual 범위는 0.8 rad로 늘리되 실제 위치·토크·속도 제한은 유지한다.

독립 평가는 학습 seed 42와 다른 기본 seed 43을 사용한다. 지정한 높이에 고정하고,
최소 80초 이상 실행하여 완료 episode 수 ≥ 환경 수, 성공률 ≥80%, 낙상·영역 이탈 0,
토크 제한 준수를 모두 요구한다. 이 평가 통과도 실물 인증은 아니다.

## 실행

현재 CAD는 provisional이므로 가정 실행에는 명시적 override가 필요하다.

```powershell
.\run_stairs.ps1 -Mode train -NumEnvs 16 -MaxIterations 300 -AllowProvisionalCad
.\run_stairs.ps1 -Mode evaluate -NumEnvs 4 -Level 0 -Steps 4100 `
  -Checkpoint 'logs/rsl_rl/dksh_cad6_stairs_l2_bno086_v1/<run>/model_299.pt' -AllowProvisionalCad
```

`-Level 0..5`는 각 높이에 대응한다. 학습 resume는 같은 task/관측 계약의 체크포인트만
허용한다. PPO/정규화 상태는 복원하지만 curriculum과 episode는 지정한 초기 level에서
새로 시작한다. 재개 실행의 추가 iteration 수와 원본 checkpoint hash를 기록한다.

결과는 `logs/rsl_rl/dksh_cad6_stairs_l2_bno086_v1/<run>/`의 checkpoint,
`run_metadata.json`, TensorBoard, config, source snapshot에 남는다.
`STAIRS_RUN_COMPLETE`와 metadata 완료 상태가 있어야 학습 완료로 인정한다.
학습 완료와 계단 평가 통과는 별개다. Kit의 종료 코드 1만으로 실패/성공을 단정하지 않는다.

## 2026-10-13 실행 기록

4환경 × 2 iteration smoke 실행에서 센서 관측, PPO, checkpoint 저장까지 확인했다.
단위 테스트 133개 중 132개 통과, CUDA 비교 1개 skip. PowerShell 문법과 Python compile 검사 통과.

초기 학습: seed 42, 16환경 × 300 iteration, 230,400 transition.
2026-10-13 20:48:03~20:53:30 KST 실행 완료.
학습 중 완료 episode 49개: 성공 0, 낙상 0, 영역 이탈 3, timeout 46.
높이 단계는 20 mm에 머물렀다. **PPO 실행은 완료했으나 등반 성공은 확인하지 못했다.**

로컬 결과:
`logs/rsl_rl/dksh_cad6_stairs_l2_bno086_v1/2026-10-13_20-48-03_912260/model_299.pt`.
SHA256: `e8fc0372b702a8d61f9149c4c28b00f1e965abd40934a51594ef7189ed13e62b`.
같은 디렉터리의 `run_metadata.json`에 `status=completed`,
`stair_climbing_verified=false`와 정확한 센서/CAD 가정이 저장됐다.
checkpoint와 원본 실행 로그는 Git ignore 대상이며 이 PC에만 보관한다.

독립 평가: seed 43, 4환경, 4,100 control step(82초), 한 단 20 mm의 3단 계단.
완료 episode 4개 중 전체 계단 통과 0개, 낙상 0개, 영역 이탈 2개, timeout 2개.
최대 몸체 x 전진 0.277652 m, 최소 지형 상대 몸체 여유 0.118751 m,
최대 관절 토크/상한 비율 1.0. 판정은 **실패**이며 높이 curriculum은 올라가지 않았다.
일부 발이 첫 단에 올라갔는지까지는 이 집계만으로 판단하지 않는다.
최대 등반 높이나 기계적 불가능성을 이 짧은 초기 학습으로 확정하지 않는다.

평가 결과:
`logs/rsl_rl/dksh_cad6_stairs_l2_bno086_v1/2026-10-13_20-53-53_578449/run_metadata.json`.
`evaluation.passed=false`, `stair_climbing_verified=false`를 기록했고 실행기는
정상적으로 실패를 반환했다. 다음 실험은 높이를 올리기 전에 횡방향 이탈,
실제 관절 목표 추종, swing 시 tibia 접촉 부위를 시간별로 분리해 검증해야 한다.
300 iteration은 초기 실행 결과이며 충분히 수렴한 정책으로 부르지 않는다.

`run_stairs.ps1`을 통한 같은 checkpoint 재개(1환경 × 추가 1 iteration)도 완료했다.
원본 iteration 299에서 300으로 이어지고, 원본 hash와 새 checkpoint가 기록됐다.
이는 재개 기능 검사이며 추가 계단 성공 증거가 아니다.

개발 이력: `Develop/MLForLegs`에서 `Feature/MLForLegs-StairsSensors`를 만들고,
기존 CAD stance/heading 보정 커밋을 의존성으로 가져왔다. 설치 버전의 quaternion API와
configclass 호환 오류는 `Bugfix/MLForLegs-StairsRuntime`에서 수정 후 feature에 병합했다.
등반 실패와 기존 평지 물리 실패가 남아 있으므로 `Develop/MLForLegs`에는 병합하지 않는다.
