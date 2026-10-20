# CAD6 지지 궤적 방향 수정과 물리 재검사

2026-10-12 `Bugfix/MLForLegs-YawStanceDirection`에서 확인된 궤적 부호와
배율 오류를 수정했다. CAD 운동학 검사는 통과했지만 실제 물리 보행은
통과하지 못했다. 전진·횡이동 회귀도 있어 Develop에 병합하지 않았다.
이 수정으로 회전 성능이 해결됐다고 해석하지 않는다.

## 확정한 코드 수정

지지 구간의 `travel`은 +0.5에서 -0.5로 감소한다. 정규화 CAD에서 양의
hip은 몸체 +Z축 반시계 회전이므로, 기존 hip 식의 추가 음수는 지지발을
몸체 yaw와 같은 방향으로 움직였다. 고정된 월드 발은 몸체 좌표에서
`dp/dt = -v - omega × p`를 따라야 하므로 hip의 추가 음수를 제거했다.
방사 방향 femur/tibia 식에도 같은 부호 문제가 있어 함께 제거했다.

원래 두 방향에 임의로 적용한 4배 이동량 배율을 CAD 영점 지렛팔에 맞췄다.
hip–foot 평면 거리는 약 0.1796 m, femur/tibia를 반대 방향으로 돌릴 때
방사 방향 미분은 약 0.1000 m/rad다. 기존 좌표 분모를 유지하면서 각각
`0.285 / 0.1796`, `0.2745 / 0.1000`으로 배율을 보정했다.
이는 영점 근처의 근사이며 전체 작업 공간에 대한 정확한 IK가 아니다.

독립적인 URDF 순기구학 테스트로 6개 다리, 전후·좌우·양방향 yaw,
wave/tripod의 지지 방향 72개 조합을 확인했다. 작은 비포화 명령에서
Cartesian 발 속도 크기도 실제 CAD 미분과 20% 이내로 맞는지 검사한다.
기존에는 yaw 24개 조합과 병진 24개 조합이 부호 검사에서 실패했다.
실제 USD도 leg_0 hip을 0에서 +0.1 rad로 직접 설정하여 발이 반시계로
이동하는 것을 확인했다. URDF와 USD 축 방향은 일치한다.

## 회전 판정의 수정

`mean_yaw_rate_rps`는 20 ms마다 읽은 마지막 물리 하위 단계의 몸체 좌표
Z축 각속도 평균이다. 이는 월드 heading의 순변화와 같지 않다. 예를 들어
최종 후보의 마찰 0.8 yaw 단계에서는 이 값이 +0.00332 rad/s였지만
실제 heading 평균 변화율은 -0.000637 rad/s였다. 정지 단계에서도 몸체
각속도 평균 약 +0.00769 rad/s와 heading 변화율 약 -0.00000245 rad/s가
달랐다. 따라서 이 평균값으로 실제 좌우 회전 방향을 단정하면 안 된다.

검사 로그에 root quaternion과 매 제어 단계의 월드 heading 증분을 추가했다.
증분은 atan2(sin(delta), cos(delta))로 +/-pi 경계를 처리하고 자동 리셋
구간은 제외한다. 회전 통과 기준은 명령 방향으로 정규화한 **월드 heading
변화율 > 0.005 rad/s**이다. 숫자 기준은 낮추지 않았다. 기존 몸체 각속도도
진단값으로 보관한다. `--yaw_rate -0.04`로 반대 방향도 동일 기준으로 검사한다.

## 물리 결과

seed 42, 4환경, 원본 CAD convex hull, 물리 5 ms/제어 20 ms에서 정지·전진·
횡이동·yaw를 각각 20초 실행했다. 아래 값은 최종 후보의 zero-action
기준 gait이며 학습 정책 평가가 아니다.

| 마찰 | yaw 명령 rad/s | 전진 변위 m | 횡이동 변위 m | 실제 heading 변화율 rad/s | 판정 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.4 | +0.04 | -0.03208 | -0.00889 | -0.01102 | 실패 |
| 0.6 | +0.04 | -0.00453 | -0.00192 | -0.00316 | 실패 |
| 0.8 | +0.04 | -0.00047 | +0.00061 | -0.000637 | 실패 |
| 0.6 | -0.04 | -0.00453 | -0.00192 | +0.00218 | 실패 |

네 실행에서 낙상·영역 이탈은 0이며 관절 토크 상한은 준수했다.
결과는 `logs/yaw_stance_fix_20261012_final_mu0.4.json`,
`logs/yaw_stance_fix_20261012_final_mu0.6_heading.json`,
`logs/yaw_stance_fix_20261012_final_mu0.8.json`,
`logs/yaw_stance_fix_20261012_final_mu0.6_reverse.json`과 같은 이름의 JSONL에 있다.
초기 비교 실행은 heading을 기록하면서 종전 몸체 각속도 판정을 사용했고,
표는 실제 heading으로 다시 대조한 결과다. 양쪽 기준 모두 실패다.
최종 코드의 월드 heading 판정도 마찰 0.6에서 별도로 실행했다.
`logs/yaw_stance_fix_20261012_final_verified_mu0.6.json`에는
`yaw_gate_metric: unwrapped_world_heading_change_per_elapsed_time`,
heading -0.00316395 rad/s, `passed: false`와 전진·횡이동·회전 실패 원인이
기록됐다. 이 실행에서도 기존 결과가 재현됐다.

전체 단위 테스트는 123개 통과, 1개 skip이다. CAD 방향/이동량과 heading의
각도 경계 처리 테스트를 포함하며, 단위 테스트 통과와 물리 검사 실패를
구분한다. 이전 정책은 변경된 참조 궤적에서 재검증하기 전까지 재사용하지 않는다.

실험 중 CAD6 leg stiffness 8.0, damping 0.10, 더 큰 swing 자세와
목표 변화율 6.25 rad/s도 확인했다. 중간에는 +0.01125 rad/s 회전이 나온
경우가 있었지만 전진이 -0.04984 m로 후퇴하는 회귀가 발생했다.
이 조정은 최종 코드에 포함하지 않았다. 기본 stiffness 3.0/damping 0.06,
변화율 1.5 rad/s, 기존 swing 형상, 초기 자세, 질량 및 모터 한계는 유지한다.

기본 제어 이득에서는 높은 swing 목표에서도 tibia 강체의 바닥 접촉이
유지되는 것을 관찰했다. 다만 이 센서는 발끝만이 아니라 tibia에 붙은
충돌 형상 전체를 측정하므로, 정확한 접촉 부위와 하중 해제 시점을 분리해
조사해야 한다. 다음 단계는 해당 접촉 부위 및 swing 복귀 타이밍과 실제
관절 추종을 검증하는 것이다. 모터 교체가 필요하다는 결론은 내리지 않는다.

## 재현

```powershell
python -m unittest discover -s isaaclab_project/tests -p 'test_*.py'
.\IsaacLab\isaaclab.bat -p .\isaaclab_project\scripts\check_mg90s.py --headless `
  --task Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0 --num_envs 4 --steps 1000 `
  --friction 0.6 --output logs/yaw_stance_fix_20261012_recheck.json
# 같은 명령에 --yaw_rate -0.04를 추가하면 반대 방향을 확인한다.
```

실패 검사는 `MG90S_PREFLIGHT_FAIL`과 JSON `passed: false`를 남긴다.
파일이 생겼거나 Kit가 종료 코드 1로 끝났다는 이유로 성공으로 판정하지 않는다.
