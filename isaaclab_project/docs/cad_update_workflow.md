# Spiderbot CAD 개조 안내

[HEXAPOD Onshape 조립 모델](https://cad.onshape.com/documents/f646c0fde8bd91cfa43cbae7/w/279609403375cff4404c6eae/e/76a6985095c8e8f19d67aec6)에서 조립과 부품 형상을 살펴볼 수 있습니다.
Onshape에서 복사 또는 내보내기 메뉴가 제공되면, 복사본에서 개조하거나 STEP으로
내보내 다른 CAD 도구에서 편집할 수 있습니다.

GitHub에서 바로 받을 수 있는 파일도 있습니다.

- [RLForHexapod URDF](../../RLForHexapod/src/RLforHexapod/assets/urdf/robot.urdf)와
  [STL 메시](../../RLForHexapod/src/RLforHexapod/assets/assets)는 현재 학습 프로젝트의 로봇 자산입니다.
- [이전 Onshape 내보내기](../assets/spiderbot_variants/onshape_20260928)는 예비 6족 조립의 URDF와 메시입니다.
- [Unity 프리팹과 6족·8족 예비 모델](../assets/spiderbot_variants/README.md)은 Unity에서 모델을 바꾸거나
  시뮬레이션용 자산을 재생성할 때 사용할 수 있습니다.

## 확정한 관절 영점과 회전 방향

관절 규약의 기계 판독 원본은
`assets/spiderbot_variants/onshape_joint_frames.json`이다. Onshape에 저장된 조립 자세를
모든 관절의 `q=0`으로 정의한다. 이 자세에서 coxa는 몸체 중심으로부터 방사형,
femur는 수직 아래, tibia는 수평 바깥쪽을 향한다. Onshape Mate 편집기에 표시되는
각도는 조립 구속조건의 내부 좌표이므로 시뮬레이션 영점 오프셋으로 사용하지 않는다.
실제로 저장 자세의 L1 tibia에는 약 92.119도가 표시됐다.

몸체 좌표는 +X 전방, +Y 좌측, +Z 위쪽이며 다리는 위에서 봤을 때 반시계 방향으로
`leg_0`부터 `leg_5`까지 센다. 최신 Onshape 이름과의 대응은 차례로
`L6, L5, L4, L3, L2, L1`이다. 새 URDF/USD 생성 시 coxa 축은 `(0, 0, +1)`,
femur와 tibia 축은 각 다리의 로컬 좌표에서 `(0, -1, 0)`으로 재구성한다. 따라서
양의 coxa 회전은 위에서 반시계, 양의 femur 회전은 영점에서 링크 끝을 바깥쪽으로,
양의 tibia 회전은 발을 위로 접는다. 이 규약은 기준 자세가 다른 기존
`spiderbot_6leg.urdf`에 소급 적용하지 않는다.

세 관절 종류 모두 정규화 영점 기준 `-90도 ~ +90도`, 총 180도의 위치 한계를
사용한다. 이것은 학습과 제어를 위한 프로젝트 소프트 한계다. 현재 Onshape Revolute
Mate의 Limits는 비활성화되어 있으므로, 실제 조립 후 간섭이 발견되면 좌우 한계를
더 좁히되 180도보다 넓히지는 않는다.

## 개조한 모델을 학습 환경에 반영하기

1. URDF와 메시를 함께 수정합니다. 메시 경로는 URDF 파일을 기준으로 유지합니다.
2. 관절 이름·축·영점·한계와 링크 질량·관성을 확인합니다. 관절 이름을 변경했다면
   [학습 환경 설정](../../RLForHexapod/src/RLforHexapod/tasks/walk/config/hexapod/env_cfg.py)도 맞춰야 합니다.
3. [학습 환경 설치](../../README.md#rlforhexapod-학습) 후 저장소 루트에서 다음 명령으로 USD를 생성합니다.

```powershell
cd RLForHexapod
uv run --locked --extra isaacsim python scripts/convert_urdf.py
```

변환 뒤 소수의 환경에서 관절 움직임과 접촉을 확인하세요. 로봇 형상이나 관측·행동이 달라졌다면
기존 정책의 호환성을 확인한 뒤 재학습하세요.
