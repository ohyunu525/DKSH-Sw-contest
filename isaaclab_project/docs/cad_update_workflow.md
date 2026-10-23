# Onshape CAD 갱신 및 학습 전환 절차

최신 설계의 단일 출처는 [HEXAPOD Onshape 최종 조립](https://cad.onshape.com/documents/f646c0fde8bd91cfa43cbae7/w/279609403375cff4404c6eae/e/76a6985095c8e8f19d67aec6)이다.
문서는 현재 링크 공유 읽기 전용이며 설계가 진행 중이다. 2026-09-30 관찰 기준으로
최종 조립은 body 1개, coxa/femur/tibia 각 6개로 총 19개 인스턴스이며,
`dof_L1_*`부터 `dof_L6_*`까지 이름이 붙은 Revolute Mate 18개가 있다.
다만 Mate Limits, 정확한 축 중심, 최종 질량 명세는 아직 확정되지 않았으므로 현재
상태에서 자동 URDF 변환이나 실물 기준 장시간 학습을 시작하지 않는다.

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

다음 명령으로 18개 Mate 이름, 영점, 축 방향 계약을 검사한다.

```powershell
python isaaclab_project/scripts/onshape_joint_frames.py
```

## 2026-09-28 Onshape URDF 내보내기

`assets/spiderbot_variants/onshape_20260928/`에는 제공받은 Onshape-to-Robot
내보내기 원본 URDF/STL과 Isaac Lab 변환 USD가 있다. 원본은 18개 Revolute joint,
25개 링크(고정 발 프레임 6개 포함), 총 질량 1.7923392 kg를 포함한다. 고정 프레임을
병합한 USD는 19강체다. 원본 URDF의 `±pi`, effort/velocity `10`은 exporter 기본값이므로
학습용 `robot.training.urdf`에서 프로젝트 관절 이름, `±90도` 한계, MG996R/RC920DMG의
보수적 한계로 정규화했다.

원본 coxa와 tibia Mate 축 부호는 프로젝트 규약과 반대였으므로 정규화 단계에서만
반전했다. CAD 변환 원점·링크 질량·관성·메시는 그대로 보존된다. 재생성은 다음과 같다.

```powershell
python isaaclab_project/scripts/normalize_onshape_urdf.py <raw-urdf> <training-urdf>
.\IsaacLab\isaaclab.bat -p .\IsaacLab\scripts\tools\convert_urdf.py --headless <training-urdf> <output-usd> --merge-joints --joint-stiffness 3.0 --joint-damping 0.06
```

이 내보내기로 관절 중심·축과 링크 질량/관성은 학습용으로 확보됐다. 그러나 모터가
질량값에 포함됐는지, 실제 간섭 한계, 최종 서보 배치, L2 장착 위치는 CAD 내보내기에
명시되지 않았으므로 자산 상태는 여전히 `current_onshape_provisional`이다.

## CAD 담당자가 확정할 항목

1. Onshape에서 조립 리비전을 `released_for_simulation` 상태로 동결한다.
2. 전체 조립을 부품 병합 없이 STEP으로 내보낸다.
3. body-coxa, coxa-femur, femur-tibia의 축·중심·영점 자세·허용 각도를 기록한다.
4. 몸체와 각 링크의 질량·무게중심을 내보내고, 값에 모터가 포함됐는지 표시한다.
5. 다리당 MG996R 1개와 RC920DMG 2개, 전체 MG996R 6개와 RC920DMG 12개 배치를 CAD에서 확인한다.
   RC920DMG의 실제 공급 전압에서 속도·정지 토크·전류를 데이터시트 또는 벤치 측정으로 남긴다.
6. L2 장착 위치·방향과 장착물을 포함한 질량을 확정한다.

## 저장소 반영 순서

1. STEP, 관절 명세, 질량 명세를 별도 CAD 입력 디렉터리에 넣고 SHA-256을 계산한다.
2. `assets/spiderbot_variants/cad_source.json`의 `release_evidence`와 lifecycle을 갱신한다.
3. 기존 `completeLEG.fbx` 추정 모델을 덮어쓰지 말고 새 리비전 자산을 생성한다.
4. URDF/USD에서 18관절의 축·영점·한계, 링크 질량 합계, 관성 양의 정부호를 검증한다.
5. 정적 자세와 무동작 중력 테스트, 4→8→16 환경 사전 검사를 수행한다.
6. 새 CAD/서보/라이다 manifest 해시로 처음부터 정책을 학습하고 별도 이름으로 보관한다.

준비 상태는 다음 명령으로 확인한다.

```powershell
python isaaclab_project/scripts/cad_readiness.py --json
```

준비되지 않은 CAD로 장시간 학습은 실패한다. 기구 확정 전 비교 실험만 필요하면
`run_cad6.ps1`에 `-AllowProvisionalCad`를 명시할 수 있으며, 이 사실은 학습
메타데이터에 남는다. 기존 정책은 최신 설계 검증용이 아니라 소프트웨어 파이프라인과
혼합 서보 가정의 예비 결과로만 유지한다.

현재 MG996R hip 6개 + RC920DMG 12개 구성의 실행과 관측·체크포인트 계약은
[절대 관절각 학습 안내](joint_angle_training.md)를 따른다.

```powershell
.\run_cad6.ps1 -Mode check -NumEnvs 4
.\run_cad6.ps1 -Mode train -Environment flat -NumEnvs 4 -MaxIterations 400 -AllowProvisionalCad
```
