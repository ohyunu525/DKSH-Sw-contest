# Spiderbot 8족 / 6족

- `Spiderbot8Leg/Spiderbot8Leg.prefab`: 8개 다리, 24개 회전 관절
- `Spiderbot6Leg/Spiderbot6Leg.prefab`: 6개 다리, 18개 회전 관절

원본 `completeLEG.fbx`의 조립된 다리 형상을 사용합니다. Unity 좌표는 Y-up,
단위는 미터입니다. 각 프리팹에 ArticulationBody, 관절별 메시, URP 재질,
단순 충돌 박스가 포함됩니다. 씬의 바닥 위에 배치해 사용합니다.

중앙 원판은 임시로 구성했고, 관절축·질량·관성은 초기 추정값입니다.
기존 Navigation Proxy/ML-Agents 컨트롤러와의 연결은 아직 하지 않았습니다.
Isaac Lab에서는 `-RobotModel cad8` / `-RobotModel cad6`로 각각 선택 가능합니다.

이 PC에서는 Unity Editor 라이선스가 활성화되지 않아 프리팹의 Editor 임포트와
Play 검증을 수행하지 못했습니다. 파일 구조·메시·관절 연결 검사와 C# 컴파일은
통과했으며, 동일한 모델의 Isaac Lab 물리 검사는 두 버전 모두 통과했습니다.

전체 설명 및 미리보기:
`isaaclab_project/assets/spiderbot_variants/README.md` (저장소 루트 기준)

재생성은 저장소 루트에서:

```powershell
uv run --python 3.12 --with-requirements scripts/spiderbot-model-requirements.txt scripts/build_spiderbot_variants.py --unity
```

이후 활성화된 Unity에서 `DKSH > Spiderbot > Build 6 and 8 Leg Models` 메뉴를
실행하면 네이티브 형식으로 재저장하고 참조를 확인합니다.
