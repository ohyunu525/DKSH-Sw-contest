# DKSH-Sw-contest

Unity 및 ROS2 기반 6족 Spiderbot 프로젝트입니다. 보행·계단 학습은 [RLForHexapod](RLForHexapod)의
Isaac Lab 3.0 환경과 RSL-RL 설정을 사용합니다.

## 6족 자율 탐색 빠른 시작

Unity `LidarScene`에는 기존 Frontier 탐지·군집·목표 선택을 보존한 독립 실행 경로가 구성되어
있습니다. 최소 8방향 A*, 이진 clearance, line-of-sight waypoint 압축,
`CharacterController` 추종 및 단일 exploration coordinator를 사용합니다. ROS 브리지와
locomotion adapter는 독립 실행 동작을 방해하지 않도록 씬에서 기본 비활성 상태입니다.

ROS2 전환 경로는 `ros2` 폴더에 분리되어 있습니다. Docker Desktop을 시작한 뒤 다음처럼
Unity TCP Endpoint만 먼저 올립니다.

```powershell
.\ros2\manage.ps1 Start
.\ros2\manage.ps1 Validate
```

Unity Play Mode에서 `/clock`, `/scan`, `/odom`, `/tf`를 게시할 때만 headless
SLAM/Nav2 launch를 실행합니다. ROS 측은 Jazzy, SLAM Toolbox, SmacPlanner2D,
MPPI Omni, velocity smoother, collision monitor와 검증된 Frontier 후보 v1.6.1을 사용합니다.
[ROS2/Nav2 실행 안내](ros2/README.md)에서 연결 설정과 실행 명령을 확인할 수 있습니다.

## CAD 설계와 개조

[HEXAPOD Onshape 조립 모델](https://cad.onshape.com/documents/f646c0fde8bd91cfa43cbae7/w/279609403375cff4404c6eae/e/76a6985095c8e8f19d67aec6)에서 로봇의 부품과 관절 구성을 확인할 수 있습니다.
저장소에는 URDF, STL 메시와 Unity 프리팹도 포함되어 있습니다. 사용할 파일과 좌표계,
수정한 모델을 시뮬레이션에 반영하는 방법은 [CAD 개조 안내](isaaclab_project/docs/cad_update_workflow.md)를 참고하세요.

## RLForHexapod 학습

학습 코드는 [RLForHexapod 원격 main의 dcbad6d](https://github.com/qorhvma/RLForHexapod/tree/dcbad6dc015cbe72d0a362a417b83a0c70713ecf)에서 가져왔습니다.
Python 3.12와 uv를 사용합니다. Isaac Lab 3.0, Isaac Sim 6.2 및 CUDA 13.0용 PyTorch 버전은
[프로젝트 설정](RLForHexapod/pyproject.toml)과 잠금 파일에 고정되어 있습니다.

```powershell
cd RLForHexapod
uv sync --locked --extra isaacsim
uv run --locked --extra isaacsim isaaclab list_envs
uv run --locked --extra isaacsim isaaclab train --rl_library rsl_rl --task Hexapod-Walk --num_envs 4 physics=isaacsim_physx
```

계단 teacher는 `Hexapod-Stair`로 학습합니다. Student 환경과 평가 옵션은
[작업 등록](RLForHexapod/src/RLforHexapod/tasks/walk/config/hexapod/__init__.py)과
[계단 평가 스크립트](RLForHexapod/scripts/eval_stairs.py)를 참고하세요.
`isaaclab_project`에는 기존 Unity 모델과 Raspberry Pi에서 사용하는 배포용 자산 및 실행 기록이 남아 있습니다.
기존 체크포인트를 이어서 학습하기 전에 새 환경의 로봇 모델과 관측·행동 계약을 확인하세요.
