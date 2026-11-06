# DKSH-Sw-contest

Unity 및 ROS2 기반 6족 Spiderbot 프로젝트입니다. 보행·계단 학습은 [RLForHexapod](RLForHexapod)의
Isaac Lab 3.0 환경과 RSL-RL 설정을 사용합니다.

## 개발 컴퓨터 선택

이 저장소는 한 컴퓨터에 모든 도구를 동시에 실행하는 방식으로 운영하지 않습니다. RTX 3070·RAM
32GB 장비는 Isaac Sim/Isaac Lab 학습과 정책 검증을 담당하고, RAM 16GB·내장 GPU 장비는
Unity 및 headless ROS2/Nav2 통합을 담당합니다. 그보다 낮은 사양의 장비는 사양에 따라 Unity 단독,
정적 테스트 또는 문서·코드 리뷰만 수행합니다.

설치 전 반드시 [개발 컴퓨터 요구사항 및 작업 분담](DEVELOPMENT_REQUIREMENTS.md)에서 해당 장비의
등급, 금지되는 동시 실행 조합과 컴퓨터 간 결과 전달 규칙을 확인하세요.

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
현재 16 GB RAM·내장 Intel Arc PC에서는 Unity와 이 headless 스택만 함께 실행하고,
Isaac Lab 학습은 반드시 별도 세션에서 실행합니다. 상세 명령, 자원 상한 및 실측치는
[ROS2/Nav2 실행 안내](ros2/README.md)를 참고하세요.

Unity와 ROS2를 서로 다른 네트워크의 두 컴퓨터에서 연결하려면 Tailscale을 사용할 수 있습니다.
ROS2 컴퓨터에서 `./ros2/manage.ps1 Start` 후 관리자 PowerShell로
`./ros2/tailscale.ps1 Configure`를 실행하고, 출력된 `100.x.y.z` 주소를 Unity의
`Ros2UnityBridge` IP에 입력하세요. Docker의 ROS 포트는 로컬 전용으로 유지되며, 원격 접속은
tailnet 안에서만 허용됩니다. 자세한 조건과 해제 방법은 [ROS2/Nav2 실행 안내](ros2/README.md)를 참고하세요.

이 개발 PC 자체에 원격 접속하려면, 관리자 PowerShell에서
`./scripts/enable_tailscale_rdp.ps1`을 한 번 실행하세요. 이 스크립트는 Windows RDP를
활성화하되 Tailscale 주소 범위에서만 3389/TCP를 허용합니다. 개인 노트북에서는 출력된
`mstsc /v:100.x.y.z` 명령이나 MagicDNS 이름으로 접속합니다.

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
