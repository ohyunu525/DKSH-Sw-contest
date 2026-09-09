# DKSH-Sw-contest

Unity와 NVIDIA Isaac Lab에서 사용할 수 있는 8족 spiderbot 강화학습 프로젝트입니다.

![Isaac Lab에서 렌더링한 DKSH spiderbot](isaaclab_project/docs/spiderbot_preview.png)

## Isaac Lab 빠른 시작

현재 구성은 Isaac Sim 4.5.0과 호환되는 Isaac Lab 2.1.0/Python 3.10을 사용합니다. NVIDIA GPU와
Isaac Sim 4.5 standalone 설치가 필요하며, 기본 설치 위치가 다르면 경로를 지정합니다.

```powershell
git submodule update --init --recursive
.\setup_isaaclab.ps1
.\run_isaaclab.ps1 -Mode smoke
```

```powershell
.\setup_isaaclab.ps1 -IsaacSimPath "D:\path\to\isaac-sim-4.5"
```

성공하면 `DKSH_ISAACLAB_SMOKE_PASS`가 출력됩니다. 실제 PPO 학습은 다음처럼 시작합니다.

```powershell
.\run_isaaclab.ps1 -Mode train -NumEnvs 32 -MaxIterations 1000
```

학습 결과는 `logs/rsl_rl/dksh_spider_navigation` 아래에 저장됩니다.

가장 최근 체크포인트를 창 없이 유한 시간 검증하려면 다음 명령을 사용합니다.

```powershell
.\run_isaaclab.ps1 -Mode evaluate -NumEnvs 4 -Steps 500
```

특정 체크포인트를 지정할 때는 `-Checkpoint`를 함께 전달합니다. 화면에서 목표 지점과 동작을
재생하려면 다음 명령을 사용합니다(체크포인트 생략 시 가장 최근 모델 사용).

```powershell
.\run_isaaclab.ps1 -Mode play -NumEnvs 1
.\run_isaaclab.ps1 -Mode play -NumEnvs 1 -Checkpoint "C:\path\to\model.pt"
```

로그가 없는 새 체크아웃에서는 함께 제공되는 `balance_baseline.pt`를 자동 사용합니다. 이 모델은
500회 PPO 학습으로 20초 자세 유지를 검증한 시작점이며, 목표 보행을 완성한 모델은 아닙니다.

등록된 환경 ID는 `Isaac-DKSH-Spider-Navigation-Direct-v0`입니다. 로봇은 실제 설계값을 반영한
8개 다리, 24개 MG996R 서보 articulation이며, 링크 길이(86.17/100/120 mm), 서보 토크
(0.980665 N·m), 질량과 관절 속도가 Isaac Lab 물리에 반영되어 있습니다.

Unity 프로젝트는 `DKSH SW Project`에 유지됩니다. Unity의 FBX는 외형 참고용이고, Isaac Lab은
안정적인 학습을 위해 동일 치수의 primitive URDF에서 변환한 로컬 USD를 사용합니다. URDF 생성
소스도 함께 보관되므로 치수와 질량을 바꾼 뒤 USD를 다시 만들 수 있습니다.

```powershell
.\rebuild_spiderbot_asset.ps1
```

Isaac Sim 4.5는 변환 파일을 정상 기록한 뒤 종료 코드 1을 반환할 수 있습니다. 재생성 스크립트는
필수 USD 4개가 실제로 생성됐는지 검사하며, 이 경우 경고를 표시하고 성공으로 처리합니다.

## 검증 범위

- 물리 주기 200 Hz, 정책 제어 주기 50 Hz
- 관측 84차원, 연속 행동 24차원
- 무작위 방향/거리 목표, 목표 도달·낙상·영역 이탈·시간 제한 종료
- PPO 보상 항목과 성공/낙상/시간초과를 TensorBoard 로그에 기록
- 초록색 원판으로 각 병렬 환경의 목표 반경 표시

URDF 구조만 빠르게 확인하는 테스트는 Isaac Sim을 시작하지 않아도 실행할 수 있습니다.

```powershell
python -m unittest discover -s .\isaaclab_project\tests -v
```

이 저장소 구성은 RTX 3070에서 32개 병렬 환경 × 500 스텝 스모크 테스트, PPO 500회
(384,000 samples), 체크포인트 32개 환경 × 1,000 스텝 평가를 통과했습니다. 포함된 기준 모델의
평가 결과는 낙상 0회, 시간 제한 종료 32회였습니다. 재생 경로에서는 MP4 렌더링과 JIT/ONNX
정책 내보내기도 확인했습니다.
