# CAD6 femur 지면 접촉 벌점 수정안

**현재 CAD6 코드에서는 `femur` 전용 접촉 센서를 추가하고, 그 센서가 지면·계단 접촉을 감지하면 보상에서 벌점을 빼면 됩니다.** 기존 `tibia` 접촉 센서는 그대로 유지합니다.

아래는 **normal·sprint·velocity·계단 보행에 적용하는 수정안**입니다. 아직 실제 파일에 적용하거나 Isaac Sim에서 실행 검증한 코드는 아닙니다.

### 1. `mg90s_env_cfg.py`: 벌점 설정 추가

[`MG90SWalkEnvCfg` 클래스](https://github.com/ohyunu525/DKSH-Sw-contest/blob/main/isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/mg90s_env_cfg.py#L123)에 추가합니다.

```python
# 초기 실험값: 실제 접촉력·보상 로그를 보고 조정
femur_contact_threshold_n = 0.5
femur_contact_penalty_weight = 1.0
```

이후 계산에서 `step_dt`를 곱하므로, 가중치 `1.0`은 **femur 한 개가 1초 동안 계속 닿으면 누적 약 -1점**이라는 뜻입니다.

### 2. `mg90s_env.py`: femur 전용 접촉 센서 생성

먼저 import를 변경합니다.

```python
# 변경 전
from isaaclab.sensors import ContactSensor

# 변경 후
from isaaclab.sensors import ContactSensor, ContactSensorCfg
```

그리고 [`_setup_scene()`의 기존 센서 등록 부분](https://github.com/ohyunu525/DKSH-Sw-contest/blob/main/isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/mg90s_env.py#L71) **바로 아래**에 추가합니다.

```python
# 기존 tibia 센서 코드는 그대로 유지
if self.cfg.foot_contact_sensor is not None:
    self._foot_contact = ContactSensor(self.cfg.foot_contact_sensor)
    self.scene.sensors["foot_contact"] = self._foot_contact

# 아래부터 새로 추가
self._femur_contacts = []

# 평지 + 계단 지형만 접촉 대상으로 지정
terrain_paths = ["/World/ground", *self._task_terrain_paths()]

for leg in range(self.cfg.action_space // 3):
    sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/leg_{leg}_femur",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=terrain_paths,
        )
    )

    self.scene.sensors[f"femur_contact_{leg}"] = sensor
    self._femur_contacts.append(sensor)
```

**각 femur에 센서를 따로 두고 지형과의 접촉력만 읽는 구조**입니다. CAD6는 이미 `robot.spawn.activate_contact_sensors = True`가 설정되어 있습니다. 실제 USD의 강체 이름이 `leg_0_femur` 등의 경로와 일치하는지는 실행 시 확인해야 합니다.

### 3. 같은 `mg90s_env.py`: 공통 벌점 함수 추가

`MG90SWalkEnv` 클래스 안에 다음 함수를 추가합니다. `_get_rewards()`와 같은 들여쓰기 수준입니다.

```python
def _femur_contact_penalty(self):
    touching_per_leg = []

    for sensor in self._femur_contacts:
        # 지형 필터를 적용한 접촉력:
        # [환경 수, 감지 강체 수, 접촉 상대 수, xyz]
        forces = sensor.data.force_matrix_w

        if forces is None:
            raise RuntimeError(
                "Femur 접촉 센서의 지형 필터가 초기화되지 않았습니다."
            )

        force_magnitude = torch.linalg.vector_norm(forces, dim=-1)

        # 지면 또는 계단에 기준 이상의 힘으로 접촉했는가?
        touching = (
            force_magnitude > self.cfg.femur_contact_threshold_n
        ).any(dim=-1).any(dim=-1)

        touching_per_leg.append(touching)

    # [환경 수, 다리 수]
    touching = torch.stack(touching_per_leg, dim=-1)

    # 환경마다 접촉 중인 femur 개수
    contact_count = touching.float().sum(dim=-1)

    self.extras["log"]["Metrics/femur_contact_count"] = (
        contact_count.mean()
    )
    self.extras["log"]["Metrics/femur_contact_fraction"] = (
        touching.any(dim=-1).float().mean()
    )

    # 양수 비용을 반환하고, 보상 함수에서 차감
    return (
        self.cfg.femur_contact_penalty_weight
        * contact_count
        * self.step_dt
    )
```

판정 흐름은 다음과 같습니다.

```text
femur와 지형 사이의 접촉력
        ↓
0.5N 초과 여부
        ↓
접촉 중인 femur 개수
        ↓
개수 × 벌점 가중치 × 경과 시간
```

### 4. 보상 함수 세 곳에서 차감

**① 기본·normal·sprint: `mg90s_env.py`**

[`MG90SWalkEnv._get_rewards()`](https://github.com/ohyunu525/DKSH-Sw-contest/blob/main/isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/mg90s_env.py#L131)의 기존 계산 뒤에 추가합니다.

```python
# 기존 코드
reward = reward * self.step_dt - 5 * (
    self._fallen | self._escaped
).float()

# 추가
reward -= self._femur_contact_penalty()
```

**② 기본 실행 프로필인 velocity: `mg90s_cad6.py`**

[`MG90SCad6VelocityEnv._get_rewards()`](https://github.com/ohyunu525/DKSH-Sw-contest/blob/main/isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/mg90s_cad6.py#L175)에도 동일하게 추가합니다.

```python
reward = reward * self.step_dt - 5 * (
    self._fallen | self._escaped
).float()

# 추가
reward -= self._femur_contact_penalty()
```

**③ 계단: `mg90s_stairs.py`**

[계단의 `_get_rewards()`](https://github.com/ohyunu525/DKSH-Sw-contest/blob/main/isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/mg90s_stairs.py#L166)에서 성공·실패 보상 뒤에 추가합니다.

```python
reward += 20 * self._success.float() - 5 * (
    self._fallen | self._escaped
).float()

# 추가
reward -= self._femur_contact_penalty()
```

**`MG90SCad6Env._get_rewards()`에는 추가하지 마세요.** 부모 함수를 호출하므로 normal·sprint의 벌점이 중복됩니다. 위 비용에는 이미 `step_dt`가 포함되어 있습니다.

### 5. 학습 전에 이 세 가지를 확인하세요

| 상태 | 기대 결과 |
|---|---|
| tibia만 지지하고 femur는 떠 있음 | `femur_contact_count = 0`, 추가 벌점 없음 |
| femur 한 개를 지면에 접촉 | 접촉 수 증가, 해당 환경의 보상 감소 |
| 같은 자세에서 벌점 가중치만 0으로 변경 | 접촉 수는 같고 추가 벌점만 사라짐 |

**이 수정은 “femur로 지지하지 말라”는 목표를 추가합니다. “tibia의 원하는 발바닥 부분으로 반드시 걸어라”까지 보장하지는 않습니다.** 또한 현재 학습 보정 범위가 약 ±2도이므로, 기준 자세에서 femur를 들어 올리는 데 그보다 큰 변화가 필요하면 기준 자세·보행도 함께 수정해야 합니다.

## 적용 범위와 학습 전 검증·주의사항

**femur가 바닥에 닿을 때 벌점이 생기고, tibia가 닿는 것은 이 벌점에서 제외**됩니다. 기존 이동·자세 보상과 함께 학습하면 femur 접촉을 줄이면서 걷도록 유도할 수 있습니다.

다만 **‘femur 접촉 금지’만으로 ‘tibia로 안정적으로 걷기’가 보장되지는 않습니다.** 학습 전에 다음을 확인해야 합니다.

1. **접촉 판정:** femur를 바닥에 대면 벌점이 생기고, tibia만 닿으면 벌점이 없어야 합니다.
2. **기본 자세:** tibia로 지지하면서 femur를 띄우는 자세가 물리적으로 가능해야 합니다.
3. **행동 범위:** 현재 기준 보행에 대한 보정이 약 **±2도**라서, 필요한 자세 변화가 이보다 크면 기준 자세·보행 또는 보정 범위도 수정해야 합니다.

**우선 두 번째 수정안인 이 문서의 femur 전용 벌점만 적용해 접촉 검출부터 검증하세요.** 이후 hip으로 지지하는 문제가 나타나면 첫 번째 수정안처럼 금지 부위를 확장하면 됩니다. 두 코드를 동시에 넣으면 femur 접촉에 벌점이 중복될 수 있습니다.

앞서 제공한 코드는 **Isaac Sim에서 실행 검증하지 않은 수정 예시**이므로, 바로 장시간 학습하기 전에 위 접촉 시험을 먼저 해야 합니다.