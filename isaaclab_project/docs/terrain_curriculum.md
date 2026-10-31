# 평지 복습과 무작위 장애물 커리큘럼

`-Environment curriculum`의 기본 실행은 혼합 서보 CAD6의 `revised_v8`을 사용한다. 18개 절대 목표각,
140개 관측, tibia 말단 20 mm 지지, 비말단 접지·미끄럼·무릎 여유 비용과 서보 제한은 유지한다.
기존 `flat`, `stairs`, `rough`, `mixed` 등 고정 코스도 비교용으로 그대로 사용할 수 있다.

## 코스 구성

- **평지:** 높이 장애물이 없는 직선 구간. 양쪽 경계벽과 시작·목표 위치가 있는 복습 코스다.
- **요철:** 위치·간격·높이가 달라지는 세 개의 단차.
- **계단:** 시작 위치와 단 높이가 달라지는 연속 상승 세 단 및 목표 플랫폼.
- **좁은 통로:** 통로 폭과 천장 위치·높이가 달라지는 구간.

시작·목표의 좌우 위치도 변화시키되, 현재 로봇의 폭을 기준으로 초기 충돌을 피할 여유를 둔다.
모든 코스는 같은 크기의 충돌체 9개를 재사용하며, 에피소드 초기화 때 위치만 변경한다.
사용하지 않는 충돌체는 지면 아래에 두고 합성 LiDAR 조회에서 제외한다. 충돌체와 발 접지,
높이 관측, LiDAR는 같은 코스 좌표를 사용한다. 진동 바닥·낙하물은 기존 개별 preset에서
사용할 수 있으며 이번 무작위 커리큘럼의 구성에는 포함하지 않는다.

`-FlatFraction` 기본값은 0.25다. 환경이 두 개 이상이면 `ceil(환경 수 × 비율)`개를
평지 복습용으로 예약하되, 평지와 장애물이 각각 최소 한 환경씩 남게 한다. 따라서 4환경은
평지 1개와 장애물 3개다. 나머지 환경은 에피소드마다 요철·계단·통로를 순환하며 새 배치를 받는다.
비율은 **병렬 환경 수 기준**이며 완료 에피소드 수나 PPO 표본의 정확한 비율을 의미하지 않는다.
1환경에서는 평지와 장애물을 에피소드 순서로 번갈아 배정한다.

## 성공률에 따른 단계 조절

장애물 종류별로 독립적인 난이도와 성공 이력을 가진다. 기본 설정은 다음과 같다.

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `-Difficulty` | 0.5 | 시작 난이도. 첫 전이는 **0 지정** 권장 |
| `-CurriculumWindow` | 20 | 최근 완료 결과를 보관할 최대 개수 |
| `-CurriculumMinEpisodes` | 20 | 난이도 판단에 필요한 최소 완료 개수 |
| `-CurriculumPromoteThreshold` | 0.8 | 성공률이 이상이면 승급 |
| `-CurriculumDemoteThreshold` | 0.3 | 성공률이 이하이면 강등 |
| `-CurriculumStep` | 0.1 | 한 번의 단계 변화. 범위는 0~1 |

성공은 기존 v7의 목표 위치·높이·자세·지지발·비말단 접지 조건을 만족한 도착이다.
낙상·이탈·시간 초과는 실패다. 평지 성공은 장애물 난이도를 올리지 않으며, 초기 reset이나
완료되지 않은 에피소드의 수동 reset도 성공 이력에 넣지 않는다. 단계가 바뀌면 해당 이력을
비우고, 이전 난이도에서 시작한 다른 환경의 뒤늦은 결과는 새 단계 판단에서 제외한다.
중간 성공률에서는 최근 결과를 계속 갱신한다. `-FreezeCurriculum`은 학습 난이도를 고정하는
비교 실험용 옵션이다. 평가·재생의 난이도는 항상 고정한다.

TensorBoard의 `Curriculum/difficulty/<종류>`, `success_rate`, `window_episodes`,
`promotions`, `demotions`로 진행 상황을 확인할 수 있다. 말단 지지와 무릎 지면 여유 지표도
기존 `Metrics/` 로그에 남는다. 과제에 맞지 않는 자세 비용은 별도 실험으로 조정해야 하며,
이번 변경은 말단 지지 비용을 약하게 만들거나 무릎 주행을 허용하지 않는다.

## 학습 실행

Isaac Sim 실행은 `DEVELOPMENT_REQUIREMENTS.md`의 A 등급 NVIDIA 장비에서 4환경부터 시작한다.
평지 v7 정책에서 전이할 때는 체크포인트를 명시한다. 관측·행동·보상 계약이 같으므로 이 전이는
`AllowBehaviorTransfer`가 필요하지 않다. v2~v6 보상 프로필은 이 커리큘럼에서 직접 사용할 수 없다.

```powershell
.\run_cad6.ps1 -Mode check -Environment curriculum -Difficulty 0 -NumEnvs 4

.\run_cad6.ps1 -Mode train -Environment curriculum -Difficulty 0 -NumEnvs 4 -FlatFraction 0.25 -Seed 42 -CourseSeed 42 -MaxIterations 4000 -Checkpoint '<평지 revised_v8 model_N.pt>' -AllowProvisionalCad
```

체크포인트를 생략하면 새 가중치로 시작한다. 기본 단계를 변경하는 예시는 다음과 같다.

```powershell
.\run_cad6.ps1 -Mode train -Environment curriculum -Difficulty 0 -NumEnvs 4 -CurriculumWindow 40 -CurriculumMinEpisodes 40 -CurriculumPromoteThreshold 0.85 -CurriculumDemoteThreshold 0.25 -CurriculumStep 0.1 -AllowProvisionalCad
```

로그 폴더는 기존 평지/고정 장애물과 분리된 `*_curriculum/<run>/`이다. `run_metadata.json`에는
코스 계약, 생성 seed, 분할, 복습 비율 및 단계 설정을 기록한다. 각 모델에는 정확한 모델 SHA와
연결된 `model_N.pt.courses.json`을 함께 저장한다. 동일 설정에서 재개하면 난이도·성공 이력·
환경별 생성 순서를 복원하고 새로운 물리 에피소드로 시작한다. PhysX 상태나 진행 중 rollout을
복원하는 것은 아니다. 코스 설정이 달라지거나 sidecar가 없으면 새 커리큘럼으로 시작한 사실을
metadata에 명시한다. 모델·metadata·해당 sidecar를 함께 보관한다.

## 학습하지 않은 코스 평가

학습은 `train`, 평가·재생은 `holdout` 분할만 사용한다. 두 분할은 별도의 해시 seed 영역을
사용하므로 같은 `CourseSeed`라도 서로 다른 배치가 나온다. 코스는 master seed, 분할,
환경 번호, 해당 환경의 에피소드 순서, 종류 및 난이도로 재현한다. 다른 환경의 reset 순서와
Torch의 무작위 수 소비는 이 생성 순서를 바꾸지 않는다. 동일 순서를 다시 재현하려면 환경 수,
복습 비율, seed와 난이도도 동일하게 유지한다.

```powershell
.\run_cad6.ps1 -Mode evaluate -Environment curriculum -Difficulty 0.3 -NumEnvs 4 -Seed 73 -CourseSeed 42 -CourseSplit holdout -Steps 48000 -Checkpoint '<커리큘럼 revised_v8 model_N.pt>' -Output 'logs/heldout_d03.json'
```

난이도 0, 0.3, 0.6, 1에서 각각 같은 방식으로 평가하고, 최종 성능 보고에 사용할 seed는 학습
튜닝용 검증 seed와도 별도로 보관한다. 여러 seed로 반복 평가해야 한 번의 유리한 배치를
일반화로 오해하지 않는다. 위 분할은 새로운 배치 검증이며, 새로운 장애물 종류나 실물 하드웨어
전체에 대한 일반화를 보장하는 것은 아니다.

JSON에는 종료 전 코스 ID·seed·분할·종류·난이도와 결과를 연결한 에피소드 기록 및 종류별/
종류-난이도별 요약이 남는다. `heldout_course_pass`의 초기 판정 기준은 **네 종류 각각**:

- 서로 다른 완료 코스가 최소 2개.
- 완료 에피소드 성공률 80% 이상, 낙상 및 이탈 0.
- 지속 순진행 속도가 기존 최소 속도 기준 이상.
- 각 에피소드의 몸체·femur·hip 접촉, 비말단 지지 및 비말단 충돌 비율이 모두 0.
- v8의 지지/충돌 진단이 모든 평가 스텝에서 기록되어야 함.

검사 기준과 부족한 조건도 JSON에 기록한다. 완료하지 않은 코스는 성공으로 세지 않으며,
표본이나 접촉 진단이 모자라면 통과하지 않는다. 깨끗한 에피소드나 다른 환경으로 금지 접촉을
평균 내 가릴 수 없다. `locomotion_pass`도 진행 중 에피소드의 금지 접촉까지 검사하며,
`heldout_course_pass`가 새 코스의 종류별 검증 결과다. 이 수치는 초기 공학적 판정 기준이며
실기 검증으로 보정할 대상이다. 목표 속도 0.20 m/s는 달성 속도가 아니다.

이 기능의 CPU 테스트는 생성·재현·부분 reset·센서 좌표·단계 조절·평가 집계를 검증한다.
실제 PhysX 충돌과 학습 수렴, 새로운 코스 통과 성능은 GPU 장비에서 위 check/train/evaluate를
실행해 별도로 확인해야 한다.
