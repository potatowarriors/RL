# MiMo-V2.6-RL Code (Harbor) 스터디 — agentic SWE teacher 후보 데이터 실사·Gym 연동 작업

_작성 2026-10-21. GPU 를 쓰지 않은 조사다. 조회 대상: HF `FineEnvs/MiMo-V2.6-RL-harbor-code` (sha `cf87dbe`, `mimo_harbor 1.3.0`, 2026-10-09) ·
HF `XiaomiMiMo/MiMo-V2.6-RL-oss` (main `e8e81e4`, 2026-10-20) · Docker Hub `xiaomimimo/mimo-v2.6-rl-oss` · GitHub `XiaomiMiMo/mimoagent` (`c3df846`) ·
NeMo-Gym 로컬 v0.6.0 `3045a793` 와 upstream main (2026-10-21 조회). 착수 때 §7 을 다시 확인한다._

## 0. 결론과 사용자 결정

**사용자 결정 (2026-10-21)**: 착수는 **MOPD1 뒤 agentic teacher 단계** (`RL_PLAN.md` 결정 18) 다. 지금은 조사·기록만 한다.
진행하면 §6 의 권고 1~5 를 그대로 따른다 (`RL_PLAN.md` 결정 40).

1. **Gym 은 이 데이터를 받을 수 있다. 막히는 곳은 컨테이너 실행 수단이다.** 태스크마다 전용 이미지가 있고 (2,698 개 모두 다르다),
   학습 노드 main1·sub1 에는 docker·apptainer·singularity·podman·enroot 가 없다.
2. **공개 Harbor 판은 낡았다.** Xiaomi 원본은 2026-10-16 에 이미지를 다시 올렸고 10-18 에 태스크 96 개를 뺐다. Harbor 판(10-09)은 옛 이미지를 digest 로 고정하고 그 96 개도 담고 있다.
3. **경로는 Harbor 판이 아니라 Xiaomi 최신 원본(2,602) + Gym `swe_agents` + 새 MiMo 채점 처리기다.**
   Gym 의 Harbor 경로에서 학습이 되는 에이전트는 Terminus-2(키 입력 JSON) 하나뿐이다. 함수 호출 형식이 아니고, 이미지 안에 python3·uvicorn 이 있어야 한다.
4. **인프라에 투자하기 전에 alpha 의 통과율부터 잰다.** 정답 패치가 없어 태스크 불량을 미리 걸러낼 수 없다.
   Ultra 블렌드에서 alpha 의 보상 > 0 비율은 code_gen 0.7% 였다 (`RL_DATA.md` §5 · `STATUS.md`).

## 1. 데이터셋 실체

| 항목 | 내용 |
|---|---|
| 규모 | Harbor 판 2,698 · Xiaomi 원본 현재 2,602 (`code.parquet`, 10-18 에 96 개 제외) |
| 과제 | 실제 리포의 이슈 문장(`instruction.md`)을 받아 `/testbed` 의 기준 커밋에서 수정한다 |
| 채점 | 숨긴 `test.patch` 가 건드리는 파일을 기준 커밋으로 되돌리고 → 패치 적용 → `test_command`(대개 `bash /testbed/mimo_test_command.sh`) 종료 코드 0 이면 1, 아니면 0. 되돌리기·적용 실패는 보상을 쓰지 않는다 (테스트베드 실패 = 채점 제외) |
| 이미지 | **태스크마다 다른 이미지** (`docker_image` 2,698 종). 40 개 무작위 표본(seed 0) 압축 크기: 최소 0.42 · 중앙값 3.11 · p90 4.49 · 최대 6.57 · 평균 2.47 GB, 공유 레이어 0. 전체 추정 압축 약 6.4~6.7 TB |
| 자원·시간 (`task.toml`) | 2 CPU · 8 GB · 에이전트 3,600 s · 채점 2,100 s · `network_mode = "public"` · 참조 스텝 상한 500 |
| 셋업 (healthcheck) | git 이력에서 기준 이후 커밋 제거 (실패 시 `.git` 숨김) · 빌드 잔여물·캐시 청소 · `git clean -fdx` (의존성 디렉토리 보존) · 정답 유출 차단 목록을 `/etc/hosts` 에 추가 (GitHub·GitLab·PyPI·npm·crates·Go proxy·HF 데이터셋 호스트 등) |
| 언어 (120 개 표본, seed 2) | Python 55 · Go 23 · JavaScript 20 · TypeScript 12 · Unknown 6 · PHP 2 · Java 1 · C++ 1 |
| 지시문 | 영어 위주, CJK 포함 92/2,698 · 원본 프롬프트 길이 중앙값 1,604 자 · p90 2,677 자 · 최대 58,653 자 |
| 정답 패치 | **없다.** 원본 `reward_model.ground_truth` 가 빈 문자열이고 `instance_json` 에 `patch` 키가 없다. 정답 패치로 채점기를 검증하는 게이트(`swe_agents` `verify_golden_patch`)를 쓸 수 없다 |
| 원본 행 형식 | parquet `prompt`(user 1 턴) · `extra_info.instance_json` = `cwd` · `docker_image` · `instance_id` · `problem_statement` · `test_command` · `test_patch` · `verifier_timeout_sec` |
| 라이선스 | Apache-2.0 (원본·변환). mimoagent 이식분은 MIT |
| 참조 성능 | Harbor parity: GLM-5.3 + OpenCode 1.18.32, 6 태스크 × 3 회 평균 0.556 ± 0.056 (Xiaomi 하네스와 같음) |

## 2. 원본 대비 낡은 부분 (2026-10-21 조회)

| 항목 | Harbor 1.3.0 (2026-10-09) | Xiaomi 현재 |
|---|---|---|
| 태스크 | 2,698 (원본 rev `639865fd3374`, 2026-09-26) | 2,602 — 10-18 커밋 `6b9046cc42d8` 이 96 개 제외 (예: `000014` · `000105` · `000121`). 남은 행 내용은 그대로라고 적혀 있다 |
| 이미지 | 옛 digest 고정 | 10-16 재푸시 (`format-code-task-000001` 태그 갱신 2026-10-16 14:45Z). 표본 20/20 이 고정 digest 와 다르다 |
| 정답 유출 방지 | mimoagent `467f0a1` 이식. 이력 제거 뒤 검증은 기준 시각 이후 `git log --all` 개수뿐 | 원본 README (10-19): 이미지에 전처리를 이미 적용했다. mimoagent `c3df846` (10-18): 셋업에서 이력을 다시 지우고 `원격 0 · 기준 밖 커밋 0 · odb 커밋 수 == 도달 가능 커밋 수` 를 검증한다. `.keep` pack · `logs` · `worktrees` 도 지운다 |

배경: Vals AI 는 공개 코드 환경의 약 2/3 에서 git 이력으로 정답을 읽을 수 있었고 MiMo 가 그것을 찾아 썼다고 보고했다.
Xiaomi 는 학습 때 확인된 해킹 궤적이 2% 미만이었다고 밝혔다. Harbor 1.3.0 의 `/etc/hosts` 차단 목록도 같은 외부 분석을 계기로 넣은 것이다.

## 3. 우리 쪽 현황 (2026-10-21)

| 구성 요소 | 상태 |
|---|---|
| Gym 서브모듈 | v0.6.0 `3045a793` (결정 10 고정). `harbor_agent` · `swe_agents` · `anyterminal_agent` · `mini_swe_agent` 를 모두 담고 있다 |
| Harbor 패키지 | 설치되어 있지 않다. `responses_api_agents/harbor_agent/requirements.txt` 가 git 커밋 `9dddd797` 을 고정한다 |
| 컨테이너 실행 수단 | main1 에 없다. Backend.AI 컨테이너 안이고 `/dev/fuse` 없음 · CapEff 0 · 비특권 user namespace 허용. 쓸 수 있는 곳은 gpu06 docker 호스트뿐 (64 코어 · RAM 441 GB · `/var/lib/docker` 4.1 TB · 공용, `RL_DATA.md` §5.9) |
| main1 자원 | 220 코어 · RAM 2,015 GB (조회 시 가용 1,647 GB) · NFS 여유 42 TB |
| 빈 rollout 처리 | 생성 토큰이 없는 Gym 결과가 오면 `nemo_rl/environments/nemo_gym.py:817-841` 이 `ValueError` 를 던져 스텝이 죽는다. Harbor 경로는 실패하면 바로 그런 결과(빈 `output`)를 돌려준다 |
| 실패 rollout 마스킹 | `mask_sample` 을 `swe_agents`(`app.py:3859`) · `anyterminal_agent`(`app.py:575`)는 설정하고, `harbor_agent` 는 설정하지 않는다 |
| alpha 레시피 규약과 충돌 | Harbor README 는 `truncate_history_thinking: false` 를 권하지만 alpha Gym 레시피는 `chat_template_kwargs: null` 이다 (`../CLAUDE.md`). 다중 턴 토큰 연속성을 alpha 템플릿으로 따로 확인해야 한다 |
| 컨텍스트 | alpha 상한 128K (`RL_PLAN.md` §4). 참조 설정(500 스텝 · 응답 65,536)은 그대로 못 쓴다. Ultra SWE teacher 는 `agent_max_turns` 200 · 196,608 · 32 × 16 · 에이전트 3,600 s · 메모리 64 GB (`examples/nemo_gym/nemotron-3-ultra/swe_teacher.yaml`) |
| 호스트 메모리 | 다중 턴 에이전트에는 `max_steps: 1` 보호가 듣지 않는다. 행 수 × 패딩 길이 × 416 B × CP8 복제 예산을 다시 잡는다 (`KNOWN_ISSUES.md` 2026-10-20 호스트 OOM) |

## 4. 실행 경로 비교

| 경로 | 장점 | 막히는 점 |
|---|---|---|
| A. `harbor_agent` + Singularity 환경 | Harbor 태스크 디렉토리를 거의 그대로 쓴다 | 학습 가능한 에이전트가 `Terminus2NemoGym` 뿐이다 — tmux 키 입력을 본문 JSON 으로 보내므로(`utils.py:273-296`) alpha SFT 의 함수 호출 형식과 다르다 · 이미지 안에 python3 + uvicorn 이 있어야 한다 (`singularity.py:294-302`, Go·JS 이미지 문제) · `mask_sample` 이 없다 · healthcheck 셋업을 `environment/files/setup.sh` 로 옮겨야 한다 · `singularity` 실행 파일 이름이 고정이다 (`singularity.py:228`) |
| **B. `swe_agents` + MiMo 처리기 (채택)** | NRL 의 Ultra·Super SWE 레시피가 쓰는 학습 경로 · OpenHands 또는 opencode 의 bash + edit 도구 · token ID · `mask_sample` · 에이전트 런타임을 호스트 설정 트리에서 마운트하므로 이미지 언어와 무관 · Harbor 형식 처리기 `DeepSWEDatasetProcessor` (`swe_agents/app.py:1228`)를 본뜰 수 있다 | 처리기를 새로 써야 한다 · Apptainer `.sif` 가 필요하다 · README 가 언급하는 `configs/swebench_deepswe.yaml` 은 v0.6.0 에 없다 |
| C. `anyterminal_agent` | Harbor `task.toml` + `tests/test.sh` + `reward.txt` 를 그대로 지원 · docker/apptainer · `mask_sample` | 안쪽 Gym 에이전트를 따로 골라야 한다 · `tb_eval_timeout` 기본 300 s (`app.py:288`) 를 2,100 s 로 늘려야 한다 · 학습 실적이 B 보다 적다 |

upstream main 에는 `harbor_agent_general` 도 있다. Harbor 기본 에이전트(OpenCode 등)를 ATIF 궤적 변환으로 붙이는 방식이다.
README 가 "token ID 를 돌려주지 않을 수 있어 학습은 지원하지 않는다 (작업 중)" 라고 적고 있어 지금은 후보가 아니다.

## 5. 필요한 작업 (경로 B)

**데이터**
1. Xiaomi 최신 `code.parquet` (2,602 행)의 `instance_json` 을 `swe_agents` 입력 행으로 바꾼다. 행 형식은 `input: []` · `metadata.instance_dict` · `agent_ref` 로, Ultra `swe.jsonl` 과 같다 (`SPEC_rl_dataset_inventory.md` Family G).
2. 이미지는 현재 태그(10-16 판)를 digest 로 고정해 기록한다.

**셋업 — 정답 유출 방지**
3. mimoagent `c3df846` 의 이력 제거·검증과 잔여물·캐시 청소, `git clean` 보존 목록을 처리기 셋업에 옮긴다.
4. 네트워크 차단(GitHub·PyPI·npm·HF 등)을 Apptainer 에서 구현하고 검증한다. `swe_agents` 는 `--no-mount home,tmp,bind-paths` 로 띄우므로 `/etc/hosts` 는 이미지 쪽 파일이 된다 — 차단 방식은 착수 때 실측으로 정한다.

**채점**
5. 테스트 파일 되돌리기 → `test.patch` 적용 → `test_command` (타임아웃 2,100 s) 순서로 채점한다. 원본 `test.sh` 와 mimoagent `opensource_code.py` 가 기준이다.
6. 되돌리기·적용 실패 같은 테스트베드 실패는 0 점이 아니라 `mask_sample` 로 처리한다.

**인프라**
7. 컨테이너 실행 수단을 확보한다 (§6 결정 4).
8. 이미지를 `.sif` 로 바꿔 둔다. 압축 기준 약 6.5 TB 로 NFS 여유 42 TB 에 들어간다. 변환 임시 공간을 따로 잡는다.

**NRL 본체**
9. 빈 rollout 이 스텝을 죽이지 않도록 마스킹으로 바꾼다 (`nemo_gym.py:817-841`). 본체 패치라 별도 커밋과 독립 검토 대상이다 (`.claude/rules/alpha.md`).

**레시피**
10. 128K 상한에 맞춰 `agent_max_turns` (Ultra 200 기준)와 에이전트 타임아웃 3,600 s 를 정한다. `vllm_cfg.turn_end_token_id: 3` 을 넣는다.
11. 호스트 메모리 예산을 다시 계산한다 (§3 마지막 행).

**게이트**
12. 무작업 에이전트로 전 태스크 0 점을 확인한다. 정답 패치가 없으므로 이것이 할 수 있는 최대 사전 검증이다.
13. 이력 유출을 검사한다 (`odb == 도달 가능` 커밋).
14. 다중 턴 logprob 정합을 확인한다 (GATES G3 형태).
15. 보상 해킹을 감시한다 (`git log` 조회, 네트워크 시도).

**처리량 민감도** (gpu06 에서 동시 32 개, 스텝당 rollout 256 = 32 × 8 로 가정)

| 평균 rollout 시간 (가정) | 10 분 | 20 분 | 40 분 |
|---|---|---|---|
| 스텝당 시간 | 1.3 h | 2.7 h | 5.3 h |

평균 rollout 시간은 실측값이 없다. gpu06 하나로는 학습 속도가 rollout 쪽에 묶인다.
동시 32 개는 태스크당 2 CPU 기준(64 코어)이다. 메모리 기준(441 GB / 8 GB)으로는 55 개다.

## 6. 결정 (사용자 2026-10-21 — 진행하면 권고대로)

| # | 항목 | 채택안 | 근거 |
|---|---|---|---|
| 1 | agentic teacher(SWE) 후보에 넣을지 | **넣는다.** 이미 예약된 SWE-Gym · SWE-rebench-V2 (`RL_DATA.md` §1.3) 와 함께 본다 | 그 둘은 Python 위주다. 이 데이터는 Go · JS · TS 가 절반에 가깝다 |
| 2 | 원본 판 | **Xiaomi 최신판** (2,602, 10-16 이미지) + mimoagent `c3df846` 셋업 이식. Harbor 1.3.0 은 쓰지 않는다 | §2 |
| 3 | 실행 경로 | **B — `swe_agents` + MiMo 처리기** | 학습 실적 · 함수 호출 형식 · token ID · 마스킹을 모두 갖춘 경로 (§4) |
| 4 | 컨테이너 실행 위치 | **(a) main1 CPU 에서 사용자 공간 Apptainer 가 FUSE 없이 도는지 GPU 없이 시험 → (b) 안 되면 gpu06 docker (동시 약 32, 캐시 4.1 TB 라 부분집합 운용) → (c) 외부 유료 sandbox(Daytona 등)는 마지막** | main1 은 220 코어 · RAM 2 TB 다. (a) 의 설치 위치(`$NRL_ROOT` 아래)가 `SETUP.md` 의 "시스템 변경 없이" 정책에 맞는지는 착수 때 다시 승인받는다 |
| 5 | 첫 단계 | **이득 상한 측정 먼저** — alpha 체크포인트로 약 100 태스크 × 4 회 통과율을 Harbor CLI + docker 로 gpu06 에서 평가만 한다 (모델 서빙 GPU 는 승인 대상). 통과율이 1% 근처면 난이도 필터나 다른 SWE 데이터를 먼저 본다 | 정답 패치가 없어 불량 태스크를 미리 거를 수 없다 · Ultra code_gen 보상 > 0 은 0.7% 였다 |

## 7. 착수 때 다시 확인할 것

- Xiaomi 원본과 FineEnvs 판의 갱신 여부 — 태스크 수, 이미지 digest, mimoagent 커밋. 이 문서의 수치는 2026-10-21 조회값이다.
- upstream Gym `harbor_agent_general` 의 학습 지원 여부. 지원되면 OpenCode 경로가 후보가 된다. 우리 Gym 은 v0.6.0 고정이므로 올리려면 별도 결정이 필요하다 (결정 10).
- alpha agentic SFT 의 도구 이름·형식이 OpenHands CodeAct (`execute_bash` · `str_replace_editor`) 또는 opencode (`bash` · `edit` 등) 와 얼마나 맞는지.
- MOPD1 에서 나온 시작 체크포인트와 그 시점의 128K 상한 결정 (`RL_DATA.md` §4 "SWE 196k vs 128k").

## 8. 출처

- 데이터: [FineEnvs/MiMo-V2.6-RL-harbor-code](https://huggingface.co/datasets/FineEnvs/MiMo-V2.6-RL-harbor-code) (README · `parity.md` · `data/tasks.jsonl` · 표본 태스크 3 개) · [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) (`code.parquet` · `image-mapping.jsonl` · 커밋 이력) · Docker Hub 레지스트리 API (manifest 크기 · 태그 digest)
- 하네스: [XiaomiMiMo/mimoagent](https://github.com/XiaomiMiMo/mimoagent) `c3df846` · [Vals AI — MiMo reward hacking](https://vals.ai/blogs/mimo-reward-hacking)
- Gym: 로컬 `3rdparty/Gym-workspace/Gym/responses_api_agents/{harbor_agent,swe_agents,anyterminal_agent}` · upstream [harbor_agent README](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/harbor_agent) · [harbor_agent_general README](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/harbor_agent_general)
