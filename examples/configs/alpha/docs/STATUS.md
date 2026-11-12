# alpha RL 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push** 한다. 상태는 여기에만 쓴다 — auto-memory 에 쓰지 않는다
(메모리는 노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.
게이트 수치는 [`GATES.md`](GATES.md) 에 쓰고 여기에는 판정과 링크만 쓴다. Pai 쪽(SFT·벤치) 상태는 Pai `examples/alpha/docs/STATUS.md`.

_마지막 갱신: 2026-11-12 14:20 (SWE 피벗 teacher 파일럿 CPU 준비 완료 · GPU 승인 대기 — 결정 61; 이전: **✅ MOPD1 완료 — 사용자 판정 "1차 MOPD 는 성공적" (11-12)**: 학생 282 · 512 × 120 · `mopd1/s3` rc 0, 학생 전 벤치 LCB 23.06 · MMLU-Pro 63.4 · GPQA-D 46.0 · IFEval 85.8 · IFBench 53.1 · SWE-V 12.0 · TB-2.1 2.9; 1차 teacher RL · MOPD1 의 지난 행은 [`archive/STATUS_2026-10-08_11-12.md`](archive/STATUS_2026-10-08_11-12.md) 로 정리. 진행 중: agentic teacher 단계 (결정 59, 다른 세션) · judge Claude 전환)_

## 진행 중

| 트랙 | 상태 | 다음 할 일 | 정본 |
|---|---|---|---|
| **SWE 피벗 teacher 파일럿 (2026-11-12~, `RL_PLAN.md` 결정 61)** | CPU 준비 완료 11-12: 블렌드 `teacher_blends/teacher_swe_pivot_v1.jsonl` 3,850 (+ 검증 128, 인스턴스 겹침 0) · D1 OK · 레시피 `teacher_swe_pivot_alpha.yaml` 검사기 ERROR 0 · 프롬프트 최대 98,568 토큰 (131,072 초과 0). 시작점 `mopd1/hf/hf_step_00120` 은 RL 시작점 게이트 미실행 — 스크립트 `$NRL_ROOT/gates/mopd1_step120/run_gates.sh` 작성만 함 | GPU 승인 대기: 게이트 M1 · M2 · M4 → R1 → 본 런 `teacher_swe_pivot/p1` (첫 3 step 으로 판정 뒤 그대로 학습) → SWE-V 짝 비교 | `RL_DATA.md` §6 · `FRONTIER.md` |
| **RL 준비 (main1·sub1, 2026-10-06~, 사용자 지시: RL 관련 작업은 전부 이 트랙)** | **환경**: Pai 컨테이너 안에서 시스템 무변경으로 구동한다(원장 #15~#27). 영속 설치는 NFS `$NRL_ROOT`(셋업 `project_s/setup_nemo_rl_env_noninvasive.sh`). 사용자 결정 10-06: 원본 NFS·venv 로컬. 버전: v0.7.0 미채택, r0.8.0 정식 릴리스 시 재검토. **검증 완료**: 모델·엔진 게이트 M1~M5 · R1~R4 · E4, RLVR 경로 게이트 G0~G8 · Z1 (수치는 `GATES.md`). 학습 엔진 = NeMo-RL mcore + Pai 기능 포팅 (`RL_PLAN.md` 결정 11). **실행 기반**: 실행기 `runs/launch.sh` — 산출물은 리포 `results/alpha/<campaign>/`, wandb 는 RL 전용 프로젝트 `alpha-rl`. 레시피가 `student_rlvr1_alpha.yaml` 로 고정돼 있어 teacher 레시피에 쓰려면 `--config` 를 받게 고친다. 레시피 기본값에 E1 레버(chunked prefill 16K · `max_trajectory_age_steps` 2)가 들어갔다. 기본 꺼짐으로 병합한 기능(advantage 0 제외 · 요청 우선순위 · 요청별 계측)과 켜는 조건은 `RLVR_READINESS.md` §5.6. 첫 RLVR 런·v2·속도 최적화(2026-10-07~08)는 아래 "완료" 절 | ⏸ R5 recompute 실측 연기 (128K full 33.4K tok/s 기준선만, selective OOM) · 셋업 재구성 승인 대기 · PivotRL 위치 재검토 (`RL_PLAN.md` 결정 14 — 결정 18 로 2차 agentic teacher 단계의 방법 후보) | `RLVR_READINESS.md` §5·§6 · `GATES.md` |

## 선행 의존 (Pai 스택)

| 항목 | 상태 (Pai STATUS 기준) | RL 쪽 영향 |
|---|---|---|
| agentic SFT 본 런 → 최종 ckpt iter2862 · 벤치 체인 | Pai `STATUS.md` "agentic 최종 iter2862 벤치" 행 | 완주 뒤 짧은 decay 여부 결정 → agentic base 확정 → PivotRL 시작점. 새 ckpt 는 `GATES.md` 재실행 조건(M1~M3 → R1)을 따른다 |

## 열린 사용자 결정

| 결정 | 선택지 · 권고 | 정본 |
|---|---|---|
| **MOPD1 후속 — General 추가 학습 여부 (2026-11-12)** | General teacher 와의 차이가 step 120 까지 선형 감소 중 (−0.274 → −0.120), IFBench 회복 77% (Ultra 보고 71.7%), 상한 loose +6.4. ① 현재 학생에서 이어 학습 (1 에폭 더 · General 프롬프트 추가) ② General 비중 상향 뒤 재학습 ③ 이대로 확정 — ①②는 에폭 · 블렌드 프론티어 조사 뒤 보고, MOPD2 (결정 59) 설계와 함께 정할 수 있다 | `study/benchmark_results.md` §0.5 · `RL_PLAN.md` 결정 60 |
| **agentic teacher 단계 (2026-11-11~, `RL_PLAN.md` 결정 59)** | 결정됨: 시작점 MOPD1 학생 step 120 · 전문 teacher 다섯 (terminal-use · conversational tool-use · SWE · Search · Safety) · MOPD2 의 STEM · Code·Math 슬롯은 현재 계획 학생 자신. **미정**: teacher 별 데이터 · 방법(피벗 단일 스텝 vs E2E) · 규모 · 순서, SWE 컨테이너 실행 수단 (**Apptainer 시험 실패 11-11** — main1 · sub1 세션 seccomp 가 user namespace 를 막음, root 로도 EPERM; 내장 proot 직접 실행은 됨, `study/mimo_code_harbor.md` §6.1; 남는 길 gpu06 docker · 인프라에 seccomp 완화 요청 · proot 변환 층 · 외부 sandbox — 결정 대기), 128K 상한, Safety 보상 (Safety-v1 은 선호 데이터라 RL 프롬프트 뱅크가 아님), Gemini 크레딧 (사용자가 방법 찾는 중) → **판정기를 Claude 로 교체 (사용자 결정 11-11)**: Pai `.env` 에 `ANTHROPIC_API_KEY` 추가, `eval_sft/runners/gemini_judge.py` 에 Claude 경로 작성 (미커밋 · 미검토). SimpleQA 저장 표본 300 재판정에서 Gemini 와 일치율 Opus 5.5 99.0% · Sonnet 5.5 96.7% (effort low). **막힘**: 약 760 호출 뒤 키가 사용 한도에 걸림 (`HTTP 400 … regain access on 2026-12-01`) — 한도 상향 전에는 T3 · τ³ 불가. 남은 일: 판정 파서 엄격화 (장문 응답에서 단어 CORRECT 를 등급으로 오독) · `run_tau.sh` 판정기 전환 · 적대적 검토 · 기준선 재측정 (저장 표본이 500 자로 잘려 재생성 필요). **준비 상태 실사 완료 11-11 (`RL_DATA.md` §6)**: 바로 블렌드 가능 = SWE 피벗 50,661 · conversational tool-use 피벗 96,968 (judge 불필요) / 데이터 0행 = terminal · Search / Safety 새 데이터는 주입 1,272 행뿐 / 판정 벤치 없음 = Search · Safety. **프론티어 조사 완료 11-12** (`FRONTIER.md` "agentic 전문가(teacher) RL"): Ultra 는 terminal · tool-use = PivotRL, SWE = PivotRL → E2E, search = SFT, safety = MOPD2 에서만; SWE 외 teacher 수치는 공개 안 됨; PivotRL 은 SWE 만 SFT 보다 낮음. 학습 계획 초안 보고 11-12 — 사용자 결정 대기 (첫 teacher · 컨테이너 수단 · Search · Safety 범위 · 프로파일링 GPU 승인) | `RL_PLAN.md` 결정 59 · `study/mimo_code_harbor.md` · `study/pivotrl_study.md` |
| **step 당 프롬프트 수 2단계 — 처음 100 step 64, 나머지 256 (사용자 2026-10-13 20:15 검토 중, Code·Math 레시피 때 결정)** | 런 중 변경 옵션이 없어 체크포인트 경계에서 멈추고 재개하는 2단계 런 (재기동 35~40분 + 재개 오버라이드 2개). 판단 근거: STEM `main2r3`(64) 대 `main2r6`(256) 의 보상 잔차·기울기 비교. 같은 자리에서 "GBS 를 쪼개 rollout 당 여러 번 업데이트" 와 lr 도 함께 결정 | `RL_PLAN.md` §2 검토 예정 · 결정 29 |
| **코드 능력 끌어올리기 — 1차 최대 과제 (사용자 2026-10-08, `RL_PLAN.md` 결정 20)** | 24K 에서 alpha 는 코드 답을 거의 끝내지 못한다 (32회 중 1회, 정답 0). 미루지 않고 48K 로 다시 재고 커리큘럼으로 올린다. Code·Math teacher 시작(세 번째) 전까지 방법을 찾는다 | `RL_DATA.md` §5.5 |
| **GBS·lr** (2026-10-08 지적, teacher 레시피 작성 때 결정) — **lr 결정됨: 도구 사용 teacher 3e-6 (`RL_PLAN.md` 결정 22) · 나머지 1차 teacher 5e-6 (사용자 결정 2026-10-09, 결정 23). 부모 `student_rlvr1_alpha.yaml` 의 1e-6 은 student 용으로 남겼다 (미정). GBS 는 아직 열려 있다** (근거: lr 1e-6 은 bf16 반영률 약 12%, 3e-6 은 step 50 에서 0.33 확인 · 짝 비교 보상 +0.009 ± 0.003, `KNOWN_ISSUES.md` 2026-10-08) | alpha 골격은 64 × 16 = GBS 1,024 · Muon lr 1e-6 · warmup 10 이다 (`student_rlvr1_alpha.yaml`, 64 는 `# 결정 대기` 임시값). 1차 teacher 규모안도 64 × 16 을 쓴다 (`RL_DATA.md` §5). Ultra 참조 (Adam, warmup 10): RLVR1·2 512 × 16 · lr 4e-6 / ifbench·rlhf teacher 128 × 16 · 2.5e-6 / reasoning teacher 128 × 16 · 3e-6 / swe teacher 32 × 16 · 3e-6 / MOPD 1024 × 1 · 2e-6. 배치 크기와 처리량: 정상 상태의 생성 노드는 최대의 약 1/3 로 돌고 64K 응답의 디코딩이 스텝을 정한다 (`RLVR_READINESS.md` §5.4). 배치를 바꿨을 때 시간당 샘플 수가 어떻게 변하는지는 실측이 없다 (앞서 쓴 "최대 처리량의 91%" 는 틀린 판단이라 §5.4 에서 정정했다) | 선택지(64·128·256)와 lr 조정안을 teacher 레시피와 함께 보고 → 사용자 결정 |
| Megatron-Bridge 포인터를 fork(`bcc4e415`)로 고정할지 | 현재 superproject 기록은 upstream `0c565c9a` — 새 클론은 AlphaBridge 가 없다 | `.claude/rules/alpha-submodules.md` |
| 셋업 재구성 제안 | 로컬 venv 스냅샷 · 인터프리터 로컬 · 바이트코드 사전 컴파일 · JIT 캐시 로컬 (2026-10-06 제안, 승인 대기) | `SETUP.md` §1 |
| RL 학습 워커에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 기본값으로 둘지 (D6) — **첫 런은 권고값(켬)으로 간다 (사용자 2026-10-07)** | 권고: 정책 워커에만 켠다. Pai SFT 는 항상 켰고 "refit 5배" 근거는 측정 없는 FP8 docstring 이다. G5 에서 refit 시간을 확인한다. 끄면 1노드 상한은 rank 당 12K 토큰(96K/CP8)이다 | `KNOWN_ISSUES.md` 2026-10-07 메모리 항목 |
| MOPD 재현 범위 · 장문 컨텍스트 캡 | 교사 슬롯은 결정 18 로 정함 (MOPD1 = teacher 3종) · SWE 교사·MOPD 192k → 128k 캡 또는 SWE 슬롯 축소 · NeMo RL/Gym 스택 포팅 vs 자체 구현(verl/ChatLearn 백엔드 검토) | `RL_PLAN.md` §4 |
| ARC-AGI-v1 사용 | 라이선스 `pending-legal-review` — 블렌드 내 nvarc 행(각 ~2%) 사용 전 법무 확인 | `RL_DATA.md` §3 |

## 보류·추적

| 항목 | 판정 | 재평가 시점 |
|---|---|---|
| T3 (SimpleQA-V · LogicKor) · τ³ — MOPD1 학생 미측정 | 판정기 Gemini 가 HTTP 402 (11-11 05:35 확인). judge 를 Claude 로 바꾸는 작업이 다른 세션에서 진행 중 (`cd036a47e`) | 판정기 복구 뒤 학생 `hf_step_00120` 측정 |
| 282 LCB 같은 조건 재측정 | 사용자 결정 11-11 로 미룸 — 지금은 10-20 값 (GDN 절반 bf16) 15.99, LCB 회복률 82% 에 조건 차이가 섞인다 | 회복률을 정확히 써야 할 때 (16 GPU 약 15분) |
| `runs/eval_lcb.sh` 실패를 ABORT 로 · 측정용 Gym 기동에 `UV_OFFLINE=1` | 11-11 venv 시드 경쟁 실패를 `DONE` 으로 기록 (KNOWN_ISSUES 11-11). 실행기 동작 변경이라 승인 대기 | 다음 LCB 측정 전 |
| Pai 게이트 A2 에 fleet 식별 추가 · main1 의 35일 된 역터널 (`-R 8199`, pid 2233680) | A2 는 응답만 본다 (Pai KNOWN_ISSUES 11-11). 묵은 터널은 main1 fleet 를 가리켜 지금은 무해 | 에이전틱을 sub1 에서 돌려야 할 때 |
| MOPD 학생 AIME · HMMT 사고 닫힘 80.9% · 77.6% (282 AIME 96.4%) | 측정 조건이 달라 미확정 (`study/benchmark_results.md` §0.5) | 다음 수학 벤치 · MOPD2 |
| `TeacherWorkerGroup.shutdown` 의 `ray.is_initialized()` 가드 (`34ed1357b`) upstream 보고 | upstream main 에도 같은 결함 (KNOWN_ISSUES 11-09 rc=1) | rebase 때 |
| **배포·평가 GDN 재귀 상태 fp32 (사용자 결정 2026-10-21)** | 업계 조사(Qwen3.5~3.8·Nemotron 3 전부 fp32 기본, `archive/KNOWN_ISSUES_2026-08_10-07.md` GDN 재귀 상태 항목) 뒤 확정. Pai 서빙 스크립트 반영 `6ed19fc` · 문서 `ff82432`. 독립 검토(Opus 5.5 xhigh): BLOCKER 0 · HIGH 1 (fp32 + prefix caching align 미검증 → 벤치 fleet 사용 전 `prefix_cache_check` 재통과, 문서·스크립트 주석으로 반영) · MEDIUM 3 (align 모드 MAMBA_BLOCK 무효 정정 · 결과 기록에 dtype 없음 · 문서 갱신) 반영. **RL ckpt 를 Pai fleet 으로 평가하면 10-21 이전 teacher τ³ (bf16 상태) 와 계열이 다르다** **11-03: 8001 vLLM 배포에도 적용** — s2r step228 교체 기동 시 `--mamba-cache-dtype float32 --mamba-ssm-cache-dtype float32` (구 SFT2400 서버는 결정 이전 기동이라 auto=bf16이었음). | fleet 게이트 V1·V2 (GPU 승인 대기, Pai `STATUS.md` 열린 결정) |
| 장문(128K 생성) 확장 때 비동기 staleness·클립 재검토 (사용자 2026-10-14) | 지금은 `max_trajectory_age_steps: 2` · 클립 0.2/0.28. 생성 128K 로 늘리면 궤적이 refit 2번 안에 못 끝나 age 초과 폐기가 늘므로, MiMo-V2.6 처럼 staleness 4 + 양의 advantage 클립 상한을 5.0 까지 여는 조합(엔트로피 연동 조절)을 검토한다 — `RL_PLAN.md` §2 검토 예정 둘째 항 | MOPD2 (SWE·장문 agentic) 레시피 때, 지표는 age 초과 폐기 비율·클립 발동률·gen KL |
| FlashQLA E2E 스텝 이득 | 커널 단위 검증만 완료(K1·M2). 레시피 opt-in | 첫 장문 RL 런 |
| vLLM 롤아웃측 FlashQLA | prefill 한정 이득, vllm 코드 변경 필요 — 후순위 | 롤아웃이 병목일 때 |
| fla 0.4.2 네이티브 GQA NaN upstream 보고 | 2026-08-18 "보고 예정" 기록 뒤 실행 여부 미기록 (mcore 는 MHA 확장으로 우회 중, 원장 #14) | — |
| RL 데이터 후속 4건 (Gym 프리페치·Math-v2 복원·SWE 상한·litmus-bench) | 미착수 | `RL_DATA.md` §4 |
| MiMo-V2.6-RL Code — agentic teacher(SWE) 후보 (사용자 2026-10-21) | 조사·기록 완료 (GPU 미사용). 진행하면 권고 1~5 대로 — 첫 작업은 alpha 통과율 측정 (`RL_PLAN.md` 결정 40) | MOPD1 뒤. 착수 때 `study/mimo_code_harbor.md` §7 재확인 (원본 갱신 · upstream Gym Harbor 학습 지원) |
| Muon optimizer-state offload 포팅 (Pai 기능 #4) | R4 실측상 1노드 128K/CP8 에 불필요 (ES 조건) — 보류 | 컨텍스트 > 128K 이거나 ES 기각 시 |
| logprob backward 메모리 절감 (int64 `one_hot` → scatter, chunk 축소, CP 호환 fused linear logprobs) | ES 로 128K 가 들어가 불필요 — 보류. chunk 1024 는 peak −3.8 GB 실측 | ES 기각 시 |
| R4 하네스 보완 (2026-10-21 적대적 검토, `GATES.md` R4 행) | 합성 `prev_logprobs` · reference 경로 없음 · torch 밖 메모리 미측정 3건이 남음. 코드 변경이라 승인 대상 | 다음 R4 재실행 전 (CP4×DP2 검토 등) |
| RL 환경 데이터의 도구 `description` 필수 검사 | 현 블렌드 0건이라 미착수 (R3 렌더 차이 #2) | 새 RL 환경 추가 시 |

## 완료

| 트랙 | 결과 | 정본 |
|---|---|---|
| **MOPD1 (2026-11-09 ~ 11-12, `RL_PLAN.md` 결정 55~58 · 60)** | 학생 Code·Math 1단계 282 · teacher General 640 · STEM 735 · Code·Math 228 · 도구 400 · 512 × 120 · 시퀀스 평균 손실 · warmup 3 · lr 3e-6 — `mopd1/s3` 14.6 h rc 0 (rc=1 수정 `34ed1357b` 확인). SFT 학생 런 s1 (토큰 평균) · s2 (시퀀스 평균) 는 Code·Math 끝맺기 정체로 멈춤. 학생: LCB 23.06 (282 15.99 · 228 24.61, 회복 82%) · MMLU-Pro 63.4 · GPQA-D 46.0 (STEM teacher 초과) · IFEval 85.8 · IFBench 53.1 (회복 70 · 77%) · SWE-V 12.0 · TB-2.1 2.9 · RULER 유지 | `GATES.md` MS1 · MS2 · `study/benchmark_results.md` §0.5 · `KNOWN_ISSUES.md` 11-09~11-11 · `FRONTIER.md` MOPD 행 · 지난 행 `archive/STATUS_2026-10-08_11-12.md` |
| **1차 teacher RL (2026-10-08 ~ 11-08, `RL_PLAN.md` 결정 18·19)** | teacher 넷 확정 — 도구 400 (결정 26) · STEM 735 · Code·Math s2r 228 (결정 48) · General 640 (결정 54). Code·Math 는 1단계 282 → 2단계 175 → s2r 228 → 코드 · 수학 라운드 (결정 44~47, MOPD1 에는 228) | `study/benchmark_results.md` §0 · §2 · §4 · §4.5 · 지난 행 `archive/STATUS_2026-10-08_11-12.md` |
| 검토 규칙 재설계 (2026-10-21, 사용자 결정) | 작업 종류별 사전 검토 → **문제 진단 독립 검증** + **스모크·게이트 적대적 검토** + **프론티어 비교표**. 규칙은 공통 원칙, 절차는 사용자 skill `problem-diagnosis`·`frontier-research`, 대장 `FRONTIER.md` (`8f9494e50`). 적용 전 검증: 10-20 OOM 블라인드 2단계 보정 6/6 (+ 새 관측 3, 미교차검증) · 발동 시험 3/3 · R4 원본 하네스 게이트 검토가 10-07 결함 2건을 다시 찾음 | 공통 원칙 · `docs/FRONTIER.md` |
| 에이전트 설정 정리 (2026-10-21, 사용자 승인) | upstream skill 6개 끄기 `31c3e0964` · `alpha-launch` skill `e488c5c74` · 사람 수준 규칙을 공통 사용자 CLAUDE.md 로 이관 `37b30120b` (Pai `d8f8e8b`). 독립 검토(Opus·xhigh)는 사용자 지시로 적용과 병행: BLOCKER 0 · HIGH 2 (검토 전 적용 = 사용자 지시 · Pai 공유 index 경합 = 커밋으로 해소) · MEDIUM 5 · LOW 8 → 결함 수정 반영 (세션 점검 문구 · migrate/mirror 가 링크 삭제 · 일반 파일 백업 가드 · 낡은 참조 · 경로). 메모리 정리 완료 — 두 프로젝트 피드백 16건 삭제, ko_chat 결정 이력 1건 유지 (백업 `/home/work/vidsearch/.claude-config/memory_backup_20261021/`). 남은 것: 정본 이력 없음(D안 수용) · 세션 점검 문구는 단순 질문에서 무시됨(시험) · sub1 로드 검증은 sub1 로그인 뒤 | `/home/work/vidsearch/.claude-config/CLAUDE.md` · `.claude/rules/alpha.md` |
| 첫 RLVR 런 · v2 · 속도 최적화 (2026-10-07 ~ 10-08, `RL_PLAN.md` 결정 15~18) | v1 첫 런은 정확성 결함 4건과 7스텝 멈춤으로 정지 → upstream 이식 7건 뒤 v2 설정 G8 PASS. 속도 레버 E1 채택 (정상 상태 15.8 → 12.3 분/스텝, KL 같음). rollout 긴 꼬리 계측 T0: 긴 요청 대기 0~3 s, 64K 디코딩 17~37분이 임계 경로 → 우선순위 시험 생략, 투기 디코딩 기각. advantage 0 제외는 Z1 PASS 뒤 기본 꺼짐 병합. 현 속도 E2E 1,650~1,830 tok/s/GPU 는 NeMo-RL H100 기준 범위이고 학습 MFU 는 약 5% 다. Ultra 형태 일정 42~44일 → 결정 18 로 규모 축소 (1차 teacher 트랙). 모두 1~10스텝 시험 런이라 학습된 모델은 없다 | `RLVR_READINESS.md` §5.1~§5.6 · `GATES.md` G8·Z1 |
| RL 인프라 준비 (2026-08-12 ~ 08-21) | fork 3종 · AlphaBridge · vLLM 플러그인 · 게이트 M1~M3·K1·D1 · RL 블렌드 2종 · FlashQLA opt-in 통합 · 2노드 분리 토폴로지 확정 | `archive/POSTTRAIN_PREP_2026-08.md` |
| vLLM 단독 서빙 정량 게이트 (2026-08-24) | M4 PASS | `GATES.md` M4 |
| Pai 컨테이너 공존 설치 (2026-10-06) | 시스템 무변경 구동, 원장 #15~#20 해결, E1~E3 PASS | `SETUP.md` §1 |
