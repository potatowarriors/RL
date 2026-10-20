# alpha RL — Known Issues & Fixes

alpha RL 단계(NeMo-RL)의 사고·수정 기록 전문이다 (최신순). [`../CLAUDE.md`](../CLAUDE.md) 의 "함정 표"가 이 문서의 한 줄 요약이다.
새 사고는 **여기에 서사를 쓰고 함정 표에는 한 줄만** 추가한다. 날짜는 절대 표기한다.
2026-10-07 이관: 워크스페이스 `project_s/NEMO_RL_SETUP.md` §4 원장 21건과 Pai `KNOWN_ISSUES.md` 10-06 항목의 RL 측 서사를 옮겼다.
pre-train·SFT·벤치 쪽 사고는 Pai `examples/alpha/docs/KNOWN_ISSUES.md` 가 정본이다.

## 초과 할당(결정 31)이 첫 본 런에서 켜지지 않았다 — 데이터로더 배치 64 가 목표당 68 을 한 번에 못 띄움 (2026-10-14 ✅ 수집기 루프 수정, `s1` 재기동으로 검증)

- **증상**: Code·Math 1단계 `s1` (`rollout_overprovision: 4`) 의 드라이버 로그에 "68 groups launched per target step" 은 찍혔지만, 수집기는 목표 0·1·2 에 각각 "batch worker for 64 prompt groups" 만 띄웠다. 결정 33 의 본 런 검증 항목(이월·"Keeping unused prompts")이 하나도 나오지 않았다.
- **원인**: `_process_batch` 가 `min(batch.size, needed)` 만 띄운다. 데이터로더 배치가 `num_prompts_per_step`(64) 라 68 중 64 만 launch 되고, 나머지 4 는 그 목표가 `_generating_targets` 에서 풀린 뒤(= 첫 worker 의 64 그룹이 모두 끝난 뒤)에야 gap fill 로 띄워진다. 그때는 이미 64 가 모여 학습이 나간 뒤라 "N+k 가 동시에 떠 있어야 낙오를 흡수한다" 는 전제가 성립하지 않았다. 단위 테스트는 루프를 mock `_process_batch` 로만 검사해 이 상호작용을 놓쳤다.
- **수정 (`alpha/async-overprovision` `eb49dfaad`)**: 초과 할당 모드에서 배치가 `needed` 보다 작으면 `_process_batch` 가 목표를 되돌려 주고 `-needed` 를 반환하고, 루프가 다음 배치를 `BatchedDataDict.from_batches` 로 이어 붙여(`batch_max_length` 재계산) 다시 시도한다. 데이터로더가 끝나면 남은 행으로 launch 한다 (`_allow_partial_launch`). 테스트 3개 추가 (이어 붙이기·끝에서 부분 launch·실제 BatchedDataDict 결합), 12/12 PASS.
- **운영**: 실행 중인 런이 import 하는 트리에는 병합하지 않고, `ckpt/step_5` 저장 뒤 `s1` 을 멈추고 병합·재개(`s1r1`)해 첫 step 로그에서 68 launch · 이월 · GBS 1,024 를 확인한다.

## `main2r7` 이 step 733 앞에서 6시간 멈췄다 — 낙오 궤적 정체 2번째 (2026-10-14 ✅ 사용자 승인으로 중단, `ckpt/step_730` 에서 `main2r8` 재개)

- **증상**: step 732 학습 완료 01:29:01 뒤 "STALLING: Need 256 trajectories for step 732, but only 255 are ready" (버퍼 767/768) 가 07:35 까지 반복. 엔진 6개 중 `vllm_policy-0-5` (main1 GPU 5) 만 생성 토큰이 늘었다 — 02:47 4.3 tok/s, 03:55 4.6 tok/s, 03:55~05:49 평균 1.5 tok/s, 05:49 이후 1분 창에서 0. 01:27 부터 누적 약 4.5만 토큰으로 턴당 상한 32K 를 넘겼으므로 멀티턴 행이다. 나머지 15장은 0% (GPU 5 는 15분 중 10~22% 표본에서만 버스트).
- **10-13 13:30 정체와의 차이**: 그때는 10~22 tok/s 로 55분 만에 저절로 풀렸다. 이번은 4 tok/s 이하로 느려지며 6시간 동안 풀리지 않았다. 256 배치에서는 step 마다 그룹 256개를 다 기다리므로 낙오 1개가 걸릴 확률이 64 배치의 4배다 (`RL_PLAN.md` 결정 29 위험 항목).
- **대응**: 01:53 · 02:48 · 03:56 알림, 07:35 사용자 지시("아직도 정체면 중단해")로 드라이버 SIGTERM (rc 134) → 07:37 `ckpt/step_730` 에서 `main2r8` 재개 (731·732 두 step 손실, 같은 오버라이드). GPU 16장 유휴 약 6시간.
- **대책 (결정 31, 2026-10-14 구현)**: 롤아웃 초과 할당 + 이월 — `grpo.async_grpo.rollout_overprovision` (기본 0). 수집기가 N+k 그룹을 띄우고 버퍼는 N 이 모이면 학습, 남은 그룹은 다음 step 으로 이월. 단위 테스트 PASS, GPU 스모크는 Code·Math 전 GPU 창에서. 남은 것: ② 턴당 생성 상한 축소 ③ 감시 신호 "스텝 무진척 + 엔진 1개만 토큰 증가" 자동화 (진단은 `$CLAUDE_JOB_DIR/tmp/engine_rate_probe.py` 방식 — 이름 붙은 액터 `vllm_policy-0-N` 의 `get_vllm_logger_metrics` 를 60초 간격 2회). 낙오자가 느린 기전은 여전히 미확정.

## 256 프롬프트 · 48K 런이 main1 호스트 메모리 부족으로 죽었다 — CP8/DP1 이라 학습 워커 8개가 전체 배치를 각자 보유 (2026-10-20 ✅ 128 로 재시작 `s4`)

- **증상**: `teacher_codemath_s3/s3` step 3 학습 시작 직후 `ray.exceptions.OutOfMemoryError: 11 worker(s) were killed … 1743.02GB / 1772.00GB`. raylet `threshold_memory_monitor`: 사용 1,855 GB > 임계 1,808 GB (95% of 1,903 GB). cgroup `memory.peak` 1,903 GB 로 실제 소진. 10-13 과 달리 디스크가 아니라 RAM.
- **기전**: OOM 보고서의 상위 사용자는 MegatronPolicyWorker 8개 각 145~160 GB. 학습 배치 객체는 84 GB (`bytes created by worker: 84,351,691,549`; 4,096 시퀀스 × 평균 17.5K 토큰 = 72M 토큰, 토큰당 약 1.2 KB — R3 라우팅 기록이 대부분). 레시피가 CP8·EP8 이라 DP=1 이고 NeMo-RL 은 DP 샤드 단위로 데이터를 나누므로 워커마다 전체 배치 + 변환 복사본을 호스트에 올린다. 여기에 raylet 77 GB · spill IO 워커 3개 약 180 GB(1.1 TB spill 처리 뒤 메모리를 반환하지 않음) · Pai shm 66 GB. 1·2단계(64, 21 GB 배치)와 10-13 STEM 256(응답 5K, 약 24 GB)은 임계 아래였다.
- **대응**: 사용자 결정으로 128 프롬프트(배치 42 GB → 워커당 약 80 GB)로 `hf_step_00175` 에서 새로 시작 (`61a1e8c4a`, 워밍업 0). 호스트 사용량 1.5 TB 경보를 watcher 에 추가. spill 디렉토리 이름 오류(함정 표 10-20)도 같이 바로잡음.
- **교훈**: 프롬프트 수를 올릴 때 GPU 메모리만이 아니라 **호스트 메모리 = 배치 객체 크기 × CP 랭크 수(+ 복사본)** 를 먼저 계산한다. 256 을 다시 쓰려면 CP4×DP2(64K 패킹) 로 워커당 배치를 절반으로 줄이거나(R4·G2 재실행), 워커가 자기 몫만 받도록 데이터 경로를 바꿔야 한다.

## DAPO 동적 샘플링이 async 수집기를 교착시켰다 — 배치가 짧게 끝나면 아무도 깨우지 않음 (2026-10-20 ✅ `57f86ee1e`, `teacher_codemath_s3/s1` → `s2`)

- **증상**: `teacher_codemath_s3/s1` (256 프롬프트 · 초과 할당 250 · `drop_zero_std_groups`) 이 step 0 검증 뒤 1,012 그룹을 생성하고(14:36) 학습을 시작하지 못했다. 드라이버는 "need 59 more trajectories for step 0" 로 대기, sub1 GPU 0%, 수집기는 "⏸️ Pausing collection: all target weights [0] … Waiting for weight update".
- **기전**: 실측 버림률 60% (전부 오답 57% · 전부 정답 3.6%) 라 목표 0 의 506 그룹 중 남은 것이 197 < 256. 수집기는 두 목표(0, 1)가 모두 진행 중일 때 생성 한도 일시정지에 들어갔고, 이 정지는 `set_weight_version`(학습 뒤) 으로만 풀린다 — 학습은 목표 0 이 차야 가능하므로 교착. 깨우더라도 `_calculate_target_weights(v)` 가 v+1..v+age 만 돌려줘 부족한 현재 목표 v 는 보충 후보가 아니었다. 버림이 없으면 자기 배치만으로 항상 N 을 넘어 드러나지 않던 경로.
- **수정** (`drop_zero_std_groups` 켤 때만): ① `_release_target` 이 생성 한도 이벤트를 set 해 루프를 깨운다 ② 목표 후보에 현재 목표 v 를 포함한다 (소비된 목표는 호출자가 거른다). 회귀 테스트 2개 (`test_async_dynamic_sampling.py`).
- **독립 검토 (10-20 16:00~16:15, 새 에이전트 2회)**: 1차가 수정 뒤에도 남은 ms 창의 깨우기 유실 경합(F1)과 보충 생성이 N+k 보유를 목표로 해 잉여가 나이 제한으로 폐기되는 낭비(F2, step 당 최대 1시간)를 찾았다 → `a3b6a3380` (clear→재확인→대기 · 무조건 깨움 · 첫 배치 N+k 이후 ceil((N−보유)/유지율) · 버퍼 (age+1)). 2차 검토 SAFE (LOW 6). `s3` 16:26 재시작.
- **교훈**: 수집기의 "목표당 N+k 생성 뒤 다음 목표" 설계는 그룹이 버려지지 않는다는 가정 위에 있다. 그룹을 버리는 기능을 넣으면 "목표가 N 에 이를 때까지 계속 생성" 이 보장되는지 코드 경로(일시정지·후보 목록·이월)를 모두 봐야 한다. 초과 할당 k 는 이월 깊이일 뿐 정확성 조건이 아니다. 손실: 생성 2시간 40분.

## `main2r6` (256 프롬프트) 가 step 729 에서 죽었다 — Ray 객체 spill 이 main1 루트 디스크를 채움 (2026-10-13 ✅ 디스크 정리 뒤 `main2r7` 재개, 근본 대책은 클러스터 재기동 때)

- **증상 (23:39, rc 1)**: step 729/735 학습 뒤 드라이버가 `log_generation_metrics_to_wandb` → `wandb run.log(figure)` 의 JSON 덤프에서 예외로 종료. 그 앞에 raylet 의 `OSError: [Errno 28] No space left on device: /tmp/ray/session_…/ray_spilled_objects/…` 4,822회, Ray 로그 싱크 기록 실패, 뒤이어 엔진 셧다운과 Gym 의 `ClientOSError` 334회·`ServerDisconnectedError` 72회 (후행 증상). 마지막 체크포인트 `ckpt/step_725` (22:52, NFS) 는 무사.
- **기전**: 256 배치는 리플레이 버퍼·DataPlane 에 궤적 12,288개(3 목표 × 256 × 16)를 Ray 객체 저장소(main1 `/dev/shm` 128 GB, Pai 가 65 GB 사용)에 올린다. 넘친 객체를 Ray 가 `/tmp/ray/…/ray_spilled_objects` (루트 overlay 878 GB) 로 흘렸고 이 런에서 누적 555 GB · 1,307 객체. 루트는 이미 617 GB 가 차 있었다 — 그중 257 GB 가 Claude 세션 scratchpad 잔재 (Pai 세션 `c0a8d222` 221 GB (10-07) · NeMo-RL 세션 `d08e8a82` 22 GB · `87daa8cf` 14 GB). 64 배치 `main2r3` 는 spill 0.
- **대응**: 사용자 승인 (23:47 "바로 진행해") 으로 scratchpad 삭제 — `nrl_probe` 183 GB · `87daa8cf/scratchpad` 14 GB 는 지웠고 (루트 617 → 420 GB 사용, 413 GB 여유), `gym060` 38 GB · `d08e8a82/scratchpad` 22 GB 는 자동 권한 분류기가 막아 사용자 직접 삭제 요청. `ckpt/step_725` 에서 `main2r7` 재개 (23:49, 같은 오버라이드, 726~729 의 4 step 손실). 안전망 cron 에 `df -h /` 150 GB 경고 추가.
- **근본 대책 (다음 Ray 클러스터 재기동 때)**: `ray start` 의 객체 저장소 크기 명시와 spill 경로를 NFS 로 (`object_spilling_config`). 지금 클러스터는 10-07 부터 Pai 세션과 공유라 재시작하지 않았다. 256 이상 배치를 쓰는 레시피는 기동 전 루트 여유 ≥ 300 GB 를 확인한다. 죽은 런의 spill 잔재 47 GB (`ray_spilled_objects`, 23:30) 는 Ray 소유라 두었다.

## 프롬프트 수를 64 → 256 으로 바꿔 `ckpt/step_710` 을 재개하자 두 번 즉시 종료됐다 — 스케줄러 샘플 수 assert · 에폭 클램프 (2026-10-13 ✅ 오버라이드 2개로 해결, `main2r6`)

- **배경**: 결정 29 (`RL_PLAN.md`) 로 `main2r3` 를 step 717 에서 멈추고 `grpo.num_prompts_per_step=256 grpo.max_num_steps=735 checkpointing.save_period=5` 로 재개했다. 레시피는 GBS 를 프롬프트 수 × 16 으로 묶어 두어 GBS 도 1,024 → 4,096 이 된다.
- **1차 `main2r4` (19:11, rc 1)**: 학습 워커 초기화의 `load_checkpoint` 에서 `OptimizerParamScheduler: class input value 40960 and checkpoint value 10240 for warmup iterations do not match`. Megatron-Bridge 는 `lr_warmup_iters: 10` 을 iters × GBS = 샘플 수로 바꿔 체크포인트에 저장하고, mcore 스케줄러는 `use_checkpoint_opt_param_scheduler`·`override_opt_param_scheduler` 가 둘 다 꺼져 있으면 값을 대조한다 (G7 의 wd 반복 수 불일치와 같은 기전, 이번엔 GBS 쪽). 대응: `++policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler=true` — 체크포인트 값(warmup 10,240 샘플 · decay 102.4M 샘플)을 그대로 쓴다. 누적 샘플 727,040 이 warmup 을 지났고 lr = min_lr = 5e-6 상수라 학습률은 바뀌지 않는다.
- **2차 `main2r5` (19:21, rc 0)**: 기동 직후 "Async GRPO training is already complete: current step 710 reached the effective limit of 205 steps". `async_grpo_train` 이 `max_num_steps = min(max_num_steps, max_num_epochs × len(dataloader))` 로 자르는데 (`grpo.py:4349~4362`), `len(dataloader)` 가 52,500 // 256 = 205 로 줄어 710 보다 작아졌다. 대응: `grpo.max_num_epochs=4` (4 × 205 = 820 ≥ 735). 실제 데이터 소비는 45,440 + 6,400 = 51,840 < 52,500 이라 1 에폭 안이고, `StatefulDataLoader` 상태는 샘플 단위(`samples_yielded`)라 위치가 이어진다 — gap fill 첫 task index 45,918 로 확인.
- **아찔했던 점**: rc 0 이라 P0-3 오케스트레이터가 정상 종료로 읽고 fleet 기동으로 넘어가기 직전이었다 (폴링 2분 주기, 19:28:23 수동 중단, fleet 미기동). 오케스트레이터에 "최종 ckpt `step_<final>` 이 없으면 ABORT" 가드를 넣었다.
- **3차 `main2r6` (19:29, commit `e6992924e`)**: 두 오버라이드를 더해 기동. 리플레이 버퍼 복원 221 그룹(64-배치 런이 711~713 용으로 모은 것) → 첫 256-step 용 192 그룹 gap fill. 결과는 `STATUS.md`.
- **교훈**: 프롬프트 수·GBS 를 바꿔 재개할 때는 (1) 스케줄러 플래그 (2) 에폭 클램프 (3) 데이터로더 위치 (4) 후속 자동화의 "정상 종료" 판정 네 가지를 먼저 본다. 함정 표 10-13.

## `main2r3` 가 step 647 뒤 약 1시간 멈췼다 — 낙오 궤적 1개가 엔진 1장에서 느리게 생성되는 동안 GPU 16장이 대기 (2026-10-13 🔶 저절로 풀림, 기전 미확정)

- **증상**: step 647 완료(13:30:25) 뒤 "Step 648/820" 머리글만 반복. 14:2x 에 저절로 풀려 step 648 학습이 이어졌다 (사용자가 GPU 재가동을 먼저 봄). 재시작하지 않았다.
- **멈춘 기전 (확정, `nemo_rl/algorithms/async_utils/replay_buffer.py`)**: 리플레이 버퍼는 "현재 스텝용(target weight)으로 지정된 그룹 64개가 전부" 있어야 샘플링한다.
  647 용은 63/64, 648·649 용은 64/64 로 버퍼 191/192 였고, 코드는 다른 스텝용으로 채우는 것을 금지하며 시간 제한도 없다 ("STALLING: Need 64 … only 63 are ready").
  버퍼(192 = 3 스텝분)가 차서 새 프롬프트도 나가지 않아 학습 8장·롤아웃 6장·judge 2장이 모두 0% (전력 112~126 W) 였다.
- **낙오자 (증거)**: sub1 엔진 `vllm_policy-0-4` 만 생성 토큰 누계가 늘었다 (45초 +430 = 9.6 tok/s, 60초 +1,311 = 21.8 tok/s), 나머지 5개는 0.
  GPU 4 는 약 39초 주기로 4초만 690 W 로 100% 돌고 나머지는 유휴 — 단일 시퀀스의 정상 디코드(40~65 tok/s 연속)와 다르다. 4초 버스트는 긴 문맥을 매 주기
  다시 prefill 하는 시간과 비슷하다. 후보는 (a) 멀티턴 루프가 매 턴 전체 문맥을 다시 보냄 (RL 엔진은 prefix caching 끔, D2) — 그러나 남은 행은 reasoning_gym
  (도구 없음) 이었고 sandbox 터널·reasoning_gym·judge 서버는 유휴였다, (b) vLLM 이 요청을 반복 선점·재계산 — 그러나 KV 는 거의 비어 있었다.
  확정 수단이 없었다: 엔진 서버는 `/v1/chat/completions`·`/tokenize` 만 열어 선점 지표를 못 읽고, `get_vllm_logger_metrics` 는 4개 키뿐, py-spy 는 컨테이너 ptrace 거부.
- **FlashInfer 교착(10-11)과 구별**: 그때는 엔진 1개가 SM 100%·전력 유휴로 영원히 돌았고, 이번은 느리게나마 토큰이 늘었다. 10-08 "in-flight 진척 0" 과도 다르다 (진척 0 아님).
- **빈도**: 647 스텝 × 1,024 ≈ 66만 샘플 중 1회. 턴당 32K 상한에 닿는 샘플은 매 스텝 있지만 (max_gen_tokens_per_turn max=32768) 정상 속도면 10~13분이라 스텝 안에 묻힌다.
- **대응 (사용자 판단 대기)**: ① 리플레이 버퍼에 대기 상한을 두고 넘으면 다른 스텝용 유효 궤적으로 채우거나 낙오 그룹을 버림 (본체 수정, 별도 커밋) ② 턴당 상한 축소
  ③ 정체 감시에 "스텝 무진척 + 엔진 1개만 토큰 증가" 신호 추가. 진단 명령은 `scratchpad` 의 `engine4_rate.py`·`gpu4_burst_probe.py` 방식 (액터 `get_vllm_logger_metrics` 의 `generation_tokens` 차분).

## `main2r2` 가 재개 첫 학습 스텝(step 41)에서 죽었다 — sub1 rank 2 의 NCCL all-reduce 가 `Cuda failure 2 'out of memory'` (2026-10-11 ✅ 옵티마이저 상태 재개 + 지연 생성 NCCL 통신자 버퍼; Pai 07-15 와 같은 결함 → `NCCL_MAX_NCHANNELS=16`)

**발견 경위**: `gdn_prefill_backend: triton` 으로 `ckpt/step_40` 에서 재개한 `main2r2` (12:37 기동) 가 12:52 step 41 머리글을 찍고 롤아웃 배치 1,024 개를 받아
(보상 평균 0.2715) logprob 까지 계산한 뒤, 학습(`policy.train`) 에서 12:58 에 죽었다 (rc 1). FlashInfer 교착과는 무관한 학습 노드 문제다.

**오류**: `MegatronPolicyWorker[rank=2]` (sub1, pid 2942071) 만 `torch.distributed.DistBackendError: NCCL error … unhandled cuda error … Cuda failure 2 'out of memory'`.
위치는 `finalize_model_grads → _update_router_expert_bias → get_updated_expert_bias → all_reduce(tokens_per_expert, group=tp_dp_cp_group)`
(`megatron/core/transformer/moe/moe_utils.py:1223`) — backward 가 끝난 직후, 옵티마이저 step 전이다. 다른 7 rank 는 정상이었다.
NCCL 2.30.7. 이 all-reduce 는 프로세스에서 **처음 쓰는 통신자 그룹**이라 NCCL 이 이 순간 채널 버퍼를 `cudaMalloc` 한다.

**관찰 (워커 `[GPU_DIAG] after_model_setup`, 체크포인트 로드 뒤 · 학습 전)**:

| 런 | 시작 방식 | rank 0 allocated / reserved | 다른 rank |
|---|---|---|---|
| `main2` (10-10) | iter2862 새로 시작 | 27.6 GB / 30.9 GB (rank 1) | rank 4 26.4 / 29.7 |
| `main2r1` (10-11 01:29) | `step_20` 재개 | 33.6 / 46.5 | rank 4·5 33.7~33.8 / 46.7~46.9 |
| `main2r2` (10-11 12:37) | `step_40` 재개 | 33.6 / 46.5 | **rank 2 37.4 / 52.9** (NVML used 55.5 GB, free 26.0 GB) |

- 새로 시작하면 바닥이 27 GB (R4 의 "고정 바닥 27.0 GB" 와 같다). 재개하면 Muon momentum 이 체크포인트에서 바로 올라와 allocated +6 GB, reserved +16 GB 다.
  새 런은 momentum 을 첫 옵티마이저 step 에서 지연 할당하므로, 첫 all-reduce 시점에는 아직 없다.
- R4 실측 (128K/CP8, expandable_segments) 의 peak reserved 59.5 GB 는 바닥 27 GB 위에 32.5 GB 를 더 쓴다. 재개 바닥 46.5~52.9 GB 에 이를 더하면 79~85 GB —
  rank 2 는 80 GB 를 넘는다. PyTorch 는 자기 할당이 막히면 캐시를 비우고 다시 시도하지만 NCCL 의 `cudaMalloc` 은 그런 재시도가 없어 먼저 죽는다.
- `main2r1` 의 step 21 재개가 살았던 것은 여유가 몇 GB 있었기 때문이고, 재개 뒤 통신자가 한 번 만들어지면 (step 22~40) 같은 all-reduce 는 문제가 없다.
  즉 **재개 직후 첫 학습 스텝만** 위험하다.
- rank 2 가 다른 rank 보다 6 GB 더 쓴 것은 우연이 아니다 — `main2r3` (학습 main1) 에서도 **rank 1·2 가 37.4 / 52.9 GB**, 나머지 6개는 33.6~33.8 / 46.5~46.9 GB 로
  같은 모양이다. EP8·CP8 분할에서 두 rank 가 더 큰 조각을 갖는 결정적 배치로 보인다 (어느 텐서인지는 미확인). 재개마다 이 두 rank 의 여유가 가장 작다.

**원인**: Pai `train.sh` 가 2026-07-15 에 같은 클러스터에서 확인한 결함과 같다 — "NCCL 기본 64 채널은 통신자마다 큰 GPU 버퍼를 할당한다. 재개 때는
로드된 옵티마이저 상태가 지연 생성 통신자보다 먼저 메모리를 채워 첫 collective 에서 NCCL 'out of memory' 가 난다. `NCCL_MAX_NCHANNELS=16` 이 그 발자국을
1/4 로 줄이고 스텝 시간 비용은 없다 (60.7 s vs 61.1 s, NVLink 포화 유지). 새 런은 영향 없다" (Pai `examples/alpha/train.sh:85-92`, Pai `CLAUDE.md` 07-15 행).
NeMo-RL 레시피에는 이 변수가 없었다 (`GATES.md` G5 때 "없이도 재개 OOM 은 없었다" 고 적었는데, 그때는 재개 뒤 첫 스텝의 여유가 있었던 경우다).

**대응 (사용자 승인 13:45 "승인할게 바로 실행해", 커밋 `f1321fe64` — 15:40 `f625acac4` 작성자 교정 재작성 전에는 `935270af7`, `runs.log` 의 `commit=` 은 옛 값)**: `student_rlvr1_alpha.yaml` `policy.megatron_cfg.env_vars` 에 `NCCL_MAX_NCHANNELS: "16"` 추가
(`lm_policy.py:147` → Ray runtime_env 로 학습 워커 8개에 전달; 검사기 ERROR 0 · WARN 4, 렌더링 확인). 13:47 `ckpt/step_40` 에서 `main2r3` 기동.
검증 계획: 학습 워커 `/proc/<pid>/environ` 에 변수가 있는지 · 재개 첫 스텝이 지나가는지 (NCCL 버퍼) · R1 (step 41~43 gen KL < 0.002, 마스킹 0) ·
스텝 시간이 step 21~40 과 같은지 (NCCL 채널 축소 비용). 롤아웃 노드의 vLLM 엔진에는 이 변수를 주지 않는다 (refit broadcast 는 eth0 TCP 라 채널 수와 무관).

**결과 (10-11 14:10)**: 이번 기동은 학습이 main1, 롤아웃이 sub1 에 배치됐다. 학습 워커 8개의 `/proc/<pid>/environ` 에 `NCCL_MAX_NCHANNELS=16` 확인.
**재개 첫 학습 스텝(step 41) 통과** — 13:48 첫 머리글 … 14:09 step 42 머리글. step 41: gen KL 0.0018 · 보상 0.2705 · 스텝 320 s · 학습 2,925 tok/s/GPU
(`main2r1` step 21 재개 때 2,564) · refit 거품 28.8 s. 같은 체크포인트·같은 rank 메모리 (rank 1·2 52.9 GB 예약) 에서 NCCL 버퍼 OOM 이 사라졌다.
**R1 PASS (14:17)**: step 41·42·43 gen KL 0.00177 / 0.00180 / 0.00198 · 마스킹 0 · 스텝 320 / 175 / 281 s (`GATES.md` R1 행). 두 변경 (Triton GDN prefill · NCCL 16채널) 이
정합과 처리량을 해치지 않는다. FlashInfer 교착의 재발 없음은 step 80 무렵 (이전 재발 간격 약 20스텝의 두 배) 까지 정지 없이 가야 확인된 것으로 본다.

## `main2r1` 이 step 41 배치 대기에서 멈췄다 — vLLM 엔진 하나의 EngineCore 가 GPU 100% 로 스핀, 3번째 재발 (2026-10-11 ✅ FlashInfer 0.6.13 GDN prefill 커널 교착 → `gdn_prefill_backend: triton`)

**발견 경위**: 사용자 보고 요청 (10:18) 으로 보다가 찾았다. step 40 은 04:16:40 에 끝났고 (ckpt/step_40 저장), step 41 은 6시간째 배치를 기다리고 있었다.
20분 정체 감시는 울리지 않았다 — 드라이버가 0.5초마다 "Step 41/820" 머리글과 버퍼 진단을 다시 찍어 (41,834회, 로그 92만 줄) 로그 mtime 이 계속 갱신됐다.
감시는 로그 mtime 이 아니라 스텝 번호 진척을 봐야 한다.

**관찰** (읽기만으로, Ray 작업으로 main1 컨테이너 안을 봤다):
- replay buffer 는 step 41 시작 약 3분 뒤 (04:20 무렵) 124 그룹에서 멈췄다. target 40 은 63/64, target 41 은 56/64, target 42 는 5/64. 이후 완료된 궤적 0.
- 이번 런은 롤아웃이 main1 이다. main1 GPU 5 만 사용률 100%, GPU 0~4 는 0% (policy 엔진 6개 = TP 1 × 6, GPU 6·7 은 judge TP 2).
- 엔진 6개 중 `VllmAsyncGenerationWorker` pid 1254251 의 자식 `VLLM::EngineCore` (pid 1255799) 만 상태 R, 메인 스레드가 5초에 499 틱 (코어 1개 100%), 자발적 문맥 전환 0.
  다른 5개 EngineCore 는 futex 대기. 생애 평균 CPU 도 88.8% 대 20% 로 혼자 높다. GPU 가 100% 인데 커널이 끝나지 않아 CUDA 동기화를 스핀 대기하는 모양이다.
- 오류·경고 로그 없음. vLLM 워커 로그는 모두 기동 직후 (01:3x) 가 마지막. `/proc/<pid>/{stack,syscall}` 은 권한 거부, py-spy 도 ptrace 거부 — 커널 이름은 못 봤다.
- Ray RUNNING: `NemoGym.run_rollouts` 3개 (03:56·04:02·04:08 시작, 롤아웃 배치 41·42·43), 수집기는 generation_limit_pause 로 새 프롬프트를 보내지 않는다 (버퍼 124/128 포화).
- 이번은 refit 일시정지가 아니라 생성 중에 멈췄다 (step 40 refit 28.7 s 정상 완료, version 40 궤적 5개 완료 뒤 정지). 10-08·10-10 은 일시정지에서 드러났지만 같은 상태다 —
  엔진 스텝이 끝나지 않으면 일시정지도 돌아오지 않는다.

**재발 간격**: 10-08 step 7 (첫 RLVR 런) · 10-10 step 22 (`main2`, sub1 GPU 2) · 10-11 step 41 (`main2r1`, main1 GPU 5). 노드·GPU 가 매번 다르다 → 하드웨어보다 소프트웨어 (커널·엔진) 쪽.
약 20스텝 (롤아웃 약 1,300 궤적) 마다 한 번. 820스텝 런은 이 상태로 끝까지 가지 못한다.

**추가 관찰 (10:40~11:30, 사용자 지시 "원인 분석·해결")**:
- GPU 5 는 SM 사용률 100% 인데 메모리 사용률 0%, 전력 120 W (유휴 GPU 0 은 126 W). 연산이 아니라 **깃발을 기다리며 스핀하는 커널**이다.
- 부모 액터 (`vllm_policy-0-5`) 안에서 `__ray_call__` 로 자식 `/proc` 을 읽었다 (ptrace_scope=1 은 조상만 허용): 메인 스레드 `syscall=running` (사용자 공간 스핀,
  cudaStreamSynchronize 폴링), CUDA 이벤트 스레드는 유휴. py-spy 는 부모가 띄워도 형제 프로세스라 거부됐고, ptrace 직접 호출은 분류기가 막았다.
- EngineCore 는 spawn 으로 떠서 faulthandler 가 없다 (SigCgt 에 SIGABRT 없음) — SIGABRT 로도 Python 스택은 안 나온다.

**원인 (upstream 확정, 우리 스택은 증상 일치)**: FlashInfer 0.6.13 (vLLM 0.25.1 고정 버전) 의 **SM90 GDN prefill JIT 커널 `FlatKernelTmaWarpSpecializedDeltaRule`** (`gdn_prefill_sm90`) 교착.
우리 엔진은 이 커널을 쓴다 (워커 로그 "Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128)"). decode 는 원래 Triton (`fused_recurrent_gated_delta_rule`).
- FlashInfer #3986 (2026-07-15, H100 · vLLM 0.21/0.25.0 · Qwen3.5 GDN · chunked prefill · 고동시성): 5~20분 또는 8~9시간 뒤 같은 정지. 관리자 (guangyunh-nv) 가
  `flat_collective_tma_warpspecialized_delta_rule.hpp` L959-961 의 옛 컴파일러 우회 코드를 원인으로 찍어 고쳤고 20시간 스트레스를 통과했다. **FlashInfer 0.6.15.post1 에 포함 — 0.6.13 에는 없다.**
  0 길이 시퀀스가 섞인 varlen 입력 (chunked prefill) 이 트리거다.
- FlashInfer #3329 (2026-05-14, H100 · 같은 커널): cuda-gdb 로 블록 하나가 mbarrier 대기에 멈춘 것을 확인했다. 확률적이다 (형상에 따라 0~100%).
  임시 조치 `--gdn-prefill-backend triton` 으로 0/35 정지, prefill 커널만 1.7~1.9배 느림.
- vLLM #50880 (2026-08-03, vLLM 0.25.1 · TP 1 · MoE · B300): SM 100% · 메모리 0% · 처리량 0 · EngineCore 메인 스레드 스핀 · Xid 없음 — 우리 관찰 목록과 같다. 미해결.
- 우리 관찰과의 대응: 블록 하나가 mbarrier 를 기다리면 커널은 상주 (SM 100%) 하되 메모리 접근·연산이 없고 (메모리 0%, 전력 유휴), 호스트는 동기화를 스핀한다.
  확률적 발생 (약 20스텝마다), 노드·GPU 무관. 10-10 의 "일시정지 무응답"도 같은 상태다 — 엔진 스텝이 끝나야 일시정지가 돌아온다.

**대응 (2026-10-11, 사용자 지시 "정확하게 원인 분석을 하고 해결해")**: `student_rlvr1_alpha.yaml` `vllm_kwargs.gdn_prefill_backend: triton` — FlashInfer 커널을 Triton/FLA
`chunk_gated_delta_rule` 로 바꾼다 (vLLM `arg_utils.py:730` → `additional_config`, `qwen_gdn_linear_attn._resolve_gdn_prefill_backend`). Triton 경로는 블록 간 mbarrier 파이프라인이 없다.
비용은 prefill 커널만 1.7~1.9배 — 이 워크로드는 프롬프트 평균 240 토큰 · 생성 8K 라 롤아웃 시간에 거의 안 보인다.
검증: 재개 첫 3스텝 gen KL < 0.002 (R1 기준) · 마스킹 0 · 엔진 로그 "Using Triton/FLA GDN prefill kernel (requested=triton)".
FlashInfer 0.6.15+ 로 올리는 것은 vLLM 0.25.1 의 고정 버전·JIT 캐시 패키지 (`flashinfer_jit_cache 0.6.13+cu130`) 와 어긋나 venv 재구축 (E1~E3) 이 필요 — 보류.
`main2r1` 은 중단하고 `ckpt/step_40` 에서 `main2r2` 로 재개한다. 정체 감시는 로그 mtime 이 아니라 스텝 번호 기준으로 바꿨다.

## STEM 본 런 `main2` 가 step 22 refit 에서 멈췄다 — vLLM 엔진 하나가 생성 일시정지에 답하지 않음 (2026-10-10, 10-08 7스텝 정지의 재발)

**발견 경위**: 정체 감시(30분)가 23:25 에 걸렸다. 드라이버 로그는 22:55:42 이후 한 줄도 없었다. 오류·경고도 없었다.

**관찰** (런을 멈추기 전, 읽기만으로):
- step 22 의 학습은 끝났다. main1 학습 GPU 8장은 0% 였다. 드라이버의 print 는 flush 하지 않는 것이 있어 로그 마지막 줄 ("Computing advantages") 은 실제 위치가 아니다.
- Ray 상태 API 에서 `AsyncTrajectoryCollector.prepare_for_refit` 와 `VllmAsyncGenerationWorker.pause_generation_async` 가 34분째 RUNNING 이었다.
  refit 전에 vLLM 엔진 6개에 `pause_generation(mode="keep")` 을 보냈고, 하나가 돌아오지 않았다 (`nemo_rl/models/generation/vllm/vllm_worker_async.py:1648`).
- sub1 GPU 8장 중 GPU 2 만 사용률 100% 이고 7장은 0% 였다. 롤아웃 수집도 690/1024 에서 함께 멈췄다.
- sub1 의 vLLM HTTP 포트별 ESTABLISHED 연결은 160 · 134 · 39 · 36 · 34 · 33 이었다. vLLM 워커 로그는 모두 기동 직후 (18:5x) 가 마지막이었다.
- py-spy 는 이번에도 ptrace 제한으로 "Permission Denied" 였다. 엔진 내부 스택은 보지 못했다.

**해석 (미확인)**: 10-08 사고와 증상이 같다 (엔진 하나만 GPU 100%, 진척 0, 오류 없음). 그때 이식한 원인 후보 수정 (교차 스레드 AsyncLLM · 문맥 초과 무응답) 뒤에도 재발했다.
GPU 를 100% 쓰면서 진척이 없으므로 끝나지 않는 GPU 연산 쪽이 더 유력하다. 일시정지는 엔진의 현재 스텝이 끝나야 돌아온다.

**대응**: 사용자 결정 (2026-10-10 23:35 무렵) 으로 중단하고 `ckpt/step_20` 에서 재개한다 (G7 경로, 태그 `main2r1` — 같은 태그는 `main2.log` 를 덮어쓴다). step 21~22 는 버린다.
중단은 2026-10-11 01:15:09 (rc 134, 사용자가 `kill -TERM` 실행 — Claude Code 자동 권한 분류기가 중단·기동 명령을 막았다). 정지 22:55 → 중단 01:15, GPU 16장 유휴 2시간 20분.
중단 뒤 런 액터 0, 멈춘 엔진의 GPU 2 점유도 풀렸다 (judge GPU 2장만 남음 — 실행기가 정리).
정체 감시를 20분으로 줄인다. 원인 조사는 재개 뒤 GPU 를 쓰지 않는 방법으로 병행한다 (upstream vLLM 의 같은 증상 수정 검색).

## J1 보충 j1c — rdkit 예/아니오 행이 맞는 답에 0 점 · 재판정 성공의 2/3 는 같은 답의 판정 뒤집힘 (2026-10-10 rdkit ✅ 수정 · 재판정 끔, 결정 27)

- **측정**: j1c (judge 128 + rdkit 64 행, W&B 오프라인 표, `tools/analyze_gym_full_results.py`) 의 첫 수집분 — equivalence 352 · ns_tools 368 · rdkit 304 표본. 표 파일은 같은 결과를 두 번씩 담는다 — 중복을 빼고 센 값이다 (처음 보고한 수치는 2배였다, 10-10 18:40 정정).
  J1 기준은 모두 충족했다 (판정 파싱 실패 0/1,129, 재판정 발생). 수집을 마친 17:24 에 중단했다 (rc 134 는 의도한 중단).
- **rdkit 예/아니오 행 채점 결함 (alpha 플러그인)**: property_type `bool` · `presence` 행(블렌드 172행)은 "Does this molecule …?" 를 묻고 정답은 1/0 인데,
  형식 예시가 `((42))` · `\boxed{42}` 라 모델은 `((No))` 로 답한다. 플러그인이 정수만 읽어 맞는 답도 0 점이었다 (예/아니오 표본 48개 중 맞는 답 24개).
  → 이 행만 yes · true → 1, no · false → 0 으로도 읽는다 (개수 행은 정수만). 단위 테스트 32건 PASS (기존 17 + 새 15, mcqa Gym venv `-o asyncio_mode=auto`).
  j1c 실제 출력 재채점: 예/아니오 정답 4 → 28 /48 (0 → 1 로 바뀐 24개 모두 이 행), 정수 행 11/256 그대로. rdkit 보상 4.9% → 12.8%.
  기록 보상과 옛 채점기 재채점은 303/304 일치 — 남은 1개는 추출이 정답과 같은데 기록 보상 0 (원인 미확인 — NeMo-RL 보상 감점으로 추정).
- **rdkit 은 대부분 생성 상한에 잘린다**: 304 표본 중 203 (67%) 이 reasoning 중 32K 에서 끝났다 (`status=incomplete`, 최종 답 없음). 끝난 표본도 출력 중앙값 23K 토큰.
  SMILES 고리 · 작용기 세기에서 생각이 끝나지 않는 모델 능력 문제다 — 채점 결함이 아니다. rdkit 그룹 대부분은 학습 신호가 없다 (블렌드 4.3%).
- **재판정 (결정 25) 의 실제 동작**: 0.5 보상 39개 중 **잘린 추출 복구는 13개뿐**이고, 26개는 첫 판정에 보낸 추출문이 이미 최종 답 전체였다 —
  같은 답을 응답 전체 문맥과 함께 다시 물어 judge 가 `[[A=B]]` 로 뒤집은 것이다. 표본에는 `A^{1}` 를 `A^{-1}` 로 읽어 준 경우 · 물리적으로 다른 설명을
  같다고 본 경우가 있다. judge 표본의 3.6% (블렌드 전체 약 1.1%) 가 반 점을 받는다. → **재판정 끔** (사용자 결정 2026-10-10, `RL_PLAN.md` 결정 27). 잘린 추출 복구 13개도 함께 잃는다. 10-10 17:32 기동한 본 런 `main` 은 켠 상태로 돈다.
  판별은 추출문 뒤에 닫는 문자만 남는지로 한다 (`analyze_gym_full_results.py` `rescue_kind`).
- **재판정 0.5 표본 수작업 감사 (2026-10-10 18:30, 사용자 요청 "재판정이 학습 잡음인지")**: j1c 의 0.5 보상 39개 (고유 37) 를 질문 · 정답 · 모델 답으로 직접 판정했다.
  기준선은 첫 판정이 맞다고 본 표본 40개 (고유). 결과 `results/alpha/teacher_stem/j1c/rescue_audit.json`. 판정자 1인 · 답 앞부분만 본 표본이 있어 근사치다.

  | 묶음 | 맞음 | 부분 (필수 요소 누락) | 오답 | 모호 | 형식 위반 |
  |---|---|---|---|---|---|
  | 재판정 0.5 전체 (39) | 16 (41%) | 15 (38%) | 2 | 4 | 2 (최종 답이 `**X**` 그대로) |
  | └ 판정 뒤집힘 (26) | 9 (35%) | 12 | 2 | 1 | 2 |
  | └ 잘린 추출 복구 (13) | 7 (54%) | 3 | 0 | 3 | 0 |
  | 기준선: 첫 판정 1 (판정 가능 32) | 28 (88%) | 4 | 0 | 0 | 0 |

  재판정 보상은 첫 판정 보상보다 정밀도가 크게 낮다 (분명히 맞음 41% 대 88%). 부분 정답에 반 점을 주고, 형식을 어긴 답에도 준다.

## judge 가 런 뒤 GPU 를 잡은 채 남는다 · J1 이 judge 판정 기록을 남기지 않았다 (2026-10-10 ✅ 실행기 순서 수정, j1c 로 보충)

- **증상 1**: j1b (rc 0, 16:19) 뒤 main1 GPU 6·7 에 75 GB 씩 남았다. 살아 있는 Ray 액터는 `LocalVLLMModelActor` · `EngineCoreActor` · `RayWorkerProc` 2개와
  배치 그룹 `judge_model_dp_rank_0` 뿐이었다 (NeMo-RL 런 액터 0). 실행기가 GPU 점유 검사 → Gym 서버 정리 순서라 다음 기동이 `BUSY_GPUS=2` 로 막힌다.
- **원인 1**: judge(`local_vllm_model`)의 vLLM 은 Gym judge 서버 프로세스(`python app.py`)가 띄운 Ray 액터다. Gym 서버는 별도 Ray 작업이라 드라이버가 끝나도 남고
  (2026-10-08 항목), judge 서버는 GPU 를 쥔 첫 Gym 서버다.
  → 실행기 순서를 바꿨다: 살아 있는 NeMo-RL 런 액터가 0 일 때만 (`runs/gpu_check.py` `NRL_RUN_ACTORS`) `gym_cleanup.py kill` → GPU 점유 검사 (60 초까지 재시도).
  런 액터가 있으면 정리하지 않고 기동을 멈춘다. 실측: sub1 Gym 서버 15개를 정리하자 GPU 가 10 초 안에 풀리고 남은 액터 0.
- **증상 2**: J1 판정 기준 중 "judge 판정 파싱 실패 ≤ 2%" · "재판정이 실제로 일어남" 을 j1b 산출물로 잴 수 없었다.
- **원인 2**: `env.should_log_nemo_gym_responses: true` (부모 레시피) 면 NeMo-RL 이 `train_data_step*.jsonl` 덤프를 건너뛰고, Gym 전체 결과는
  `logger.wandb.log_nemo_gym_full_result_tables` 를 켠 W&B 표로만 남는다 (`grpo.py` `_should_log_nemo_gym_responses` · `utils/logger.py:101`). 게이트는 W&B 를 끄고,
  부모의 `metric_allowlist` 도 표를 거른다. TensorBoard 의 agent 별 지표는 응답의 숫자 필드만 담는데, judge 응답에는 판정 · 재판정을 나타내는 숫자 필드가 없다.
  → 보충 스모크 j1c: `EXTRA_ENV="WANDB_MODE=offline"` (업로드 없음) + `logger.wandb.metric_allowlist=null` + `++logger.wandb.log_nemo_gym_full_result_tables=true`,
  smoke_v3 의 judge 행 128 + rdkit 행 64 (`teacher_stem_smoke_v3_j1c.jsonl`) 3스텝. `tools/analyze_gym_full_results.py` 가 오프라인 표에서
  파싱 실패 · 재판정 · rdkit 추출 실패를 센다. 본 런(820 스텝)은 표를 켜지 않는다 — 스텝당 1,024 개 전체 응답(평균 생성 17K 토큰)이라 크다.

## J1 첫 기동 실패 — ns_tools `sandbox_port` 타입 · judge Gym venv 의 torch CUDA 불일치 (2026-10-10 ✅ 둘 다 수정, j1b 재기동)

- **증상**: J1 (`teacher_stem` case `j1`, 14:47) 이 Gym 기동 중 rc=1 로 끝났다 — `RuntimeError: Process ns_tools finished unexpectedly!`
  (`ValidationError: NSToolsConfig sandbox_port — Input should be a valid string, input_value=6000`). 판정 분할 본체 수정은 의도대로 동작했다
  (롤아웃 노드 GPU 0~5 vLLM, 6·7 은 judge 몫으로 비움 — "2 GPUs per inference node left free for NeMo Gym GPU services").
- **원인 1**: 레시피가 `sandbox_port: 6000` (정수) 을 줬다. Gym 서버 설정은 기동 때 각 서버 venv 의 pydantic 클래스로 검증되므로 레시피 검사기가 보지 못했다.
  → `sandbox_port: "6000"` (`40a392280`). 검사기에 이 타입 검사를 넣었다.
- **원인 2 (같은 기동에서 발견)**: judge 서버(`local_vllm_model`)의 Gym venv 는 이번에 처음 만들어졌다. 그 서버의 `pyproject.toml` 은
  `[tool.uv.pip] torch-backend = "auto"` 라 uv 가 드라이버(535, CUDA 13.2 는 forward-compat)를 보고 `torch 2.11.0+cu129` 를 골랐는데
  `vllm 0.25.1` 은 cu13 빌드라 `import vllm` 이 `libcudart.so.13` 에서 죽는다 (NeMo-RL 본체도 같은 이유로 `pytorch-cu130` 인덱스를 고정한다, `pyproject.toml:113`).
  ns_tools 가 먼저 죽어 이 오류는 드러나지 않았지만, 설치가 중간에 끊긴 venv 는 `skip_venv_if_present` 로 다음 기동에 그대로 쓰인다 (Gym `cli/env.py:522-530` 이 경고하는 경우).
  → 끊긴 venv 를 `.venv.broken-cu129-20261010` 로 옮기고 Gym 과 같은 명령에 `UV_TORCH_BACKEND=cu130` 만 더해 다시 만들었다 (33 s, uv 캐시):

  ```bash
  cd 3rdparty/Gym-workspace/Gym/responses_api_models/local_vllm_model
  V=$NRL_ROOT/gates/gym_smoke/gym_venvs/responses_api_models/local_vllm_model/.venv
  export UV_CACHE_DIR=$NRL_ROOT/uv-cache UV_PYTHON_INSTALL_DIR=$NRL_ROOT/uv-python UV_TORCH_BACKEND=cu130 PATH=$NRL_ROOT/bin:$PATH
  uv venv --seed --allow-existing --python 3.13.14 $V && source $V/bin/activate \
    && uv pip install '-e .' 'ray[default]==2.56.1' 'openai==2.44.0'     # head_server_deps = 드라이버의 ray · openai 버전
  ```

  검증: `torch 2.11.0+cu130` · `vllm 0.25.1` · `from vllm.entrypoints.openai.api_server import run_server` OK.
- **기동 전 검증**: 병합한 ns_tools · judge_model 설정을 각 서버 venv 에서 Gym 설정 클래스(`NSToolsConfig` · `LocalVLLMModelConfig`)로 검증해 둘 다 OK,
  정수 포트 대조는 같은 오류를 재현했다. Gym 이 기동에서 만드는 서버 venv 는 새 서버를 처음 쓸 때 위처럼 따로 만들어 확인한다.

## Science-v1 정답이 자기 `output_regex` 에서 잘린다 — STEM 블렌드 636행 (2026-10-10 ✅ 교체 + 재판정 켬)

- **발견**: P0-4 judge 검증 중. Science-v1 행은 최종 답을 감쌀 형식(`(Answer: …)` · `((…))` · `[Answer: …]` 등)을 지시하고, Gym 은 그 행의
  `template_metadata.output_regex` 로 마지막 일치를 꺼내 judge 에게 보낸다 (`equivalence_llm_judge/app.py:144-200`). 정규식이 비탐욕(`.+?`)이라
  **첫 닫는 문자에서 멈춘다.** 정답 자체에 그 문자가 있으면 정답과 같은 표기로 쓴 맞는 답도 조각만 judge 에게 간다.

  | 정답 | judge 가 받는 것 |
  |---|---|
  | `…chain rule to \(U(N,T,V)=U(z,T,V)\).` | `…chain rule to \(U(N,T,V` |
  | `\([PXP,P]=i\hbar P^{2}=…\)` (`Answer is […]`) | `\([PXP,P` |
  | `\(U = \dfrac{m g h}{2}\) (because … at \(h/2\)).` (`((…))`) | `… at \(h/2\` |

- **실제 학습 경로에서 그대로다**: Gym 코드의 구제 장치 두 개(정답 120자 초과면 추출 생략 · 추출 판정 실패 시 응답 전체로 재판정 0.5점)는
  파이썬 기본값으로는 켜져 있지만 `equivalence_llm_judge.yaml` 이 끈다 (`null` · `false` · `0.0`). ns_tools 행도 행 전체를 equivalence judge 로
  넘겨 같은 경로를 탄다 (`ns_tools/app.py:437-441`). 정답을 요청 형식 그대로 넣은 탐침 60개 중 보상 14개 (`$NRL_ROOT/gates/p04_judge/truncation_probe.json`, 보상 0/1 만 — 재판정 꺼짐과 일치).
- **규모** (`teacher_stem_v1.jsonl` Science-v1 7,909행, `measure_judge_p04.gold_regex_defect`): 잘림 **636행** (python 도구 421 · 주관식 215).
  처음 집계한 662행 중 26행은 결함이 아니었다 — 중괄호가 너무 깊어 `\boxed{}` 정규식이 매치되지 않는 25행은 Gym 이 응답 전체를 판정하고, `400 J**.` 1행은 `400 J` 로 같다.

  | 원인 | 행 | 모델이 피할 수 있나 |
  |---|---|---|
  | `(Answer: …)` 안의 문장 괄호 | 257 | 문장을 바꾸면 |
  | `(Answer: …)` 안의 LaTeX `\)` | 157 | `$…$` 로 쓰면 |
  | `(Answer: …)` 안의 함수 괄호 `f(x)` · `U(N,T,V)` | 52 | 거의 불가 |
  | `[Answer: …]` · `Answer is […]` 안의 `]` (`\]` · `[A,B]` · 농도 `[A]`) | 77 | 거의 불가 |
  | `((…))` 안의 `))` | 63 | 일부 |
  | `**…**` 안의 굵은 글씨 · `\|\|…\|\|` | 29 · 1 | 쉬움 |

- **영향**: 0점이 확정되지는 않는다 — 모델이 닫는 문자를 피하면 보상을 받는다. 그래서 이 행의 기울기는 정답성보다 "괄호·LaTeX 를 피하라"를 가르치고,
  피할 수 없는 행(`f(x)` · `[A,B]`)은 늘 0 에 가깝다. 정답에 닫는 문자가 없는 행도 모델 답에 들어가면 똑같이 잘린다 — 데이터 교체로는 막지 못한다.
  NVIDIA 도 이 행들을 Ultra 에 썼지만 (`used_in: ultra_v3`) 그때의 judge 설정은 모른다.
- **수정 (사용자 결정 2026-10-10, `RL_PLAN.md` 결정 25)**:
  1. 636행을 같은 agent · 분야 · 정답 형식의 결함 없는 Science-v1 행으로 교체 — `teacher_stem_v2.jsonl` · `v2_val` (6행) · `smoke_v3` (16행),
     자리 · 행 수 · 비율 그대로 (`tools/fix_stem_sci_regex.py`). 교체 후보 2,609개는 벤치 12,612문항과 오염 검사 0건.
     독립 검증: 바뀐 줄 = 잘림 줄 (636/6/16) · 교체 행 = 원천 행 · agent · 분야 · verifier 분포 동일 · 세 파일 간 겹침 0 · D1 OK · 최대 프롬프트 9,352 토큰.
  2. 레시피 `teacher_stem_alpha.yaml` 에 `check_full_generation_on_fail: true` · `reward_if_full_generation_succeeds: 0.5` — 모델 답 쪽 잘림을 살린다.
     정규식이 없는 행(Ultra reasoning · ExamQA)에는 걸리지 않는다. judge 호출은 실패한 Science-v1 응답마다 1회 늘어난다 (스텝당 최대 약 150회).
     위험: 여러 답을 늘어놓은 응답이 전체 판정에서 통과할 수 있다 (상한 0.5) — J1 · 첫 스텝들에서 재판정 횟수와 0.5 보상 표본을 본다.
- **검사기**: `check_alpha_recipe.py` 가 데이터를 받으면 equivalence judge 로 채점되는 행의 정답 잘림을 ERROR, 재판정 꺼짐을 WARN 으로 낸다 (v1 ERROR 636·6 · v2 ERROR 0 · 재판정 끈 대조 WARN).

## bf16 가중치에 갱신이 반영되지 않는다 — lr 1e-6 도구 사용 teacher 의 보상 평탄 (2026-10-08 ✅ lr 3e-6 재시작, 10-09 반영 계수 0.33 확인)

- **증상**: `teacher_tool_alpha.yaml` (Muon lr 1e-6) 런의 보상이 42스텝 동안 평탄했다. 1~10스텝 평균 0.481, 31~40스텝 평균도 0.481 이다
  (추세 10스텝당 +0.0008 ± 0.0057). 환경 3종이 모두 평탄했다. 정합 지표는 정상이었다 (gen KL 0.00134 · 마스킹 0 · 잘림 0.5% 이하).
- **학습 경로 점검 (버그 없음)**:

  | 단계 | 결과 |
  |---|---|
  | 보상 채점 | step 50 리플레이 버퍼 120그룹 1,920샘플을 데이터셋 `expected_action` 과 대조. 보상 1 인 886개는 모두 도구 이름이 맞다. 인자까지 같은데 0 인 경우는 0건 (인자는 설계상 단어 Jaccard 비교) |
  | 학습 신호 | 그룹 192개 중 성공·실패가 섞인 그룹 153개 |
  | advantage | 최댓값 3.6148 = 16개 중 2개 성공 그룹의 LOO 정규화 해석값 |
  | 학습 토큰 | 스텝당 469K = 생성 토큰 전부 (프롬프트 제외) |
  | 옵티마이저 | 학습 파라미터 전부 상태 있음. 라우터는 동결돼 시작점과 비트 단위로 같다 |
  | fp32 갱신 | step 40→50 이 Muon 최대치(0.2 × lr × 스텝)의 62~94%. 방향은 momentum 반대 (하강) |
  | 저장·refit | bf16 = round(fp32 master) 100%. refit 매 스텝 28~30 s |

- **원인**: 시작점(HF iter2862) 대비 50스텝 누적 갱신 중 bf16 가중치에 반영된 것은 약 12% 다 (투영 계수 0.10~0.13). 바뀐 원소는 4.4~4.9% 다.
  시작 가중치가 bf16 HF 체크포인트라 fp32 master 가 bf16 격자점에서 출발한다. 누적 갱신이 반올림 경계를 넘어야 bf16 값이 바뀐다.
  경계는 가중치 0.025 근처에서 ulp/2 = 6e-5 이고, 스텝당 Muon 갱신 RMS 는 0.2 × lr = 2e-7 이다. 바뀐 원소 비율(4.42%)과 경계를 넘은 원소 비율(4.40%)이 같다.
  학습 forward 와 vLLM refit 이 모두 bf16 을 쓰므로 정책이 거의 그대로였다. 2스텝 묵은 롤아웃의 gen KL 이 같은 가중치 게이트값과 같다 (2스텝 드리프트 < 1e-4).
  norm·A_log 처럼 값이 큰 파라미터는 경계까지 수천~수만 스텝이 걸린다.
- **대응**: lr 3e-6 (사용자 결정 2026-10-08, `RL_PLAN.md` 결정 22, `b3115c226`). 같은 방향 1차 추정으로 누적 갱신이 3배면 반영 계수 0.34, 같은 스텝 실제 반영량은 약 8.5배다 (8배면 0.71).
  1e-6 런은 55스텝을 마치고 56스텝 중에 멈췄다 (23:23). 체크포인트는 `results/alpha/teacher_tool/ckpt_lr1e6_main/` (step_40·50) 에 보관했다가 10-09 검증 뒤 지웠다.
- **검증 도구**: `tools/measure_bf16_delivery.py <ckpt>/step_N <hfmodel>` (CPU 11 s). 1e-6 런 step_50 에서 위 수치를 재현한다.
- **검증 결과 (2026-10-09, lr 3e-6 런 `lr3e6`)**: 예측대로다. 보상 이득은 작다.

  | 지표 (step 50) | lr 1e-6 | lr 3e-6 |
  |---|---|---|
  | 반영 계수 (텐서 5개) | 0.10~0.13 | 0.30~0.36 (예측 0.34) |
  | 바뀐 원소 | 4.4~4.9% | 9.5~14.3% |
  | 라우터 (동결 대조) | 0 | 0 |
  | 짝지은 보상 차이 11~50스텝 (3e-6 − 1e-6) | — | +0.0088 ± 0.0028 (40스텝 중 26 양수) |
  | gen KL · 마스킹 · entropy (31~50스텝) | 0.00134 · 0 · 0.763 | 0.00135 · 0 · 0.764 |

  두 런은 seed 42 로 50스텝 모두 프롬프트가 같아 스텝끼리 짝지었다. 같은 스텝 실제 반영량은 평균 기준 약 8.7배다 (예측 8.5배).
  보상 차이는 10스텝 구간별로 +0.012 · +0.001 · +0.012 · +0.010 이고 커지는 추세는 아직 없다 (10스텝당 +0.002 ± 0.002).
- **정정**: 처음에는 "Muon 1e-6 = AdamW 1e-6 이라 Ultra teacher(Adam 2.5~3e-6)의 1/2.5~1/3" 이라고 보고했다. 그런데 이 런에서 Adam 으로 학습되는 output layer 는
  스텝당 0.10 × lr, Muon 행렬은 0.2 × lr 움직였다. Ultra 와의 갱신 크기 차이는 1.5배 정도일 수 있다. lr 근거는 Ultra 비교가 아니라 반영률 실측이다.
- **교훈**: 보상이 평탄하면 정합 지표만 보지 말고 체크포인트로 반영률을 잰다. RL lr 은 bf16 해상도 대비 갱신 크기로도 본다.

## STEM 블렌드의 rdkit 행은 Gym 0.6.0 에 서버가 없고, Science-v1 도구 행은 ns_tools 에 채점기가 등록돼 있지 않다 (2026-10-08 ✅ 플러그인·레시피·검사기)

- **발견 경위**: STEM 레시피를 쓰면서 블렌드의 agent 마다 Gym 설정을 찾았다. `rdkit_chemistry_agent` (2,262행, 4.3%)는 Gym v0.6.0 `3045a793` 어디에도 없다 — Ultra 블렌드에만 행이 있다.
  첫 RLVR 런은 이 환경을 쓰지 않아 드러나지 않았다. STEM judge 부분을 만들면서 Science-v1 python 도구 행의 `verifier_type: equivalence_llm_judge` 도 찾았다.
  `ns_tools` 는 `verifiers` 에 없는 채점기를 `ValueError("Unknown verifier")` 로 거부한다 (`resources_servers/ns_tools/app.py` verify). 기본 설정에는 math_with_judge 만 있다.
  둘 다 런을 띄워 롤아웃이 돌아야 드러나는 실패다.
- **대응**: rdkit 채점기를 alpha Gym 플러그인 `gym_plugins/resources_servers/rdkit_chemistry` 로 만들었다. 정답이 모두 정수라 지시한 형식(`\boxed{N}` 또는 `((N))`)의 마지막 값과 일치하면 1 이다 (테스트 17건).
  STEM 레시피가 `ns_tools.verifiers.equivalence_llm_judge` 를 등록한다. `check_alpha_recipe.py` 가 데이터 행의 agent·verifier_type 을 Gym 설정과 대조한다 — 음성 대조에서 rdkit 설정을 빼면 2,298행, verifier 를 빼면 5,301행을 ERROR 로 잡는다.
- **교훈**: 블렌드를 바꾸면 레시피 검사기를 데이터 경로와 함께 다시 돌린다. Ultra 블렌드 행이라고 우리 Gym 버전에 서버가 있는 것은 아니다.

## Code·Math 1단계 `s1r5` 가 step 205 부터 롤아웃·학습 정책 불일치로 무너졌다 (2026-10-16, 사후 분석 — CPU 재현으로 기전 확인)

- **증상** (STATUS 2026-10-16): DAPO 길이 감점으로 잘림 0.74 → 0 · 생성 26.7K → 6K · 보상 0.12 → 0.26 (step 151~175) 까지 간 뒤, 길이 재증가 (173) → think tag 형식 오류 (185~) → `policy_kl_error` 0.002 → 0.05~0.08 · threshold 2 마스킹 시퀀스 최대 39% (205~) → grad norm 0.3~0.6 → 보상 0.17 로 하락. 감점은 step 115 부터 어떤 샘플에도 걸리지 않았다.
- **확인된 사실 (CPU 분석, 보존 ckpt `ckpt_keep/step_200~280` · TensorBoard)**:
  - KL 두 방향의 비 (policy/gen) 가 1.0 → 2.1 (176) → 5 (178) → 6~25 (205~). 마스킹 수는 policy KL (r 0.96) · mult error (0.95) · TIS 범위 밖 (0.67) 과 함께 움직이고 think tag 오류와는 0.18. 시퀀스 하나에 Δlogp ≈ −10 인 토큰 하나면 마스킹되므로, "시퀀스 39% 에 학습기가 거의 불가능하다고 보는 토큰이 섞였다" 가 정확한 서술이다.
  - 리플레이 버퍼의 생성 샘플(step 200·230·282 각 약 2,200개)은 정상: 반복 루프 0.5% 이하, `routed_experts` 기록 100% 온전, 샘플 토큰 평균 logprob −0.8 → −1.0, logprob < −5 토큰 2% → 3%. `</think>` 중복은 붕괴 전 step 200 (8.6%) 이 가장 많았고 마스킹과 무관.
  - 5 step 간 파라미터 이동량 (‖Δθ‖/‖θ‖, DCP 직접 로드)은 200→205 6.1e-4 ~ 275→280 6.6e-4 로 붕괴 전·중·후가 같다. 라우터 0 (동결 확인), norm ≈ 0. **정정 (2026-10-16 12:40)**: 재학습 `teacher_codemath2` 의 정상 구간 10→15 (lr 은 이미 5e-6) 는 9.4e-5 로 **6.5배 작다**. Muon 은 모멘텀을 직교화하므로 걸음 크기가 기울기 크기에는 둔감하지만 유효 랭크에는 비례한다 — 잡음이 많은 고랭크 기울기일수록 걸음이 커진다. 따라서 s1r5 는 step 200 에 이미 초반의 6.5배로 걷고 있었고, 신호가 희소·손상될수록 걸음이 커지는 되먹임이 성립한다. 재학습 추적 결과 (2026-10-18): 25→30 2.3e-4 → 100→105 4.8e-4 → 200→205 6.0e-4 → 275→280 6.3e-4 로 **정상 런도 같은 값에 도달** — 걸음 크기는 학습 진행에 따른 자연 증가이며 붕괴를 가르는 지표가 아니다 (H2 최종 기각). 붕괴를 가르는 것은 같은 걸음에서 토큰 확률이 얼마나 변하는가(민감도)다.
  - 기각: 투기 디코딩(꺼짐) · 2 에폭 재방문(142~170 깨끗) · 길이 자체(25K 구간 KL 0.0019) · refit 실패(57회 정상, 점진 상승) · 형식 오류(증상) · 생성기 퇴행 · 라우팅 기록 누락 · 파라미터 이동 급증.
- **남은 가설**: 정책이 "같은 크기의 작은 가중치 변화(또는 엔진 수치 차이)가 샘플 토큰의 logprob 를 크게 바꾸는" 민감한 영역으로 들어갔다 — GDN 재귀 상태가 긴 시퀀스에서 작은 파라미터 차이를 증폭하는 경로가 유력 (10-06 의 "오차가 생성 위치를 따라 커짐" 과 같은 기전, 이번엔 수치가 아니라 민감도). 감점이 만든 급격한 길이 붕괴(entropy 최저 0.75, 발생 시점 일치)가 그 영역으로 밀어넣은 것으로 본다 (사용자 가설의 정밀화).
- **결정적 실험 (GPU 필요)**: ckpt_230 의 on-policy R1 게이트 (같은 가중치로 샘플링 → 학습기 logprob, 8-GPU 1노드 약 1시간). KL ≈ 0.002 면 stale 생성기 × 민감도, 0.07 수준이면 엔진 × 민감도. 덤프 위치별 KL 로 "뒤쪽 토큰에서 커지는지" 를 함께 본다. 실행 시점은 재학습(`teacher_codemath2`) GPU 창에 맞춰 결정.
- **CPU 재현 (2026-10-16 10:00~12:20, GPU 없이)**: 보존 ckpt 200·225·230 을 `tools/export_rl_hf.sh` 로 HF 반출(CPU, 3~8분) → 시스템 python(transformers 4.57, GDN torch 대체 경로, `use_cache=False` 금지·스레드 8 이하가 아니면 segfault) 으로 리플레이 버퍼 샘플을 채점. 스크립트 `$JOBTMP/pm/score_seqs_hf3.py`(자체 라우팅 / 기록 `routed_experts` 를 HF MoE 에 강제 재생) · `hf4.py`(라우팅 불일치율). 결과:
  - **step 200**: on-policy(v200) 극단 토큰 0. stale(v197, 3 step 전) 은 자체 라우팅에서 |Δ|>5 토큰 3개·마스킹 1/6 이지만 **기록 라우팅을 재생하면 전부 사라짐** (k3 3.58 → 0.0018). 즉 그 시점의 불일치는 전부 MoE 라우팅 뒤집힘이고 R3 가 막아 주던 것이다.
  - **step 230**: on-policy(v230) 는 재생으로 깨끗 (0/6). stale(v227) 은 **재생해도 3/6 마스킹** (k3_pol 50 · 1.9 · 19). 같은 v227 샘플을 **2 step 전 가중치 HF@225** 로 채점해도 재생 상태로 3/6 마스킹 (k3_pol 2565 · 10.5 · 1.1) — 시간 양방향으로 2~3 step 떨어진 가중치가 특정 토큰 확률을 10 nats 이상 다르게 본다.
  - 극단 토큰 위치는 어긋난 전문가 수가 평균의 4~15배인 **라우팅 연쇄 뒤집힘 지점**이지만, 전체 뒤집힘 빈도(stale 기준 토큰당 5~7)는 200 과 230 이 같다. 위치별 |Δ| 는 평탄 (GDN 누적 오차 아님).
  - 극단 토큰은 특수 토큰이 아니라 보통 텍스트(`?`·`But`·` points`·` Takah`[ashi]·`holders`)이고, 흩어진 희귀 토큰(` %#`·`aneous`·`سر`, 생성 p 1e-3~1e-5) 뒤의 문맥에 자주 있다.
- **기전 (확인된 범위)**: 파라미터 걸음은 일정한데 **토큰 확률 공간에서의 step 당 변화가 특정 문맥에서 10 nats 이상**으로 커졌다 (정책의 출력 민감도 상승). async 는 나이 1~2 의 샘플을 R3 재생으로 학습하므로, 그 샘플의 희소 토큰이 현재 가중치에서 "불가능" 이 되어 threshold 2 마스킹(최대 39%)·TIS 절단·유효 토큰 감소(grad norm 10배)로 이어지고, 손상된 신호가 드리프트를 더 키웠다. 민감도 상승의 유발 조건은 감점이 만든 급격한 분포 이동(길이 26K → 6K, entropy 최저 0.75 뒤 0.9~1.0 으로 반등, 희귀 토큰 비율 2% → 3.4%) 이다. 생성 중 refit(`in_flight_weight_updates: true`, 상태 재계산 없음)이 섞인 상태를 만들 가능성은 남지만, 궤적이 refit 을 넘는 비율(p95 궤적 시간/step 시간)은 71~100 구간 2.1배·붕괴 구간 1.3~1.5배·STEM 1.2~1.7배라 그것만으로는 설명되지 않는다 (GPU 창에서 `in_flight_weight_updates` 끄고 마스킹 비교로 확정 가능).
- **교훈**: (1) 감시 지표에 `policy_kl_error`·마스킹 수·KL 두 방향의 비를 넣는다 — gen KL 만 보면 0.004 로 평온해 보인다 (cron 에 반영, 2026-10-16). (2) 보상 설계를 런 중간에 바꾸지 않는다; 바꾸면 세기는 논문 상대값(0/1 보상이면 감점 0.5)으로, 처음부터. (3) 드리프트가 빨라지는 구간(entropy 급변·길이 급변)에서는 staleness 를 1 로 줄이거나 lr·KL 앵커로 속도를 제한한다 — Muon 에서는 grad norm 이 아니라 ‖Δθ‖(체크포인트 쌍으로 측정)가 걸음의 지표이고, 잡음 기울기는 걸음을 키운다. (4) 체크포인트 keep-2 는 최고 구간을 지운다 — 25 step 마다 하드링크 보존. (5) 학습 중 샘플(train_data 덤프 또는 샘플별 jsonl)이 없으면 사후 분석이 리플레이 버퍼(18 GB/ckpt)에만 의존한다. (6) in-loss 마스킹은 증상을 숨긴다 — 마스킹이 1% 를 넘으면 멈추고 본다.

## upstream `grpo.reward_shaping`(DAPO 길이 감점)은 async GRPO 경로에서 적용되지 않는다 (2026-10-08 조사 · 2026-10-15 ✅ 본체 패치, 결정 36)

- **사실**: `reward_shaping` 은 동기 `grpo_train` 에서만 호출된다 (`nemo_rl/algorithms/grpo.py:3131`). alpha 레시피가 쓰는 `async_grpo_train` 에는 호출이 없다. 원보상 기준 표준편차(`std_rewards`)도 동기 경로에서만 넘어간다 (3149행). 레시피에서 켜도 조용히 아무 일도 없다.
- **대응**: 길이 보상은 그룹이 모이는 `nemo_rl/experience/rollouts.py` 의 `_postprocess_single_nemo_gym_group` (effort shaping·페널티 계산 자리)에 넣는다. 설계·오프라인 검증은 `RL_DATA.md` §5.5 (Code·Math 차례에 구현).
- **대응 (2026-10-15, `RL_PLAN.md` 결정 36)**: 그룹 후처리(`rollouts.py`)가 아니라 **async 학습 스텝**에 넣었다 — `nemo_rl/algorithms/grpo.py` `apply_async_reward_shaping` 이 버퍼에서 뽑은 배치에 sync 루프와 같은 위치(롤아웃 지표 뒤 · advantage 앞)에서 upstream `apply_reward_shaping` 을 호출한다. 학습 시점 적용이라 체크포인트에서 복원한 버퍼 그룹에도 적용되고 버퍼 키가 늘지 않는다. 응답 길이는 턴별 최대 assistant 토큰 (`hit_max_tokens` 와 같은 호출 단위; data-plane 슬라이스의 `response_token_lengths` 가 있으면 그것). 브랜치 `alpha/async-dapo` (`bb0835bd2`), 테스트 `tests/unit/algorithms/test_async_reward_shaping.py`. **확인법**: 켜진 레시피에서 `train/reward_shaping/penalized_rate` 가 잘림률과 비슷해야 한다 — 0 이면 미적용 (TensorBoard · W&B allowlist 포함).
- **함정 (2026-10-15 `s1r4` 기동 실패)**: 두 진입 스크립트(`examples/run_grpo.py` · `examples/nemo_gym/run_grpo_nemo_gym.py`)가 async 에서 `reward_shaping.enabled` 를 `NotImplementedError` 로 막는다 — 학습 전 rc 1. `nemo_rl/` 만 grep 해서 놓쳤다. 감점 가드만 제거 (`2b069dbb6`), 동적 샘플링·`reward_scaling` 가드는 async 미구현이라 유지. 본체 기능을 켤 때는 `examples/` 진입 스크립트의 가드도 같이 본다.

## alpha vLLM 을 `--data-parallel-size 8` 로 띄우면 NCCL 초기화에서 `free(): double free detected in tcache 2` 로 죽는다 (2026-10-08, Pai 문서에 이미 있던 함정)

- **증상**: P0-3 사전 측정용으로 `vllm serve <hfmodel> --data-parallel-size 8` 을 main1·sub1 에 띄웠다. 두 노드 모두 `vLLM is using nccl==2.28.9` 직후 워커 8개가 `double free` 로 죽었다. 깨끗한 환경(sub1, Ray actor)에서도 같았으니 환경 누출(#18)이 원인이 아니다.
- **원인**: vLLM DP 모드 문제다. Pai 는 이미 우회하고 문서화했다 (Pai `SFT_BENCHMARKS.md` §2.6 "단일GPU 서버 N개 + 라운드로빈 프록시 (vLLM DP munmap 우회)").
- **대응**: alpha 단독 서빙은 Pai `eval_sft/serve_fleet.sh`(GPU 당 `serve_alpha.sh` 1개 + `lb_proxy.py`)로 띄우고 `stop_fleet.sh` 로 내린다. ssh 가 막힌 sub1 은 `$NRL_ROOT/p0_measure/ray_serve_alpha.py`(Ray detached actor 가 `serve_alpha.sh` N개를 띄움)로 띄운다.
- **교훈**: 서빙·벤치 경로는 Pai 문서가 정본이다. 새 서빙 스크립트를 만들기 전에 Pai `SFT_BENCHMARKS.md` §2.5~2.6 과 `gym/README.md` 를 먼저 읽는다.

## NeMo-Skills sandbox 이미지 빌드가 `vedas==0.0.1` 을 찾지 못해 실패한다 (2026-10-08 ✅ 고정 줄 제거)

- **증상**: gpu06 `alpha-eval` 에서 NeMo-Skills `da85a881` 의 `dockerfiles/Dockerfile.sandbox` 빌드가 `uv pip install -r requirements/sandbox.lock` 단계에서 `vedas was not found in the package registry` 로 멈췄다.
- **원인**: `requirements/stem.txt` 가 넣은 `vedas==0.0.1` 이 PyPI 에서 사라졌다.
- **대응**: 체크아웃(`/opt/nemo-skills/Skills`)의 `sandbox.lock` 에서 그 줄만 지웠다 (원본 `sandbox.lock.orig`). SciCode 테스트 데이터(Google Drive) 내려받기는 `--build-arg GITHUB_CI=1` 로 건너뛴다. 이미지 `alpha-nemo-skills-sandbox:da85a88` 23.2 GB, 내보내기만 8.5분.
- 검증: Python 실행 p50 0.14 s · 동시 32 에서 초당 117건 (main1·sub1 모두), Lean 4 맞는 증명 통과·틀린 증명 실패 (`RL_DATA.md` §5.9).

## vLLM ngram 투기 디코딩(CPU 제안기)이 엔진 기동에서 죽는다 — `Numba needs NumPy 2.4 or less. Got NumPy 2.5.` (2026-10-08, `ngram_gpu` 로 우회)

**발견 경위**: rollout 속도 시험 T2 에서 `speculative_config.method=ngram` 으로 띄우자 vLLM EngineCore 가 기동 중 ImportError 로 죽었다 (rc 134).
**원인**: vLLM 0.25.1 의 CPU 용 ngram 제안기(`v1/spec_decode/ngram_proposer.py`)는 numba 를 import 한다. vLLM 워커 venv 의 NumPy 2.5 를 설치된 numba 가 지원하지 않는다.
**대응**: GPU 용 제안기 `method=ngram_gpu` (`ngram_proposer_gpu.py`, torch 만 쓴다)로 시험한다 (T2b). venv 의 NumPy·numba 를 바꾸는 것은 환경 재구축(E 게이트)이라 하지 않았다.

## Gym 경로 `truncation_rate` 가 생성 상한(max_new_tokens) 잘림을 세지 않는다 (2026-10-08 ✅ `78568c330` 병합)

**발견 경위**: E0(G8, 생성 상한 16K)에서 128개 중 68개가 정확히 16,384 토큰에서 끝났는데 `train/truncation_rate` 는 0 이었다.
E1a(상한 64K)도 덤프의 생성 구간 길이로 센 잘림 12.6~16.5% 가 지표로는 0 이었다.

**원인**: NeMo-Gym 후처리(`_postprocess_single_nemo_gym_group`)는 prompt+response 가 `max_model_len` 을 정확히 채운 샘플만 잘림으로 셌다.
모델 호출이 요청별 출력 상한(`max_output_tokens = min(행 상한, max_new_tokens)`, `_prepare_nemo_gym_rows`)에서 멈춘 경우는 정상 종료로 셌다.
비-Gym 경로는 생성의 `truncated` 플래그로 이 경우를 센다. upstream main 8bf6bd4ba 도 같은 코드다.

**영향**: 지표만 틀렸다 (`truncation_rate`·`natural_termination_rate`). 배치의 `truncated` 는 overlong filtering 에만 쓰이고 alpha 레시피는 끈다.
64K 상한(D7)의 잘림 비율을 본 런 내내 볼 수 없었을 것이다. 덧붙여 async 경로의 그룹별 지표 합산은 `max_` 로 시작하는 키를 그룹 간 최댓값으로 합친다
(`grpo.py` `aggregate_rollout_metrics`) — `max_gen_tokens_per_turn/mean` 같은 값은 잘림 비율로 쓸 수 없다.

**수정**: assistant 메시지 길이가 그 행의 `max_output_tokens` 이상이면 잘림으로 센다. NeMo-Gym 은 `generation_token_ids` 를 모델 호출의 마지막 출력 항목에만 붙이므로
assistant 메시지 하나가 호출 한 번의 생성 전체다 (`nemo_rl/environments/nemo_gym.py`). 브랜치 `alpha/gym-trunc-metric` (`6a6bad156`) → `78568c330` 병합.

**검증**: 새 테스트 `test_postprocess_nemo_gym_group_counts_output_cap_truncation` 이 수정 전 코드에서 `[False, False, True]` 로 실패하고 수정 뒤 통과한다.
관련 테스트 4개 통과 (`pytest --noconftest` — 운영 Ray 클러스터에 붙지 않는다). 본 런 구간 A 의 덤프 계수와 지표를 대조한다.

## 첫 RLVR 런이 7스텝에서 6.5시간 멈췄다 — Gym HTTP 경로의 vLLM 요청이 엔진 하나에서 끝나지 않았다 (2026-10-08, 원인 후보 이식)

**발견 경위**: 구간 1 의 6스텝이 01:13 에 끝난 뒤, 7스텝은 배치(target 6)를 기다렸다. 01:49 이후 replay buffer 가 87 그룹에서 멈췄다
(target 6 은 57/64, target 7 은 30/64). 07:40 사용자 결정으로 런을 멈출 때까지 진척이 없었다. 오류·경고 로그는 없었다.

**관찰** (런을 멈추기 전에 남긴 것):
- sub1 vLLM GPU 8장 중 GPU 6 만 사용률 100% 이고 7장은 0% 였다.
- sub1 의 vLLM HTTP 포트 8개 중 한 곳(38837)에만 ESTABLISHED 연결 67개가 있었다. 나머지 7개는 0 이었다 (`/proc/net/tcp`).
- Gym simple_agent 는 롤아웃마다 새 세션으로 시작하고, 엔드포인트는 세션 해시로 고른다 (`vllm_model/app.py:1500-1510`). 그래서 요청 67개가 한 엔진에 몰린 것은 해시 쏠림으로 설명되지 않는다.
- 진단 수단이 막혀 있었다. py-spy 는 ptrace 제한(CUDA IPC 와 같은 컨테이너 제약)으로 "Permission Denied" 였고, NeMo-RL 의 vLLM HTTP 서버는 `/metrics` 를 내놓지 않는다 (404).

**원인 후보**: upstream 04ed9a946 (#3968) 이 고친 결함이다. Gym 경로의 HTTP 서버는 별도 스레드의 이벤트 루프에서 `AsyncLLM.generate` 를 불렀다.
AsyncLLM 의 요청 상태는 액터의 엔진 루프에 속해, 교차 스레드 사용에서는 출력을 기다리는 요청이 깨어나지 못할 수 있다.
같은 계열의 1d33dba75 (#3844) 는 문맥 초과 요청에 응답 없이 끝나지 않고 HTTP 400 을 돌려준다. 확정은 하지 못했다 — 스택을 볼 수 없었다.
5스텝의 대기 1,767 s 도 같은 현상의 약한 형태일 수 있다.

**대응**: 두 커밋을 이식했다 (`4768471eb`·`7bc9c8d93`, 단위 테스트 8 passed). v2 런에서는 스텝이 일정 시간 이상 끝나지 않으면 경보를 띄워 다시 본다.
생성 상한 64K 와 chunked prefill(KV 약 2배)도 엔진 하나에 긴 요청이 몰릴 때의 압박을 줄인다.

**교훈**: async 런에는 "진척 없음" 감시가 필요하다. 이 런은 오류 없이 6.5시간을 기다렸다.

## NeMo-Gym 서버 프로세스가 런이 끝나도 남는다 — 132개 누적 (2026-10-08 ✅ 기동 전 정리)

**발견 경위**: 구간 1 을 멈춘 뒤 `ray list jobs` 에 RUNNING 작업이 많이 남았다. main1 에 `python app.py` 44개(약 13시간 전 G5 런), sub1 에 88개(G7 런 등)가 살아 있었다.
Gym 서버는 각자 별도 Ray 작업(드라이버)으로 떠서 NeMo-RL 드라이버가 끝나도 함께 정리되지 않는다. 런마다 22~44개씩 쌓여 메모리·포트·CPU 를 쓴다.

**대응**: `$NRL_ROOT/runs/rlvr1_alpha_v2/gym_cleanup.py` 가 두 노드의 남은 Gym 서버를 찾아 종료한다. v2 실행기(`launch.sh`)는 GPU 점유 검사(런 없음)를 통과한 뒤 기동 전에 이걸 부른다.
이번 정리 뒤 Ray 작업 231개가 모두 SUCCEEDED 가 됐다.

## 첫 RLVR 런이 레시피와 다른 보상으로 학습됐다 — async 경로의 upstream 결함 4건 (2026-10-08 ✅ 이식, 브랜치 `alpha/perf-fixes`)

**발견 경위**: 첫 RLVR 런 구간 1 의 병목 판정 뒤 최신 NeMo-RL(upstream main 8bf6bd4ba, 분기점 이후 268커밋)을 검토하다 찾았다.
네 건 모두 upstream 에 수정이 있고, 우리 브랜치(분기점 2026-08-12)에는 없었다. 코드로 직접 확인했다.

| # | 결함 | 영향 | upstream 수정 |
|---|---|---|---|
| 1 | async 수집기가 `run_async_nemo_gym_rollout` 에 `effort_config` 를 넘기지 않는다 (`trajectory_collector.py:818-840`, 동기 경로 `grpo.py:2957` 은 넘김) | 레시피의 `env.nemo_gym.effort_levels`(Ultra 계수)가 조용히 무시된다. 마커 프롬프트(블렌드 3,429행)의 긴 응답에 감점이 없었다 — 4스텝 "efficient" math 44,735 토큰 무감점 | 4c6e5c84e (#3885) |
| 2 | async 루프가 `max_num_steps` 만 본다 (`grpo.py:4499`) | 1에폭 런은 데이터가 떨어질 때 "dataloader exhausted" RuntimeError 로 끝나 마지막 저장 뒤 최대 9스텝을 잃는다. `training_info.json` 의 `total_steps` 도 0 이었다 | f49e41dda (#3248) |
| 3 | LOO + `normalize_rewards` 가 std 를 E[r²]−E[r]² 로 구한다 (`algorithms/utils.py:189-195`) | 그룹 보상이 모두 같은 비이진 값이면 fp32 잡음 std 로 나눠 advantage 가 튄다. #1 을 고치면 보상이 비이진이 된다 | 8535caefc (#4158) |
| 4 | in-flight refit 중 vLLM 을 멈추지 않는다 | upstream 은 refit 과 forward 의 경쟁 조건이 NaN logprob·KL 튐을 낸다고 설명한다. 우리 KL 은 정상이었다 (미검증) | 4f653c4dc (#3839) |

**대응**: 네 건을 `alpha/perf-fixes` 에 이식했다 (각 커밋에 REBASE NOTE). #2 는 upstream 이 기대는 재개 lookahead 인프라 없이 핵심만 옮겼다.
첫 런은 warmup 10스텝(lr 1e-7~1e-6) 뒤 처음부터 다시 시작한다 (사용자 결정 2026-10-08). 같은 검토에서 upstream #4171(loss 안 시퀀스 마스킹)도 이식했다.
충돌을 풀며 깨끗이 합쳐진 파일이 이 브랜치에 없는 모듈을 import 하는 경우를 찾았다 (`draft_config.coerce_draft_config` — 그대로면 `grpo.py` 전체 ImportError).
그래서 이식 뒤에는 바뀐 파일 전체를 `ruff --select F821,F401` 로 검사한다.

**교훈**: fork 가 upstream 에서 두 달 떨어지면 정확성 수정도 함께 놓친다. 큰 런 전에 분기점 이후 upstream 의 `fix(` 커밋을 우리 경로(async·Gym·Megatron) 기준으로 훑는다.

## NeMo-RL 단위 테스트가 실행 중인 운영 Ray 클러스터에 붙는다 (2026-10-08)

**발견 경위**: 이식 검증으로 main1 에서 `test_vllm_generation.py` 를 돌렸다. 통합 테스트 하나가 placement group 대기 시간 초과로 실패했는데,
`ray list placement-groups` 에 그 테스트의 `vllm-test-policy-cluster-separate-node0` 가 운영 클러스터 기록으로 남아 있었다 (REMOVED).
원인은 `tests/unit/conftest.py:403` 의 session autouse fixture `init_ray_cluster`(`init_ray()`)다. Ray 를 쓰지 않는 테스트를 골라 돌려도
세션마다 같은 노드의 실행 중 클러스터에 연결하고(로그 "Connecting to existing Ray cluster at address: 10.0.37.4:6379"), GPU 모니터(`ray_gpu_monitor`)도 띄운다.
`RAY_ADDRESS` 를 지워도 같다. 이번에는 GPU 가 모두 런에 잡혀 있어 테스트가 자원을 못 얻고 끝났다. 런이 끝나 GPU 가 비는 순간에 돌았다면 테스트가 GPU 를 가로챌 수 있었다.

**대응**: 운영 클러스터가 떠 있는 노드에서는 NeMo-RL pytest 를 돌리지 않는다 — 클러스터가 비어 있을 때만 돌린다. CPU 단위 테스트는 `CUDA_VISIBLE_DEVICES=""` 로 GPU 를 가린다.
운영 런이 쓰는 venv 도 건드리지 않는다 — pytest 는 scratchpad 의 별도 venv 에 드라이버 venv 의 site-packages 를 `.pth` 로 이어 붙여 쓴다.

**재발 (2026-10-08 21:3x, `--noconftest` 로도 붙음)**: 도구 teacher 런 중에 `--noconftest -k noncolocated` 로 `tests/unit/algorithms/test_grpo.py` 를 돌렸다.
출력에 `(pid=…, ip=10.0.37.2)` 워커 로그가 섞여 나왔다. 이 모듈 안의 autouse fixture `reset_env_calls` 가 `mock_env`·`mock_envs` 를 끌어오고,
그 fixture 들이 `MockEnvironment.remote()` 로 Ray 액터를 만든다. 첫 `.remote()` 가 Ray 를 자동 init 하면서 운영 클러스터(10.0.37.4:6379)에 붙었다.
액터는 `num_cpus=0` 이고 fixture 정리 때 지워졌다. 런은 영향이 없었다 (36/400 스텝, KL 0.0013).
- **격리 방법**: Ray 를 쓰는 테스트 모듈은 `RAY_ADDRESS=local RAY_TMPDIR=$(mktemp -d /tmp/rut.XXXX)` 로 새 로컬 Ray 에서 돌린다.
  `RAY_ADDRESS=local` 이면 새 인스턴스를 띄운다 (`ray/_private/services.py:705-708`). `RAY_TMPDIR` 는 소켓 경로 길이 제한(107자) 때문에 짧은 `/tmp` 경로를 쓰고 끝나면 지운다.
  pytest 는 `clean_run.sh env RAY_ADDRESS=… RAY_TMPDIR=… uv run --no-sync --with pytest python -m pytest --noconftest -p no:cacheprovider` 로 돌린다.
  `--no-sync` 는 운영 venv 동기화를 막는다 — worktree 에서 동기화하면 editable `nemo_rl` 이 worktree 를 가리키게 될 수 있다.
- **이 방식의 한계**: Ray 상태 API 의 자동 탐색은 운영·로컬 두 인스턴스를 보고 `ConnectionError: Found multiple active Ray instances` 로 실패한다. 연결은 하지 않는다.
  `test_grpo.py` 에서는 22개가 이렇게 실패하므로, 변경 전후 실행의 실패 집합을 비교해 판정한다 (`alpha/judge-split` 검증에서 같은 22개).

## 런을 연장해 재개하면 Megatron 스케줄러 assert 로 멈춘다 — 스케줄 길이가 max_num_steps 를 따라간다 (2026-10-07 ✅ `scheduler.max_steps`)

**발견 경위**: G7(저장 → 재개) 첫 시도에서 G5 의 step_3 를 `grpo.max_num_steps` 3 → 5 로 재개했다 (런 연장 시나리오).
MegatronPolicyWorker 8개가 초기화에서 `AssertionError: OptimizerParamScheduler: class input value 640 and checkpointvalue 384 for total number of weight decay iterations do not match` 로 죽었다.

**원인**: GRPO 는 `megatron_cfg.train_iters = min(max_num_steps, max_num_epochs × 에폭당 스텝)` 을 넣는다 (`nemo_rl/algorithms/grpo.py:1055`).
`scheduler.max_steps` 가 없으면 Bridge 는 `wd_incr_steps = train_iters × GBS` 로 계산한다 (`megatron/bridge/training/config.py` `_calculate_scheduler_steps`).
mcore `OptimizerParamScheduler.load_state_dict` 는 기본값(`override_opt_param_scheduler`·`use_checkpoint_opt_param_scheduler` 모두 false)에서 체크포인트 값과 다르면 assert 한다.
384 = 3 × 128, 640 = 5 × 128 이다. 스텝 수·에폭·블렌드 크기 중 하나라도 바꾸고 재개하면 같은 assert 가 난다.

**대응**: 레시피에 `policy.megatron_cfg.scheduler.max_steps: 100000` 을 둔다. Bridge 는 이 값으로 wd 증가·lr 감쇠 길이를 정해 train_iters 와 떼어 놓는다 (≥ train_iters 여야 한다).
lr·wd 가 상수(lr = min_lr = 1e-6, wd 상수)라 수치는 같다. lr·warmup·wd 값을 바꾸고 재개하면 여전히 assert 한다 — 의도된 보호다.
`check_alpha_recipe.py` 는 미설정을 WARN 으로, 실효 train_iters 보다 작은 값을 ERROR 로 막는다. 이 수정 전에 저장한 스모크 체크포인트(G5)는 감쇠 길이가 달라 이 레시피로 재개할 수 없다.

**교훈**: 저장 → 재개 게이트는 같은 설정 재개만이 아니라 런 연장 시나리오로 돌린다.

## RL 체크포인트를 HF 로 반출하면 Pai 가 토크나이저를 못 읽는다 — 변환기가 메타데이터를 transformers 5 형식으로 다시 쓴다 (2026-10-07 ✅ `tools/export_rl_hf.sh`)

**발견 경위**: G6(RL 체크포인트 → HF → Pai)에서 G5 step_3 를 NeMo-RL 변환기(`examples/converters/convert_megatron_to_hf.py`)로 반출했다.
가중치는 정상이었다. 텐서 14,181개의 이름·shape 가 일치했고, 동결된 라우터·expert bias 는 비트 동일했다. 그런데 Pai 환경(transformers 4.57)에서 반출본의 토크나이저를 열면
`ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.` 로 실패했다. Pai 의 forward_sanity·벤치·다음 단계 입력이 모두 여기서 멈춘다.

**원인**: 변환기는 Megatron 워커 venv 의 transformers 5.8.1 로 tokenizer·config 를 다시 저장한다.
- `tokenizer_config.json`: `tokenizer_class` 가 `PreTrainedTokenizerFast` 에서 transformers 5 전용 이름 `TokenizersBackend` 로 바뀐다. `added_tokens_decoder`·`additional_special_tokens` 가 빠진다.
- chat template 이 `tokenizer_config.json` 밖의 `chat_template.jinja` 로 빠진다 (끝 개행만 다르다). `special_tokens_map.json`·`tokenizer_metadata.json` 은 쓰지 않는다.
- `config.json` 은 키가 바뀐다 (`_full_attention_interval`, `rope_parameters`·`layer_types` 추가, `dtype`). 그래도 Pai 의 `configuration_alpha.py` 는 같은 값으로 해석한다 (`compare_hf_forward.py --raw-config` 의 속성 차이는 `rope_parameters` 추가뿐이다).
- `A_log` 18개가 fp32 대신 bf16 로 저장된다. 값은 같다. Pai 도 bf16 로 학습한 값을 fp32 로 저장했을 뿐이라 bf16 왕복이 18/18 무손실이다. HF `modeling_alpha.py` 는 `A_log.float()` 로 쓴다.
- 문서의 변환 명령(`uv run --locked --extra mcore`)은 driver venv 의 패키지 4개를 바꾸고 81개를 설치한다. driver 를 쓰는 런이 돌 때 실행하면 그 런의 환경이 바뀐다.

**대응**: 반출은 `tools/export_rl_hf.sh <ckpt>/step_N <out>` 로만 한다. 이 스크립트는 세 단계로 돈다.
① 변환기를 Megatron 워커 venv 의 python 으로 직접 실행한다 (CPU·gloo, step_3 에서 2분 17초).
② 가중치가 아닌 파일은 시작점 hfmodel(`policy.model_name`)에서 그대로 복사하고, 시작점에 없는 파일(`chat_template.jinja`)은 지운다.
③ `compare_hf_weights.py` 로 시작점과 대조한다. 출처(step·가중치 경로·커밋)는 `rl_export.json` 에 남는다.

**교훈**: 두 스택의 인터페이스는 HF 체크포인트 하나다. 가중치 검증만으로는 부족하다. 소비자(Pai) 환경에서 토크나이저와 config 까지 열어 본다.

## 2노드 학습이 2스텝째 체크포인트 저장에서 멈춘다 — 비동기 writer 가 ES 메모리의 CUDA IPC 를 받지 못한다 (2026-10-07 ✅ `async_save: false`)

**발견 경위**: G5(실레시피 2노드 스모크) 1스텝은 정상이었다(KL 0.0016). 2스텝째 저장(`save_period 2`)에서 학습 워커 8개의 비동기 writer
(nvidia-resiliency-ext `PersistentAsyncCaller`)가 `Failed to receive CUDA IPC handle from the training process (pidfd_getfd: Operation not permitted)` 로 죽었다.
학습 프로세스는 저장 마무리를 기다리며 8분 넘게 GPU 0% 로 멈췄다.

**원인**: colocated refit 사고(아래)와 같다. 상주 writer 는 학습 텐서를 CUDA IPC 로 받는데, 128K 학습에 필요한 `expandable_segments` 메모리는
`pidfd_getfd` 가 있어야 열린다. 이 컨테이너는 `ptrace_scope=1`·`CAP_SYS_PTRACE` 없음이다. `async_save: true` 는 `grpo_math_1B.yaml` 상속값이었다.

**대응**: 레시피 `policy.megatron_cfg.checkpoint.async_save: false` (동기 저장 — 학습이 저장 동안 멈추지만 IPC 를 쓰지 않는다).
`check_alpha_recipe.py` 가 ES + 비동기 저장 조합을 ERROR 로 막는다. 대안(호스트에서 `kernel.yama.ptrace_scope=0`)은 컨테이너 밖 권한이라 쓰지 않는다.

**교훈**: ES 를 켜면 같은 노드 안의 프로세스 간 CUDA 메모리 공유 경로(colocated refit, 비동기 저장)가 전부 막힌다. 새 기능을 켤 때 IPC 를 쓰는지 먼저 본다.

## Gym 도구 환경의 롤아웃이 off-policy 다 — vLLM 이 `strict: true` 도구에 제약 디코딩을 건다 (2026-10-07 ✅ `VLLM_ENFORCE_STRICT_TOOL_CALLING=0`)

**발견 경위**: G3(Gym 1노드 스모크, judge 불필요 10개 환경 138행)의 Generation KL 이 0.0043·0.0045 로 기준 0.002 의 2배를 넘었다.
같은 모델의 네이티브 경로(R1·G2)는 0.0015 였다. `seq_logprob_error_threshold 2` 가 64개 중 17~21개 시퀀스를 마스킹했다.

**진단** (`tools/analyze_gym_logprob_gap.py`, 토큰 덤프 = `env.should_log_nemo_gym_responses=false`):

| 환경 | 시퀀스 | mult_prob_error > 2 | 중앙값 (강제 켬 → 끔) |
|---|---|---|---|
| toolcall_schema 단일 스텝 도구 | 20 | 14 → 0 | 6.53 → 1.015 |
| swe_pivot | 8 | 2 → 0 | 1.24 → 1.007 |
| single_step_tool_use | 8 | 1 → 0 | 1.11 → 1.022 |
| 도구 없는 6개 환경 | 28 | 0 → 0 | 1.02~1.03 (변화 없음) |

- 어긋남은 도구를 선언한 행에만 있었고, 특수 토큰이 아니라 추론 텍스트 중간에서 시작했다.
- R3 는 원인이 아니다 — forward 검증 3,840건 불일치 0, CP 토큰 일치 256,662행.
- 로그에 xgrammar 마스크 커널(`apply_token_bitmask_inplace_kernel`) 컴파일이 있었다 (강제 끔 런에서는 0회).

**원인**: vLLM 0.25.1 `tool_parsers/structural_tag_registry.py::get_model_structural_tag` 는 `tool_choice` 가 auto 여도 도구 하나라도
`strict: true` 면 구조 태그(xgrammar)를 만든다. 환경변수 `VLLM_ENFORCE_STRICT_TOOL_CALLING` 기본값이 True 다 (`vllm/envs.py`).
결정 13 으로 남긴 `strict: true`(RL 블렌드 도구 행 전부)가 롤아웃을 제약된 분포에서 샘플링하게 만들었고, vLLM 이 돌려준 logprob 도 그 분포 기준이었다.
학습측은 제약 없는 분포로 계산하므로 importance ratio 가 틀리고, threshold 마스킹이 도구 샘플을 대량으로 버린다.

**대응**: `policy.generation.vllm_cfg.env_vars.VLLM_ENFORCE_STRICT_TOOL_CALLING: "0"` (레시피 기본값). `strict` 는 프롬프트 렌더에 그대로 남는다 —
도구 행 프롬프트 36/36 에 `<strict>True</strict>` 확인. 결과 KL 0.0017, 마스킹 0, 도구 환경 k3 0.0006~0.0011. `check_alpha_recipe.py` 가 누락을 ERROR 로 막는다.

**교훈**: ① 서빙 계층의 "편의 기능"(구조 태그·제약 디코딩·파서)이 RL 에서는 샘플링 분포를 바꾼다. 롤아웃 경로를 바꾸면 R1 식 KL 을 그 경로에서 다시 잰다.
② KL 평균만 보지 말고 시퀀스·환경별로 쪼갠다 — 이번 결함은 전체의 1/3 인 도구 시퀀스에만 있었다.

## vLLM 엔진이 `max_num_batched_tokens (16384) is smaller than max_model_len` 으로 기동하지 못한다 — chunked prefill 을 끈 레시피 (2026-10-07 ✅ 레시피 수정)

**발견 경위**: G3 를 32K 로 올리자 vLLM async 워커가 `SchedulerConfig` 검증에서 죽었다.
**원인**: `student_rlvr1_alpha.yaml` 은 R3 route 누락을 피하려고 chunked prefill 을 끈다(D2, upstream R3 레시피와 같음). chunked prefill 이 꺼지면
vLLM 은 프롬프트 하나를 한 스텝에 넣어야 하므로 `max_num_batched_tokens >= max_model_len` 을 요구한다. 레시피 값 16384 는 128K 본 런에서도 실패한다.
**대응**: `vllm_kwargs.max_num_batched_tokens: ${policy.max_total_sequence_length}`. `check_alpha_recipe.py` 가 이 조합을 ERROR 로 막는다.

## NeMo-RL 은 첫 턴 프롬프트가 `max_model_len` 을 넘는 Gym 행을 건너뛰지 않고 런을 멈춘다 (2026-10-07, 데이터 게이트)

**발견 경위**: G3 를 8K 로 돌렸더니 프롬프트가 8K 를 넘는 행(스모크 144행 중 15행)에서 vLLM 이 400 을 돌려줬다. NeMo-RL 은
`ValueError: NeMo Gym returned a result with no generation data ... the prompt for the first turn already exceeds the vLLM max_model_len` 로 런 전체를 멈췄다 (`nemo_rl/environments/nemo_gym.py:826`).
**함의**: 블렌드의 렌더 프롬프트 길이를 런 전에 잰다. `rlvr1_alpha_judgefree.jsonl` 은 최대 39,266 토큰(중앙값 3,381, p99 33,972, 32K 초과 1,144행 — 전부 SWE 피벗, 64K 초과 0행)이라 128K 첫 런에는 해당하지 않는다.
측정 함정: Responses API 입력의 `function_call_output` 본문은 `output` 필드, 이전 추론은 `reasoning.summary` 다. 첫 측정은 이 둘을 빠뜨려 최대 22,503 으로 과소 측정했다.
도구 호출 `arguments` 는 JSON 문자열을 dict 로 바꿔야 alpha 템플릿이 렌더된다 (`tools/measure_blend_prompt_lengths.py`).
멀티턴 환경은 턴이 쌓이며 넘을 수 있으므로 환경을 추가할 때 다시 본다.

## colocated refit 이 `pidfd_getfd: Operation not permitted` 로 실패한다 — `expandable_segments` 메모리의 CUDA IPC 를 컨테이너가 막는다 (2026-10-07 ✅ colocated 에서는 ES 끔)

**발견 경위**: G3(Gym 1노드 colocated 스모크)가 첫 refit 에서 죽었다. vLLM async 워커(EngineCore)가 학습 워커의 CUDA IPC 핸들을 열다가
`weight load failed: RuntimeError: pidfd_getfd: Operation not permitted; missing keys (14181)` 를 냈다. 같은 1노드 colocated 인 R1·G2 는 통과했다.

**원인**: 레시피 `student_rlvr1_alpha.yaml` 이 정책 워커에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(결정 D6 권고값)를 켠다.
ES 로 잡은 메모리를 다른 프로세스가 CUDA IPC 로 열려면 수신 측이 `pidfd_getfd` 로 내보낸 쪽의 파일 기술자를 가져와야 한다.
이 컨테이너는 `kernel.yama.ptrace_scope=1` 이고 `CAP_SYS_PTRACE` 가 없다. vLLM EngineCore 는 학습 워커의 자손이 아니라서 거부된다.
R1·G2 는 ES 를 켜지 않아 일반 cudaMalloc IPC 를 썼다.

**대응**: colocated(IPC refit) 런에서는 ES 를 끈다 (`policy.megatron_cfg.env_vars=null`). 2노드 분리 토폴로지는 NCCL broadcast 로 refit 하므로 IPC 를 쓰지 않는다 — G5 에서 확인한다.
`tools/check_alpha_recipe.py` 가 colocated + ES 조합을 ERROR 로 막는다. 128K 학습(R4: ES 없으면 rank 당 12K 토큰 상한)은 분리 토폴로지 전제다.

**교훈**: 할당자 설정은 학습 메모리만이 아니라 프로세스 간 메모리 공유(refit) 경로까지 바꾼다. NeMo-RL 주석의 "ES 는 weight transfer 5배 느림"보다 이 환경에서는 "colocated IPC 불가"가 실제 제약이다.

## R4 결과 JSON 의 loss 가 전부 NaN 이고 처리량도 과소 측정됐다 — 하네스가 없는 키를 읽고 워밍업 스텝을 쟀다 (2026-10-07 ✅ 수정)

**발견 경위**: 이식 위험 보고(`RLVR_READINESS.md`)를 쓰며 R4 산출물을 다시 읽었다. 5개 조건(16K/CP2 ~ 128K/CP8) 전 rank·전 스텝의 `loss` 가 NaN 이었다.

**원인**: `tools/measure_train_memory.py` 가 `res.get("loss", nan)` 으로 읽는다. `MegatronPolicyWorkerImpl.train` 은 `global_loss`·`grad_norm` 을 반환한다
(`megatron_policy_worker.py:1004-1011`). 키가 없어 기본값 NaN 이 찍혔다. 수치 문제가 아니다.

**함의**: R4 는 메모리·실행 가능성만 증명했다. 128K/CP8 에서 loss·grad norm 이 유한한지는 본 적이 없다.

**같은 하네스의 두 번째 오류 — 처리량**: R4 의 2번째 스텝 시간(8.5~8.7 s)을 정상 상태로 읽어 128K 처리량을 약 15K tok/s 로 보고했다.
R5(2026-10-07)에서 3 스텝을 재니 정상 상태는 샘플 1개/스텝 5.47 s, 4개/스텝 15.7 s(33.4K tok/s)였다. 2번째 스텝에도 커널 autotune 등 워밍업이 남아 있었고,
샘플 1개 스텝은 옵티마이저·동기화 고정비가 샘플당으로 붙는다.

**교훈**: 게이트 하네스는 읽는 키가 실제로 있는지 assert 한다. 기본값으로 덮인 지표는 "측정 안 함"과 구별되지 않는다.
처리량은 워밍업이 끝난 스텝(3번째 이후)과 여러 마이크로배치 스텝으로 잰다.

## 장문맥 학습 스텝 OOM 은 총량이 아니라 단편화다 — PyTorch 캐시가 쥔 빈 블록 때문에 Triton 할당이 실패한다 (2026-10-07, ES 로 128K/CP8 통과 · 기본값 채택은 결정 대기)

**발견 경위**: R4 메모리 실측(sub1, Muon 레시피, 학습 전용 노드 가정)에서 rank 당 16K 토큰 조건이 길이·CP 와 무관하게 전부 OOM 이었다(32K/CP2 · 64K/CP4 · 128K/CP8).
rank 당 12K 토큰(96K/CP8)까지는 통과했다. 수치표는 [`GATES.md`](GATES.md) R4.

**진단**:
- 예외가 `RuntimeError: Triton Error [CUDA]: out of memory` 였다. PyTorch OOM 이 아니다. GDN 커널(fla Triton·FlashQLA TileLang)은 PyTorch 캐싱 할당자 밖에서 메모리를 잡는다.
  PyTorch 자신의 할당은 OOM 직전에 캐시를 비우고 재시도하지만, 바깥 할당에는 그 기회가 없다.
- 96K/CP8 에서 이미 reserved − alloc 이 13.6 GB 였다(67.6 − 54.0). 캐시가 쥔 빈 블록이 Triton 몫을 막는다.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(ES) 하나로 128K/CP8 이 통과했다. reserved − alloc 은 2.3 GB 로 줄었다.
  `logprob_chunk_size` 를 2048 → 1024 로 줄이면 peak 가 3.8 GB 더 내려가지만 필수는 아니다.
- 메모리 구성(`torch.cuda.memory._record_memory_history` 스냅샷, 16K/CP2 rank 0): 고정 바닥 27.0 GB 가 가중치·grad·Muon 상태다.
  나머지 대부분은 RL loss 의 `ChunkedDistributedLogprob.backward`(`nemo_rl/distributed/model_utils.py:267`)가 차지한다.
  `grad_input = zeros_like(logits)`(:347, T×V), 저장된 logits, chunk 별 fp32 softmax, int64 `one_hot`(:365) 이다. Muon 상태는 지배 항이 아니다.

**함의**:
- Muon optimizer-state offload(Pai 기능 #4) 포팅은 1노드 128K/CP8 에 필요 없다. 2026-10-06 추정(바닥 ≈45 GB, 오프로드 필요 가능성)을 실측 27 GB 가 대체한다.
- ES 에는 대가가 있다. NeMo-RL 주석(`megatron_policy_worker.py:2742`)은 ES 가 weight transfer 를 약 5배 느리게 한다고 적는다. upstream Megatron 레시피 다수는 `expandable_segments:False` 를 명시한다.
  alpha 구성에서의 refit 비용은 미측정이다. RL 기본값 채택은 결정 대기다 (`STATUS.md`).
  - 2026-10-07 재확인: "5배" 근거는 `_clear_fp8_caches` docstring 한 줄이다(FP8 단편화 맥락, 측정값·전송 방식 없음). 2노드 분리 refit 은 NCCL broadcast 라 TCP 대역이 지배한다.
    Pai SFT 는 `train.sh:82` 에서 ES 를 항상 켰다. 같은 docstring 은 `max_split_size_mb:512` 를 대안으로 든다 (G4 에서 대조).

**같이 드러난 NeMo-RL 제약** (전부 `nemo_rl/models/megatron/setup.py`):
1. CP>1 은 sequence packing 이 필수다 (:839-840 assert). alpha 레시피 불변량 "packing 끔"(GDN)과 충돌한다.
   장문맥 RL 은 packing 을 켜야 하고, packing+CP 경로의 GDN 수치(R1 KL·M5)는 미검증이다.
2. CP>1 은 `use_fused_linear_logprobs` 와 함께 쓸 수 없다 (:843-844). 대신 `logprob_chunk_size` + `sequence_packing.fuse_loss` 를 쓴다 (Ultra 레시피 값).
3. `logprob_chunk_size` 는 `defer_fp32_logits: true` 를 요구한다 (:1130).

**교훈**: ① PyTorch 밖에서 할당하는 커널이 있는 모델은 `max_memory_allocated` 가 여유로워도 OOM 이 난다. reserved 와 alloc 의 차이를 같이 본다.
② 메모리 추정은 실측으로 대체한다. 오프로드가 필요하다는 추정은 틀렸다.

## NeMo Gym 멀티턴이 2번째 모델 호출부터 전부 실패한다 — 턴 경계를 EOS(id 0)로 찾았다 (2026-10-07 ✅ `turn_end_token_id`)

**발견 경위**: R3 렌더 게이트의 Gym 경로 P4 가 0/40 이었다. 이전 턴 토큰을 이어 붙이는 2번째 호출부터 전부
`AssertionError: EOS token #0 not found in template_token_ids` 로 실패했다.
R1 KL 게이트는 단일 턴 네이티브 경로라 이 경로를 지나지 않는다.

**원인**: NeMo-RL `replace_prefix_tokens`(`nemo_rl/models/generation/openai_server_utils.py`)는 다음 턴 요청의 템플릿 토큰에서 이전 턴 끝을 `tokenizer.eos_token_id` 로 찾는다.
그 자리에 모델이 실제로 낸 토큰을 이어 붙인다. alpha 의 `eos_token` 은 `<|endoftext|>`(0, 사전학습 문서 경계)이고 chat 턴 끝은 `<|im_end|>`(3)다.
템플릿 렌더에는 0 이 없으므로 경계를 찾지 못한다. 롤아웃 정지 토큰 {0, 3} 은 맞았다 — 정지와 경계는 다른 문제다.

**수정 (본체, rebase 충돌 후보)**: `replace_prefix_tokens(..., turn_end_token_id=None)` 인자를 추가했다. `None` 이면 기존 동작이다.
vLLM 워커가 `policy.generation.vllm_cfg.turn_end_token_id` 를 넘긴다 (`vllm_worker_async.py`, 설정 키 `vllm/config.py`).
alpha 기본값은 `grpo_alpha_smoke.yaml` 의 `turn_end_token_id: 3` 이다. 단위 테스트 2건을 추가했고 `test_openai_server_utils.py` 7/7 pass 다.
R3 P5(실제 패치 호출)에서 경계 문제는 사라졌고, 남은 차이는 도구 정의뿐이다.

**교훈**: EOS 와 턴 끝이 다른 토크나이저에서는 NeMo-RL 의 "EOS = 턴 끝" 가정을 의심한다. 정지 토큰 확인만으로는 멀티턴 경계를 보장하지 못한다.

## Gym → vLLM 렌더 경로가 SFT 형식과 3군데 다르다 — `strict` 제거 · `description` None · 비도구 이력 reasoning 복원 (2026-10-07, `strict` 유지 결정 · 구현 대기)

R3 렌더 게이트가 찾았다. 턴 경계(위 항목)를 고친 뒤 남는 차이들이다.

| # | 차이 | 영향 | 처리 |
|---|---|---|---|
| 1 | Gym `VLLMModel._strip_hosted_only_tool_fields`(`responses_api_models/vllm_model/app.py:471`)가 도구 정의의 `strict` 를 무조건 지운다 | SFT Agentic-v2 interactive 셋은 `<strict>True</strict>` 를 렌더한 형식으로 학습했다. RL 블렌드 도구 행 13,190/13,190 이 `strict: true` 다. 다른 SFT 도구 셋(kotool 등)에는 strict 가 없다 | **사용자 결정 2026-10-07: 유지.** 미구현. 방법: Gym 0.6.0 플러그인 루트(`NEMO_GYM_EXTRA_ROOTS`)에 `VLLMModel` 서브클래스 서버를 두고 `strict` 가 bool 이면 남기고 null 이면 지운다 — Gym fork 불필요. 선례는 Gym 의 `responses_api_models/vllm_model_with_compaction`. 구현 뒤 R3 P6 을 실경로로 재실행 |
| 2 | description 없는 도구를 vLLM 이 `<description>None</description>` 으로 렌더한다 (`FunctionDefinition.model_dump()` 가 None 을 남김) | RL 블렌드 0건 | 데이터 게이트: RL 환경 데이터에 description 필수 검사 |
| 3 | 도구 없는 멀티턴 롤아웃(c3b)에서 이전 턴 reasoning 이 이력에 복원된다. SFT 는 비도구 이력의 reasoning 을 strip 해 학습했다 | 현 블렌드의 해당 환경 영향 없음 | 구조적. 멀티턴 비도구 환경을 추가할 때 R3 재실행 |

**설정 권고**: alpha Gym 레시피는 `chat_template_kwargs: null` 로 둔다. Ultra 레시피의 `truncate_history_thinking: false` 를 복사하면 도구 없는 이력의 reasoning 까지 복원돼 SFT 규약과 어긋난다 (R3 의 `dataset/ultra` 케이스).

## 엔진 동등성은 절대 임계가 아니라 노이즈 기준선으로 판정한다 — 하네스의 라우터 bf16 누락도 기준선이 잡았다 (2026-10-07 ✅ M5 PASS)

**경위**: M5(SFT 엔진 Pai ↔ RL 엔진 NeMo-RL) 첫 하네스에서 Pai 측 라우터가 bf16 으로 계산됐다.
Pai `tools/alpha_config.py emit-megatron-flags` 는 가중치 검증용이라 forward 설정(`--moe-router-dtype fp32`, router·permute fusion, dropout 0)을 내보내지 않는다.
SFT 구성(Pai `configs/model/baseline_48L.yaml`)대로 플래그를 더해 다시 쟀다. 잘못된 런은 버리지 않고 섭동 기준선으로 썼다 (`$NRL_ROOT/gates/engine_parity_iter2400/pai_router_bf16_harness_bug/`).

**판정 원리**: MoE·bf16 에서는 같은 엔진끼리도 gradient cos 중앙값이 0.98 근처에서 포화한다. 토큰이 적게 몰린 expert 는 라우팅 하나만 뒤집혀도 gradient 가 크게 바뀐다.
0.999 같은 절대 임계를 쓰면 같은 엔진끼리도 FAIL 이다. 그래서 HF(제3 구현)와 섭동 런을 기준선으로 두고, Pai↔NeMo-RL 거리가 그 안에 있는지 본다. 수치는 [`GATES.md`](GATES.md) M5.

**교훈**: ① 동등성 게이트에는 같은 측정을 구현만 바꾼 기준선을 둔다. ② 가중치 검증용 설정 도구를 forward·학습 하네스에 그대로 쓰지 않는다.
③ GDN `A_log` 는 SFT 에서도 bf16 으로 계산했으므로 RL 과 차이가 없다 (확인 완료).

## GDN 재귀 상태가 vLLM 에서 bf16 으로 저장된다 — 디코드 토큰마다 반올림이 누적돼 rollout-vs-train logprob 이 생성 위치에 따라 벌어진다 (2026-10-06 ✅ RL 레시피 기본값 fp32)

원장 #21. 벤치 fleet·채팅·SDG 교사 서빙(전부 bf16 상태)의 처리 결정은 Pai 쪽 미결이다 — Pai `KNOWN_ISSUES.md` 10-06 항목과 Pai `STATUS.md` "열린 사용자 결정".

**발견 경위**: NeMo-RL alpha 8-GPU GRPO KL 게이트(agentic iter2400, `grpo_alpha_smoke.yaml`, main1, EP8·colocated·16×8 롤아웃·최대 4096 토큰)가 Generation KL 0.0042 / 0.0037 / 0.0040 으로 기준 0.002 FAIL.
같은 환경의 1.5B dense 스모크는 0.0007, 같은 ckpt 의 refit 검증(10 토큰)은 mean(exp|Δ|) 1.0293 PASS 였다.

**진단 방법**: NeMo-RL 이 step 마다 남기는 `train_data_step*.jsonl` 의 토큰별 `generation_logprobs`(vLLM)·`prev_logprobs`(mcore)를 오프라인 분해(`tools/analyze_rollout_logprob_gap.py`, CPU). 재계산 k3 KL 이 게이트 출력과 일치(0.00423 = 0.0042).
seed 가 같아 런 사이 롤아웃이 동일하다(생성 토큰 440,577 · Avg Reward 0.1953 동일) → 설정 하나만 바꾼 통제 대조. 산출 `$NRL_ROOT/gates/agentic_iter2400/`.

| 런 (Adam) | KL step 1 / 2 / 3 | \|Δ\|>0.5 토큰 | 위치 0–256 | 1k–2k | 3k–4k | mult_prob_err |
|---|---|---|---|---|---|---|
| ① 기본 | 0.0042 / 0.0037 / 0.0040 | 0.248% | 0.0024 | 0.0044 | 0.0050 | 1.050 |
| ② + R3 (Router Replay) | 0.0026 / 0.0022 / 0.0024 | 0.056% | 0.0014 | 0.0026 | 0.0032 | 1.036 |
| ③ + R3 + GDN 상태 fp32 | **0.0015 / 0.0013 / 0.0014** | 0.011% | 0.0013 | 0.0015 | **0.0015** | 1.024 |

**원인은 둘이다.**
1. **MoE 라우팅 뒤집힘 (꼬리 오차)** — vLLM 과 mcore 의 hidden state 가 앞단 커널 차로 bf16 수준 달라 top-8 경계 expert 가 갈린다. 라우터 자체 정밀도는 문제가 아니다: 연산은 mcore fp64(NeMo-RL 기본 `moe_router_dtype`)·vLLM 플러그인 fp32, `expert_bias` 는 양쪽 fp32, refit 은 dtype 무변환 전송(라운드트립 게이트가 dtype 변화를 FAIL 로 잡는데 14,181/14,181 통과). R3 로 해결.
2. **GDN 재귀 상태 bf16 저장 (위치 누적)** — vLLM `mamba_ssm_cache_dtype` 기본 `"auto"` 는 재귀 상태에 conv/KV 캐시 dtype = 모델 dtype(bf16)을 쓴다(vLLM 0.25.1 `model_executor/layers/mamba/mamba_utils.py:91-92`). 디코드는 토큰마다 상태를 읽고·갱신하고·bf16 으로 다시 쓰므로 반올림이 다음 토큰으로 넘어가 쌓인다. 학습(mcore fla chunked)은 한 forward 안에서 상태를 fp32 로 들고 간다. fp32 로 바꾸자 위치 프로파일이 평탄해졌다(③, step 2·3 도 0.0011~0.0015).

**적용 (사용자 지시 2026-10-06)**: alpha RL 레시피 기본값에 `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype: float32` + `policy.router_replay.enabled: true` (`grpo_alpha_smoke.yaml`, 모든 alpha GRPO 레시피가 상속). NeMo-RL upstream 의 Mamba 계열 레시피(Nemotron nano·super·ultra 등 12종)는 전부 float32 를 지정한다 — alpha 레시피에만 빠져 있었다.
Muon 레시피(`grpo_alpha_smoke_muon.yaml`)도 같은 기본값으로 0.0015 / 0.0013 / 0.0013 PASS.

**교훈**: ① 시간축으로 누적되는 상태(GDN·Mamba·KDA)는 **저장 dtype** 이 정확도를 좌우한다. 토큰마다 새로 계산하는 값(라우터 logits)과 달리 오차가 다음 토큰으로 넘어간다.
② 짧은 검증은 이 누적을 못 본다 — forward 패리티는 teacher-forced prefill(디코드 없음), refit 검증은 10 토큰 디코드였다. 디코드 길이에 따른 **위치별 지표**로 본다.
③ 오차의 원인이 하나라고 가정하지 않는다 — R3 만 켰을 때 0.0026 이라 "거의 됐다"로 오판할 수 있었다. 꼬리(라우팅)와 위치 누적(상태 dtype)은 분해해야 갈린다.

## Qwen3-Next 계열 코드의 zero-centered RMSNorm 기본값 — 가중치 검증은 통과하고 forward 만 조용히 깨진다 (2026-08-13 ✅, 2회 조우)

alpha 는 **표준 RMSNorm**(`w*x`)인데 Qwen3-Next 계열 구현은 Megatron-Bridge(mcore)와 vLLM 양쪽 모두 zero-centered(`(1+w)x`)가 기본이다.
이 델타를 놓치면 **가중치 검증은 100% 통과하면서 forward 가 전멸**한다 (Pai 의 1p 사건 2026-05-26 과 동일 클래스).
AlphaBridge 와 vLLM 플러그인 포팅에서 각각 한 번씩 조우했다. 플러그인은 표준 RMSNorm 으로 전면 교체하고,
융합 QK-norm 커널은 zero-centered +1.0 이 하드코딩돼 있어 비활성화했다.
**대응**: 체크포인트·브리지·플러그인을 바꾸면 라운드트립과 forward 패리티 게이트를 **함께** 돌린다 (`GATES.md` 1·2).

## 환경 구축·운영 원장 #1~#27 (2026-08-13 RL 전용 세션 · 2026-10-06~07 Pai 컨테이너 공존)

증상으로 검색한다. #1~#14 의 해법은 `project_s/setup_nemo_rl_env.sh`(RL 전용 세션용)에, #15~#20 은
`project_s/setup_nemo_rl_env_noninvasive.sh`(Pai 컨테이너 공존, 현행 기본)에 인코딩돼 있다. 설치 절차는 [`SETUP.md`](SETUP.md).

| # | 증상 | 원인 | 해법 |
|---|---|---|---|
| 1 | `The NVIDIA driver on your system is too old (found version 12080/12020)` | 드라이버 535는 CUDA 12.x 세대, NeMo-RL은 torch cu130 | `cuda-compat-13-2` 설치. 단 Backend.AI `libcudahook`(/etc/ld.so.preload)이 **하드코딩 경로 `/usr/local/cuda/compat/lib.real`** 에서 libcuda를 선점 로드하므로 LD_LIBRARY_PATH는 무효 — `compat/lib.real/`에 595 라이브러리 + `lib→lib.real` 심볼릭 링크 + `.checked` 마커 구조를 재현해야 함 |
| 2 | uv 설치 중 `No space left on device` | `/home/work`는 49G 루프 디바이스, uv 캐시 45G+ | `UV_CACHE_DIR=/opt/uv-cache` (오버레이 로컬 디스크) |
| 3 | deep-ep 빌드: `detected CUDA version (12.8) mismatches ... (13.0)` | 시스템 nvcc 12.8로 cu130 대상 빌드 | CUDA 13.0 툴킷 병행 설치, `CUDA_HOME=/usr/local/cuda-13.0` |
| 4 | `fatal error: cuda/std/array: No such file` | CUDA 13에서 CCCL(libcu++) 헤더가 `include/cccl/`로 이동 | `CPATH=$CUDA_HOME/include/cccl` |
| 5 | `ptxas error: Feature 'elect' requires .target sm_90` (compute_75로 빌드됨) | uv 빌드 샌드박스에서 GPU 미감지 → 구형 아키텍처 폴백 | `TORCH_CUDA_ARCH_LIST=9.0a`, `NVTE_CUDA_ARCHS=90` 명시 |
| 6 | `uv sync --extra mcore --extra vllm` → `incompatible with the declared conflicts` | 의도된 설계 (역할별 venv) | extra별로 따로 sync |
| 7 | TE fused attn: `Multiple libcudart libraries found: .so.12 and .so.13` | cu12 빌드 휠(cudnn frontend 등)이 ldconfig의 CUDA 12 경로에서 libcudart.so.12를 로드 | `/usr/local/cuda`→13.0 전환 + `988_cuda-12.conf`/`gds-12-8.conf`를 `/etc/ld.so.conf.d/disabled/`로 이동 + ldconfig (NGC와 동일 상태 재현) |
| 8 | TE fused attn: `cuDNN Error: No valid engine configs` | 시스템 cuDNN 9.8(cu12)이 로드됨 (TE는 9.20 기준) | `CUDNN_HOME`/`LD_LIBRARY_PATH`를 워커 venv의 pip cuDNN으로 고정 (`nemo_rl_env.sh`에 포함) |
| 9 | Ray 워커 venv 구성: `nemo-gym build_editable` 실패 (`egg_base 'cache' does not exist`) | **uv 버전 불일치.** 신버전 uv(0.12.x)로 `uv lock` 재생성 시 editable 패키지의 `[package.metadata]`가 소실됨 | **uv는 반드시 0.11.28** (NeMo-RL `docker/Dockerfile`의 `UV_VERSION` 핀). 이 버전으로 재잠금하면 diff가 최소화됨. 스크립트가 핀 설치함 |
| 10 | 로컬 vLLM fork를 lock에 연결 시 `flashinfer-python` 충돌 (`nemo-gym[vllm]`은 0.6.12 핀) | 소스 오버라이드가 워크스페이스 전체(gym 분기 포함)에 전파 | **fork를 lock에 넣지 말 것.** vLLM 커스텀 모델은 플러그인 패키지로 통합 (`../README.md` 디렉토리 표) |
| 11 | mcore 변환 캐시 로드: `run_config.yaml not found` | 이전 실행 크래시가 남긴 **부분 변환 캐시** (캐시 검사는 디렉토리 존재만 확인) | `$NRL_MEGATRON_CHECKPOINT_DIR/model_*` 삭제 후 재실행. 브리지 코드 수정 중엔 `force_reconvert_from_hf: true` |
| 12 | `tools/refit_verifier.py` 각종 KeyError/AssertionError | 도구가 최신 설정 스키마 대비 낡음 | 수정 커밋 1eca2f383 (스키마 키 5종 + trust_remote_code + 정책 오프로드 + 단일GPU 메모리 설정) — 우리 브랜치에 반영됨 |
| 13 | `/home/work` 49G 재포화 (HF 모델/데이터셋 캐시) | HF 캐시 기본 경로가 소용량 루프 디바이스 | `~/.cache/huggingface` → `/opt/hf_cache/huggingface` 이전 + 심볼릭 링크 (세션 재생성 시 재수행 필요) |
| 14 | fla 0.4.2 `chunk_gated_delta_rule` 네이티브 GQA(Hk≠Hv) **NaN** | fla 자체 버그 (배치·varlen 모두) | mcore는 호출 전 repeat_interleave MHA 확장으로 우회 중이라 실경로 안전. 커널 직접 호출 코드에서는 반드시 확장 후 호출. upstream 보고 예정 |
| 15 | TE fused attn `Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13` (Pai 컨테이너) — `libtransformer_engine.so` 가 `libcudart.so.12`·`libcublas.so.12` 를 NEEDED | NGC 이미지 전역 `CUDACXX=/usr/local/cuda/bin/nvcc` → TE 의 CMake 단계가 12.9 nvcc·라이브러리를 씀(파이썬 단계는 `CUDA_HOME` 을 따르므로 착시). torch cpp_extension 빌드(DeepEP·mamba 등)는 무관 | `CUDACXX=$CUDA_HOME/bin/nvcc`·`LIBRARY_PATH=$CUDA_HOME/lib64/stubs`. **재빌드 시 uv 캐시 git checkout 의 `build/cmake/CMakeCache.txt` 를 지워야 한다** — 휠 캐시만 지우면 11 초 만에 같은 cu12 링크로 다시 나온다 |
| 16 | #15 해결 뒤에도 fused attn 에서 같은 예외 | cudnn-frontend shim(`cudnn_frontend_shim.h::load_cudart_so`)이 `libcudart.so.12`·`.13` 을 soname 으로 둘 다 dlopen, 둘 다 열리면 예외. `.so.12` 는 시스템 `ld.so.cache`(`000_cuda`·`cuda`·`988_cuda-12.conf`)로 열림 — LD_LIBRARY_PATH 정리로는 못 막음 | `LD_LIBRARY_PATH` 맨 앞에 **차단 디렉토리**(내용이 ELF 가 아닌 `libcudart.so.12`·`libcudart.so` 파일)를 둔다 → 해당 이름 dlopen 이 실패, 캐시로 넘어가지 않음. 프로세스 한정 |
| 17 | Ray Megatron 워커에서만 #16 재발 (단독 프로브는 통과) | `flashinfer/jit/__init__.py:102-105` 가 import 시 `$CUDA_LIB_PATH/libcudart.so.12`(기본 `/usr/local/cuda/targets/x86_64-linux/lib/`)를 **절대경로**·`RTLD_GLOBAL` 로 선로딩 | `CUDA_LIB_PATH=$CUDA_HOME/targets/x86_64-linux/lib` (cu13 툴킷엔 `.so.12` 가 없어 조건 거짓) |
| 18 | 워커 `ImportError: ... experimental_attention_variant_module_specs (…/Pai-Megatron-Patch/backends/megatron/Megatron-LM-251125/…)` | Pai 의 `PYTHONPATH` 가 Ray 워커로 누출. 또 **bash 는 stdin 이 소켓이면 비대화형이어도 `~/.bashrc` 를 source** 하고, 그 10행이 `~/.pai_megatron_alpha_env` 를 무조건 로드 → `env -i` 후에도 자식 bash 에서 되살아남 | 화이트리스트 `env -i`(BACKENDAI_*·NVIDIA_*·LD_PRELOAD·NCCL_CUDA_PATH·HOME 등) + **`</dev/null`** + `unset PYTHONPATH`·`PYTHONNOUSERSITE=1`. 같이 새던 것: `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`(logprob 정밀도), `OMP_NUM_THREADS=220`, py3.12 user site |
| 19 | 워커 `CUDA error: invalid device ordinal` (`torch.cuda.set_device(local_rank)`) | NeMo-RL 은 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 로 워커에 전 GPU 를 보이고 `ray.get_gpu_ids()[0]`(물리 번호)로 set_device. 드라이버에 `CUDA_VISIBLE_DEVICES=7` 을 주면 Ray 는 7 을 주는데 워커엔 0 번만 보임 | **`CUDA_VISIBLE_DEVICES` 로 GPU 부분집합을 지정하지 않는다.** 장수 제한은 `cluster.gpus_per_node`·Ray 리소스로 |
| 20 | `uv sync --extra vllm` 이 `deep-ep` 빌드 실패(`detected CUDA version 12.9 mismatches 13.0`) | vllm extra 도 deep-ep 소스 빌드를 포함 — 롤아웃 venv 에도 nvcc 13 필요 | `SETUP.md` §1 의 사설 CUDA 13 툴킷 |
| 21 | alpha 8-GPU GRPO KL 게이트 FAIL(Generation KL 0.0042, 기준 0.002) — 오차가 생성 위치를 따라 커짐 | ① vLLM `mamba_ssm_cache_dtype` 기본 auto = GDN 재귀 상태 bf16 → 디코드마다 반올림 누적 ② MoE top-8 경계 라우팅 뒤집힘 | alpha 레시피 기본값 `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype: float32` + `policy.router_replay.enabled: true` → 0.0015/0.0013/0.0014 PASS. 진단·대조표는 이 문서의 GDN 재귀 상태 항목 |
| 22 | `verify_muon_optimizer.py` 초판이 OmegaConf `${mul:...}` 해석 실패 · 8 rank 가 전부 GPU 0 · rank 0 만 보고 PASS | 도구 결함 3건: resolver 미등록, `set_device` 누락, layer-wise optimizer 는 dense 파라미터를 rank 별로 나누는데 rank 0 만 검사함 | `register_omegaconf_resolvers()` · `torch.cuda.set_device(LOCAL_RANK)` · all-gather 로 전 rank 합집합 판정 (R2, 2026-10-06) |
| 23 | `clean_run.sh` 아래에서 `MEM_SNAPSHOT_DIR`·`PYTORCH_CUDA_ALLOC_CONF` 같은 환경변수가 무시된다 | `clean_run.sh` 는 whitelist `env -i`(#18)라 호출 측 export 가 지워진다 | `clean_run.sh /usr/bin/env VAR=값 <cmd>` 로 whitelist 안쪽에서 주입한다 (2026-10-07) |
| 24 | `pytest` 를 어느 venv 에서도 못 찾는다 | lock 의 런타임 venv 에 test 의존성이 없다 | `$NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked --with pytest python -m pytest <test> </dev/null` (2026-10-07) |
| 25 | 2026-10-07 11:57 GPU 게이트를 벤치 fleet 이 쓰던 main1 GPU 위에 기동했다. 약 20 초 뒤 중단했고 fleet 은 재기동 없이 health 8/8 이었다 | 기동 전 점유 확인 누락. 정리 중 `pgrep -f` 가 자기 셸까지 매칭했다(exit 144) | GPU 런처는 기동 직전 점유를 검사해 한 장이라도 1 GiB 이상이면 중단한다 (`$NRL_ROOT/gates/*/run_*_sub1.sh` 패턴). 프로세스 정리는 PID 를 확인한 뒤 한다 |
| 26 | 실행 중인 게이트가 `SyntaxError` 로 죽었다 | 그 잡이 import 하는 NeMo-RL 소스(`openai_server_utils.py`)를 실행 도중 편집했다. NFS 공유 워킹트리라 즉시 반영된다 | 실행 중인 잡이 import 하는 코드는 편집하지 않는다 (2026-10-07) |
| 27 | 셋업 1차 실행 뒤 HOME 에 uv 캐시 24 GB 가 생겼다 (HOME 여유 43 → 20 GB) | 셋업 스크립트가 생성한 `clean_run.sh` 의 heredoc 인용 실수로 `NRL_ROOT` 가 빈 값이 됐고, 캐시가 기본 경로(HOME)로 갔다 | 생성 시점에 절대경로를 박는다. 스크립트 수정 후 처음부터 재실행해 검증했다(1151 s, rc=0). HOME 산출물은 삭제해 복구 (2026-10-06) |

기타: `from vllm import LLM`은 lock의 openai 2.6.1과 vllm 0.25.1 tool_parsers 불일치로
깨져 있으나 **NeMo-RL의 실제 vLLM 사용 경로는 무관** (GRPO 스모크로 확인). vLLM 단독
스모크 테스트를 이 임포트로 하지 말 것.
