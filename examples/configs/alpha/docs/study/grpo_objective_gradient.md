# GRPO 목적함수·기울기 스터디 — 현 레시피에서 실제로 미분되는 것

_작성 2026-10-15. 원전: `student_rlvr1_alpha.yaml`(teacher 레시피가 상속) · `teacher_codemath_alpha.yaml` ·
`nemo_rl/algorithms/loss/loss_functions.py` `ClippedPGLossFn` · `nemo_rl/algorithms/advantage_estimator.py` ·
`nemo_rl/algorithms/utils.py` · `nemo_rl/models/policy/workers/megatron_policy_worker.py` (리포 `639a47fe0` 기준). 줄 번호는 그 커밋 기준이다._

## 0. 결론

1. **advantage 는 미분하지 않는다.** 보상에서 미리 계산한 **상수 가중치** 다. 미분 대상은 정책의 토큰 로그확률 `log π_θ(y_t | x, y_<t)` 하나뿐이다.
2. 현 설정에서 KL 항은 0 이고 PPO clip 은 수학적으로 켜질 수 없다. 남는 objective 는 **토큰 단위 REINFORCE + 그룹 leave-one-out baseline + 절단 중요도 가중치(TIS)** 다.
3. 기울기는 한 줄이다.
   ```
   ∇_θ L = −(1/G) Σ_i Σ_t  m_{i,t} · w_{i,t} · A_i · ∇_θ log π_θ(y_{i,t} | x_i, y_{i,<t})
   G = 전역 유효 응답 토큰 수 · m = 마스크 · w = 절단 중요도 가중치(stop-grad) · A_i = 시퀀스 i 의 advantage(모든 토큰에 복제)
   ```
4. advantage 의 그룹 합은 정규화 뒤 0 이 아니고, 성공 1개 그룹은 정규화를 건너뛰어 성공 2개 그룹보다 작은 advantage 를 받는다 (§5).
5. 토큰 단위 정규화라 긴 시퀀스가 총 기울기에서 큰 몫을 차지한다. 잘린 응답이 가장 길고 DAPO 감점으로 `A < 0` 이므로 기울기의 큰 부분이 "그 토큰들 확률 낮추기" 로 간다 (§7).

## 1. 설정에서 살아 있는 항과 죽은 항

| 항 | 레시피 값 | 코드에서 실제로 |
|---|---|---|
| KL (reference) | `reference_policy_kl_penalty: 0.0` | `kl = 0`. reference 모델을 띄우지 않는다 (`grpo.py` `init_reference_model = kl_penalty > 0`) |
| PPO clip | `ratio_clip_min 0.2` · `ratio_clip_max 0.28` | **죽은 값.** `force_on_policy_ratio: true` 면 `ratios_clamped = ratios` 라 `max(loss1, loss2)` 가 항등 (`loss_functions.py:604-609`) |
| ratio | `force_on_policy_ratio: true` | `exp(curr − curr.detach())` → 값은 항상 1, 기울기는 `∇log π_θ` (`:607`) |
| 중요도 가중치 | `use_importance_sampling_correction: true` · `tis` · `[0.2, 5]` | `w_t = clamp(π_θ(y_t)/π_vLLM(y_t), 0.2, 5)`, detach (`:663`, `:696`) |
| 손실 집계 | `token_level_loss: true` | 전역 토큰 수 `G` 로 나눈 토큰 평균 (`:763`) |
| in-loss 필터 | `seq_logprob_error_in_loss: true` · 임계 2 | `mult_prob_error > 2` 시퀀스를 기울기에서 제외, 뒤에 `G/K` 보정 (`:457-481`, 워커 `:692`) |
| VAPO NLL | 가중치 0 | 없음 |
| 최종 | — | `loss = actor_loss + kl + 0` (`:819`) |

## 2. 목적함수에서 기울기까지

**정책 기울기.** 목적은 기대 보상이다.

```
J(θ) = E_{x~D} E_{y~π_θ(·|x)} [ R(x, y) ]
∇J   = E_{y~π_θ} [ R(y) · ∇_θ log π_θ(y|x) ]            (score-function 항등식)
log π_θ(y|x) = Σ_t log π_θ(y_t | x, y_<t)               (자기회귀 분해)
```

`R` 은 상수다. 미분은 `log π_θ` 에만 걸린다.

**baseline.** `y` 와 독립인 `b` 를 빼도 기대값은 같다.

```
E_{y~π_θ}[ b · ∇log π_θ(y) ] = b · ∇ Σ_y π_θ(y) = b · ∇1 = 0
```

leave-one-out baseline `b_i` 는 같은 프롬프트의 **다른** 15개 샘플로 만들어 `y_i` 와 독립이다. 편향이 없다.
std 로 나누는 것은 그룹별 스케일 조정이라 엄밀히는 추정량을 바꾸지만 부호와 방향은 유지한다.

**off-policy 보정.** 샘플은 `π_θ` 가 아니라 vLLM 의 `π_gen` 에서 나온다. async 라 가중치는 최대 2 스텝 전 것이고(`max_trajectory_age_steps: 2`) 엔진 수치도 다르다.

```
E_{y~π_θ}[f(y)] = E_{y~π_gen}[ (π_θ(y)/π_gen(y)) · f(y) ]
π_θ(y)/π_gen(y) = Π_t π_θ(y_t)/π_gen(y_t)                 (수천 토큰의 곱 → 분산 폭발)
```

그래서 시퀀스 비율 대신 **토큰별 비율** `w_t = π_θ(y_t)/π_gen(y_t)` 를 쓴다. 편향은 생기지만 분산이 작다.
`[0.2, 5]` 절단을 더한 것이 `tis` 다. `w_t` 는 stop-gradient 라 미분되지 않는다. 코드는 `prev_logprobs = curr_logprobs.detach()` 로 두고 `w = exp(prev − gen)` 을 만든다 (`:454`, `:663`).

**최종 기울기 추정량.** §0 의 식이다. 몬테카를로 표본은 글로벌 배치 64 프롬프트 × 16 생성 = 1,024 시퀀스다.

## 3. 코드의 surrogate

**왜 `exp(u − sg(u))` 인가.** autograd 로 §0 의 기울기를 얻으려면 미분하면 그 식이 되는 스칼라가 필요하다. 가장 단순한 것은 `−(1/G) Σ w A log π_θ` 다.
코드는 PPO 공통 경로를 재사용하려고 `ratio` 자리에 다음을 넣는다 (`:607`).

```
ratio_t = exp( log π_θ(y_t) − stopgrad(log π_θ(y_t)) )
d/dθ ratio_t = exp(0) · ∇log π_θ(y_t) = ∇log π_θ(y_t)
```

forward 값은 1 이고 기울기는 `log π_θ` 와 똑같다. `loss1 = −A·ratio` (`:632`) 와 `−A·log π` 는 **기울기가 동일** 하고 forward 값만 다르다.

**PPO 일반형과의 관계.** 일반형은 `r_t = π_θ/π_old` 다. `θ = θ_old` 인 순간 `r_t = 1` 이고 `∇r_t = ∇log π_θ` 다.
롤아웃당 옵티마이저 step 이 1 번이면 (GBS = `num_prompts × num_generations` 라 정확히 1 번) 기울기를 구하는 순간은 항상 `θ = θ_old` 라서
`r_t` 가 `[0.8, 1.28]` 밖으로 나갈 수 없다. clip 은 **결코 켜지지 않는다**. `force_on_policy_ratio` 는 이를 정확히 만들고 `prev_logprobs` 패스를 생략한다.

**로그되는 loss 값.** `masked_mean(w · (−A))` 이므로 토큰 가중 advantage 평균의 음수다. 학습 진행 지표가 아니다. 작아져도 좋아진 것이 아니다.

## 4. 토큰 하나가 요구하는 방향

위치 `t` 의 logit 을 `z_t`, `p_t = softmax(z_t)` 라 하면

```
∂ log p_t[y_t] / ∂ z_t = e_{y_t} − p_t
∂L / ∂ z_t = −(w_t · A_i / G) · (e_{y_t} − p_t)
경사하강 Δz_t = −η ∂L/∂z_t = +η (w_t A_i / G) · (e_{y_t} − p_t)
```

| 대상 | `A > 0` | `A < 0` |
|---|---|---|
| 샘플된 토큰 `y_t` | `+η wA(1−p_y)/G` 올림 | 그만큼 내림 |
| 다른 토큰 `k` | `−η wA p_k/G` 내림 | `p_k` 비례로 올림 |

- `(1−p_y)` 인자 때문에 이미 확률이 1 에 가까운 토큰은 거의 움직이지 않는다.
- `A < 0` 일 때 빠진 확률 질량은 **이미 확률이 큰 대안 토큰** 에 비례 배분된다.
- 이 `∂L/∂z_t` 가 lm_head 에서 24층 전체로 역전파된다. 파라미터 공간의 실제 이동 방향은 Muon 이 다시 바꾼다. 위 표는 "한 토큰이 요구하는 방향" 이다.

### 4.1 "그룹 안에서 상대적으로 좋은 토큰은 올리고 나쁜 토큰은 내린다" 의 수식 (2026-10-15 추가)

GRPO 는 샘플 안의 토큰을 좋고 나쁘다고 가르지 않는다. 한 샘플의 모든 토큰은 같은 `A_i` 로 한 방향으로 밀린다. "상대적" 은 세 층 중 3층에서만 나온다.

| 층 | 식 | 뜻 |
|---|---|---|
| 1. 샘플 하나 | `g_i = A_i · Σ_t ∇log π_θ(y_{i,t} | x, y_{i,<t})` | `A_i > 0` 이면 `y_i` 전체에 대한 SFT 기울기 × `A_i`. 나쁜 토큰이 섞여 있어도 같이 올라간다. `A_i < 0` 이면 전체를 낮춘다 |
| 2. 한 위치 안 | `∂log p[y]/∂z = e_y − p` | `y` 를 올리면 나머지는 `p_k` 비례로 내려간다. 확률 합 1 제약이지 credit assignment 가 아니다 |
| 3. 그룹 합 | `g = Σ_i A_i s_i`, `s_i = ∇log π_θ(y_i)`. `Σ_i A_i = 0` 이면 `s_i = s̄ + d_i` 에 대해 `g = Σ_i A_i d_i` | **모든 샘플에 공통인 기울기 성분은 사라지고 샘플 간 편차만 남는다** |

**2 샘플 예.** `y_1 = [A B C]` 보상 1, `y_2 = [A B D]` 보상 0. leave-one-out 이면 `A_1 = +1`, `A_2 = −1`.

| 위치 | 문맥 | logit 기울기 합 |
|---|---|---|
| 1, 2 | 같은 문맥, 같은 토큰 | `(+1)(e_A − p) + (−1)(e_A − p) = 0` |
| 3 | 같은 문맥 `x A B` | `(+1)(e_C − p) + (−1)(e_D − p) = e_C − e_D` |

```
g = ∇log π(C | x A B) − ∇log π(D | x A B) = ∇ log [ π(C | ·) / π(D | ·) ]
일반 2 샘플:  g = (r_1 − r_2) · [ ∇log π(y_1) − ∇log π(y_2) ]
```

공통 접두는 정확히 상쇄되고 분기점의 log-odds 만 움직인다. DPO 의 쌍 비교와 같은 꼴이고 시그모이드만 없다.

**16 샘플.** 여러 샘플이 공유하는 문맥 `c` 에서 토큰 `v` 가 받는 밀기는 `push(v | c) = Σ_{i: c 공유, y_{i,t}=v} A_i` 다.
좋은 샘플들이 고른 토큰은 위, 나쁜 샘플들이 고른 토큰은 아래, 둘 다 고른 토큰은 순합이다. 이 상대성은 **그룹 합에서만** 생긴다.

**분기 뒤의 토큰.** 그 샘플만 가진 문맥이라 1층 그대로다. 토큰 단위 credit 은 공식에 없고 통계적으로만 생긴다.
성공·실패 양쪽에 나오는 패턴은 여러 샘플·스텝에 걸쳐 ± 가 상쇄되고, 성공에만 상관된 패턴이 누적된다.
프롬프트당 16 샘플이 필요한 이유이고, PPO critic·GAE·PRM 이 하려는 토큰 credit 을 GRPO 는 의도적으로 포기한 것이다.

**현 레시피에서 어긋나는 두 가지.** (1) std 정규화 뒤에는 `Σ A_i ≠ 0` 이라 (§5 표, k=1 이면 −2.87) 공통 접두인 형식 토큰도 작은 순 밀기를 받는다.
(2) 토큰 단위 `1/G` 라 긴 샘플의 `d_i` 가 더 무겁다 (§7).

## 5. advantage 계산 (기울기 없음)

`advantage_estimator.py:57` `GRPOAdvantageEstimator` 와 `utils.py:104` `calculate_baseline_and_std_per_prompt`. 모두 학습 forward 전에 `no_grad` 로 끝난다.

1. 보상 `r_i`. codemath teacher 는 DAPO 길이 감점 뒤 값이라 `[−1, 1]` 이다 (결정 36, 학습 스텝의 `apply_async_reward_shaping`).
2. baseline `b_i` = 같은 프롬프트 16개 중 **자기를 뺀** 15개 평균 (`use_leave_one_out_baseline: true`).
3. std 도 자기를 뺀 15개로, Bessel 보정 `15/14` 포함.
4. `A_i = (r_i − b_i) / (std_i + 1e-6)` (`normalize_rewards: true`, `:92-99`). 나머지 15개 보상이 전부 같으면(trivial) 정규화를 건너뛴다.
5. `[−20, 20]` clamp (`grpo.py:2414`).
6. 시퀀스 `i` 의 **모든 응답 토큰에 같은 `A_i`** 복제 (`expand(mask.shape)`).

**수치.** 16개 그룹, 0/1 보상, 성공 `k` 개일 때 위 식을 그대로 재현한 값이다.

| k (성공 수) | A(성공) | A(실패) | 그룹 합 |
|---|---|---|---|
| 1 | +1.000 | −0.258 | −2.87 |
| 2 | +3.615 | −0.379 | +1.92 |
| 4 | +1.932 | −0.583 | +0.74 |
| 8 | +1.033 | −1.033 | 0.00 |
| 12 | +0.583 | −1.932 | −0.74 |
| 15 | +0.258 | −1.000 | +2.87 |

- **그룹 합이 0 이 아니다.** 정규화 전 `r − b` 는 그룹 합이 정확히 0 이지만, std 가 샘플마다 다르게 계산되므로 정규화 뒤에는 깨진다.
- **k=1 은 불연속이다.** 유일한 성공 샘플은 "나머지 15개가 전부 0" 이라 trivial 판정을 받아 정규화를 건너뛴다. 그래서 `+1.0` 인데 k=2 의 성공은 `+3.6` 이다. 드문 성공이 k=2 보다 약하게 보상된다.
- k=0 · k=16 은 모든 샘플이 `A = 0` 이라 그룹이 기울기에 기여하지 않는다 (zero-std 그룹, 결정 36 의 dynamic sampling 논의 대상).

재현 스크립트 (CPU, 의존성 없음):

```python
import math
n = 16
for k in [1, 2, 4, 8, 12, 15]:
    r = [1.0]*k + [0.0]*(n-k); A = []
    for i in range(n):
        o = [r[j] for j in range(n) if j != i]
        m = sum(o)/15; msq = sum(x*x for x in o)/15
        std = math.sqrt(max((msq - m*m) * (15/14), 0.0))
        a = r[i] - m
        if std > 0 and min(o) != max(o): a = a / (std + 1e-6)
        A.append(max(-20, min(20, a)))
    print(k, round(A[0], 3), round(A[-1], 3), round(sum(A), 2))
```

## 6. 집계·정규화·업데이트 순서 (Megatron 워커)

| 단계 | 내용 | 코드 |
|---|---|---|
| 마스크 | `mask = token_mask[:, 1:] × sample_mask`. `[:, 1:]` 는 위치 `t` 의 logprob 이 토큰 `t+1` 을 예측하기 때문. 프롬프트 토큰은 `token_mask` 0 | `loss_functions.py:449` |
| forward | micro batch 1 = packed 128K 토큰, CP8. `fuse_loss` 라 logits 를 2048 청크로 fp32 승격해 `log_softmax` → 샘플 토큰 gather → `curr_logprobs`. 손실 함수는 microbatch 마다 호출 | 레시피 `logprob_chunk_size`·`defer_fp32_logits` |
| in-loss 필터 | `mult_prob_error > 2` 시퀀스는 `torch.where` 로 `curr_logprobs` 를 0 으로 바꿔 기울기를 끊고 마스크도 0 | `:457-481` |
| 전역 평균 | `G` = 1,024 시퀀스의 응답 토큰 합. microbatch 마다 `/G` → `no_sync` 로 `main_grad` 누적 → DP all-reduce 합 = 전역 토큰 평균. CP8 은 한 시퀀스를 8 랭크에 나누지만 같은 `G` 로 나누므로 합산이 맞다 | `:763`, 워커 `:1420` 부근 |
| 생존 토큰 보정 | 필터로 뺀 토큰 때문에 분모가 `G` 로 남으므로 모든 microbatch 뒤 `scale_gradients(G/K)` 로 `S/K` 복원 | 워커 `:692`, `:749` |
| clip + step | `max_grad_norm 1.0` clipping 뒤 `optimizer.step()`. Muon, teacher lr 5e-6 (결정 23) | 워커 `:978` |

롤아웃 1 번에 옵티마이저 step 1 번이다. 이것이 §3 의 "clip 은 켜지지 않는다" 의 전제다.

## 7. 토큰 가중과 DAPO 감점의 상호작용

같은 `A_i` 가 시퀀스의 모든 토큰에 복제되고 `G` 로 나누므로, 긴 시퀀스가 짧은 것보다 총 기울기에서 더 큰 몫을 차지한다.
잘린 응답은 가장 길고 DAPO 감점으로 `A < 0` 이므로, 전체 기울기의 큰 부분이 "그 토큰들의 확률 낮추기" 로 간다.
`s1r5` step 71 에서 `penalized_rate 0.681` · shaped reward 평균 `−0.477` 이 이 구도다 (`STATUS.md` 2026-10-15). 시퀀스 단위 집계(`token_level_loss: false`)였다면 길이와 무관하게 시퀀스당 1 표가 된다.

## 8. 미확인

- MoE 보조 손실(load balancing)이 켜져 있으면 같은 `1/G` 로 스케일된 별도 항이 더해진다 (`_compute_moe_grad_scale`). 이 모델의 계수 설정값은 레시피에 없어 확인하지 않았다.
- 토큰별 중요도 비율의 편향 크기는 실측하지 않았다. 관측 지표는 `sample_importance_ratio` · `is_oob_ratio` 다.

## 참고 — 관련 문서

- upstream `docs/guides/grpo.md` (importance-sampling-correction · metrics 절)
- `RL_PLAN.md` 결정 36 (DAPO 길이 감점 · dynamic sampling 보류), MiMo-V2.6 objective 비교 메모 (`e4eae3b11`)
- `GATES.md` R1·G2·G3 (logprob 정합, `mult_prob_error` 기준)
