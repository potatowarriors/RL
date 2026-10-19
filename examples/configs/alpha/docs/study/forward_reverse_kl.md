# Forward KL vs Reverse KL 스터디 — 방향이 학습 결과를 바꾼다

_작성 2026-10-19. 원전: Hassan Hamad, "Forward vs Reverse KL Divergence: Why the Direction Matters", 2025-09,
<https://www.hassanhamad.com/blog/forward-vs-reverse-kl/forward-vs-reverse-kl.html>.
§1~§6 은 원문 내용을 우리말로 요약·재구성했다. 문장 단위 번역이 아니며, 그림·코드·원문 표현은 원문에서 본다.
§3 수치는 원문 실험을 numpy 로 다시 돌려 확인했다 (부록 A). §7 은 원문에 없는 alpha·NeMo-RL 연결이다.
코드 줄 번호는 `a34930bda` 기준이다._

## 0. 결론

1. KL 은 비대칭이다. `D(P‖Q) ≠ D(Q‖P)`. P 는 맞추려는 목표(데이터·teacher·reference), Q 는 학습하는 모델(정책·학생)이다.
2. **Forward KL `D(P‖Q)`** 는 P 에서 샘플을 뽑는다. Q 는 P 의 모든 봉우리를 덮으려고 넓게 퍼진다 (mode-covering). 사전학습·SFT 의 NLL 이 이것이다.
3. **Reverse KL `D(Q‖P)`** 는 Q 에서 샘플을 뽑는다. Q 는 P 의 한 봉우리에 모인다 (mode-seeking). RLHF 목표, reference KL 규제, on-policy 증류가 이것이다.
4. reverse KL 은 초기값에 민감하다. 재현에서 원문이 다루지 않은 세 번째 국소 최소 (μ 2.0, σ 3.45) 를 확인했다 (§3).
5. 실전에서는 KL 을 샘플로 추정한다. 단순 추정량 k1 보다 Schulman k3 `(r−1) − log r` 의 분산이 작다 (§6).
6. **alpha MOPD 는 reverse KL `D(π_θ‖π_T)` 를 줄인다.** 토큰 advantage `log π_T − log π_θ` 는 k1 의 부호를 바꾼 값이다.
   학생이 샘플하지 않는 teacher 의 봉우리에는 신호가 없다 (§7).

## 1. 정의와 두 방향

```
D(P‖Q) = Σ_x P(x)·log(P(x)/Q(x)) = E_{x~P}[ log P(x) − log Q(x) ]      ← forward
D(Q‖P) = Σ_x Q(x)·log(Q(x)/P(x)) = E_{x~Q}[ log Q(x) − log P(x) ]      ← reverse
```

- KL ≥ 0 이고 P = Q 일 때만 0 이다 (Jensen 부등식으로 보인다).
- 원문이 드는 직관: P 에서 나온 샘플을 Q 에서 나왔다고 가정하고 다룰 때 생기는 비효율.

| | forward `D(P‖Q)` | reverse `D(Q‖P)` |
|---|---|---|
| 샘플 출처 | 목표 P | 학습 모델 Q |
| 계산에 필요한 것 | P 의 샘플과 `log Q` | Q 의 샘플과 `log Q`·`log P` 둘 다 |
| 큰 벌점이 나는 곳 | P > 0 인데 Q ≈ 0 (놓친 봉우리) | Q > 0 인데 P ≈ 0 (엉뚱한 곳에 둔 질량) |
| 벌점이 없는 곳 | P = 0 인 곳에 둔 Q 의 질량 | Q = 0 인 곳의 P 질량 (놓친 봉우리) |
| 결과 | 넓게 덮는다 · recall 지향 | 한 곳에 모인다 · precision 지향 |

## 2. 장난감 실험 설정 (원문)

- 목표 P: 같은 비율의 정규분포 혼합 `0.5·N(−2, 1) + 0.5·N(6, 1)`.
- 모델 Q: 단일 정규분포 `N(μ, σ²)`. μ·σ 를 학습한다.
- 두 KL 모두 몬테카를로로 추정한다. 혼합분포의 `log P` 는 log-sum-exp 로 수치 안정화한다.
- **forward**: `log P` 는 Q 의 파라미터와 무관하다. 손실은 `E_{x~P}[−log Q(x)] + 상수`, 즉 NLL 이다.
  정규분포 Q 의 해는 모멘트 일치다: μ = 2, σ² = 1 + 16 = 17 (σ = 4.123).
- **reverse**: 샘플이 Q 에서 나온다. 재매개화 `x = μ + σε, ε ~ N(0, 1)` 로 기울기를 μ·σ 에 흘린다 (Kingma & Welling 2014).
- 최적화: 초기값 μ = 0, σ = 1 에서 경사하강 1000 스텝. 학습률은 forward 0.1, reverse 0.01.

## 3. 결과 — 원문과 재현

| 설정 | 원문 | 재현 (스텝당 샘플 1000) |
|---|---|---|
| forward, 초기 (0, 1) | μ ≈ 1.99 · σ ≈ 4.12 | μ 1.990 · σ 4.124 (해석해 2.000 · 4.123) |
| reverse, 초기 (0, 1) | μ ≈ −2.00 · σ ≈ 1.00 | μ −1.998 · σ 1.002 |
| reverse, 초기 (3, 1) | — | μ 6.001 · σ 0.997 (오른쪽 봉우리) |
| reverse, 초기 (2, 1) | 어느 봉우리로도 갈 수 있다고 서술 | 20,000 스텝 뒤에도 μ 1.98 · σ 3.48. 어느 봉우리로도 가지 않는다 |

세 해에서 잰 KL (샘플 40만):

| Q | reverse KL | forward KL |
|---|---|---|
| 한 봉우리 N(−2, 1) | **0.693** (= log 2) | 15.29 |
| 가운데 N(2, 4.12²) | 2.111 | **0.723** |
| 대칭점 N(2, 3.47²) | 1.869 | 0.756 |

- **두 목적은 반대 해를 고른다.** 한 봉우리 해는 reverse 에서 가장 낮다. P 질량 절반을 버린 대가가 log 2 다.
  forward 는 같은 해에 15.3 의 벌점을 준다. +6 쪽 P 샘플에서 Q 가 거의 0 이기 때문이다.
- **대칭점은 reverse KL 의 국소 최소다.** 원문은 이 점을 다루지 않았다.
  μ 를 고정하고 σ 를 최적화한 reverse KL 은 μ 2.0 에서 1.877 로 가장 낮다. μ 1.5 에서 1.887, 2.5 에서 1.885, 4.0 에서 1.937 로 올라간다.
  한 봉우리 해(0.693)로 가려면 이 언덕을 넘어야 해서 경사하강이 머문다.
- 원문의 교훈은 그대로 맞다. reverse KL 은 초기값에 따라 다른 국소 최소로 간다. 원문은 annealing 이나 무작위 재시작을 대책으로 든다.

## 4. 왜 그렇게 되나 (원문 요지)

- **forward 의 가중치는 `P(x)` 다.** P 가 있는 곳에서 Q ≈ 0 이면 `−log Q → ∞` 다.
  그래서 Q 는 P 가 있는 곳 어디에나 질량을 둔다 (zero-avoiding). 대가로 봉우리 사이 빈 곳에도 질량이 생긴다.
- **reverse 의 가중치는 `Q(x)` 다.** Q 가 있는 곳에서 P ≈ 0 이면 `log(Q/P) → ∞` 다.
  그래서 Q 는 P 가 없는 곳을 피한다 (zero-forcing). Q 가 가지 않는 곳은 가중치가 0 이라 놓친 봉우리에 벌점이 없다.
- 원문의 요약: forward 는 행동 복제·커버리지(보수적, recall), reverse 는 특화·정밀(확신, precision).

## 5. LLM 학습에서 어느 방향인가 (원문 퀴즈 3개와 사례)

| 상황 | 방향 | 이유 |
|---|---|---|
| 사전학습·SFT | forward `D(P_data‖Q_θ)` | 표준 손실 `E_{x~P_data}[−log Q_θ(x)]` 가 forward KL + 상수다. 데이터 패턴을 놓치는 비용이 크다. reverse 는 `P_data(x)` 값을 계산할 수 없어 쓸 수 없다 |
| RLHF 목표 | reverse `D(Q_θ‖P*)`, `P* ∝ exp(R/β)` | 아래 식 |
| RLHF 규제 | reverse `D(Q_θ‖Q_ref)` | 고보상 응답으로 좁아지는 것은 허용하고, reference 가 질량을 두지 않은 곳으로 가는 것은 막는다. forward 였다면 reference 의 모든 행동을 덮어야 해서 특화를 막는다 |
| 언러닝 (Wang et al., NAACL 2025) | reverse KL 증류 | teacher 는 지울 사실(예: 주민등록번호에 해당하는 SSN)의 토큰을 깎은 분포를 만든다. 학생이 그 토큰에 확률을 주면 reverse KL 이 크게 벌한다. forward 는 더 관대해 누출 여지가 남는다 |

RLHF 목표가 reverse KL 인 이유:

```
log P*(x) = R(x)/β − log Z
D(Q_θ‖P*) = E_{Q_θ}[log Q_θ − R/β] + log Z = −(1/β)·E_{Q_θ}[R] − H(Q_θ) + log Z
⇒  max  E_{Q_θ}[R] + β·H(Q_θ)   ≡   min  D(Q_θ‖P*)
```

보충 (원문에 없음): `P* ∝ Q_ref·exp(R/β)` 로 두면 같은 계산에서 `max E[R] − β·D(Q_θ‖Q_ref)` 가 나온다. KL 규제 RLHF 의 표준 목표다.

원문은 VAE·지식 증류·OOD 탐지에서는 두 방향 모두 쓰일 수 있다고 덧붙인다.

## 6. KL 추정 (원문 마지막 절)

DeepSeek-R1 의 KL 항은 교과서 정의와 다르게 생겼다. 실전의 KL 은 모두 샘플 추정량이기 때문이다.
Schulman 의 정리(<http://joschu.net/blog/kl-approx.html>)에서 x ~ Q, `r = P(x)/Q(x)` 로 두면 둘 다 `D(Q‖P)` 를 추정한다.

| 추정량 | 식 | 성질 |
|---|---|---|
| k1 | `−log r = log Q(x) − log P(x)` | 불편 추정량. 샘플 하나의 값은 음수가 될 수 있고 분산이 크다 |
| k3 | `(r − 1) − log r` | 불편 추정량. 항상 ≥ 0 이고 분산이 작다. DeepSeek GRPO 의 KL 항이 이 형태다 |

## 7. alpha·NeMo-RL 에 대응시키면 (원문에 없음)

| 쓰임 | 방향 | NeMo-RL 코드 | alpha 현재 |
|---|---|---|---|
| SFT (Pai 스택) | forward `D(P_data‖π_θ)` | — | agentic SFT iter2862 까지 완료 |
| GRPO reference KL | reverse `D(π_θ‖π_ref)`, k3 | `nemo_rl/algorithms/utils.py:72` (`logr = ref − logprobs`) · `:89-90` (k3) | 꺼짐. `reference_policy_kl_penalty: 0.0` ([grpo_objective_gradient.md](grpo_objective_gradient.md) §1) |
| MOPD | reverse `D(π_θ‖π_T)`, 토큰당 k1 | `nemo_rl/algorithms/advantage_estimator.py:611` | MOPD1 레시피 미작성 |
| top-k logit 증류 (`distillation.py`) | forward · reverse · mixed 중 선택 | `nemo_rl/algorithms/loss/loss_functions.py:1250-1264` | 쓰지 않는다. 예제 기본값은 `mixed`, forward 가중 0.5 (`examples/configs/distillation_math.yaml:17-18`) |

MOPD 에 대한 함의:

1. **reverse KL 을 쓸 수 있는 이유는 teacher logprob 을 계산할 수 있어서다.** SFT 데이터와 달리 teacher 는 학생이 낸 토큰마다 `log π_T(y_t)` 를 준다.
2. **advantage 는 토큰당 k1 의 부호를 바꾼 값이다.** `Â_t = log π_T(y_t) − log π_θ(y_t) = −k1`.
   샘플 하나의 값이라 크게 흔들린다. Ultra 레시피는 ±20 으로 자른다 (`examples/nemo_gym/nemotron-3-ultra/mopd.yaml` `advantage_clip_low/high`).
3. **zero-forcing 이 강하게 걸린다.** teacher 가 거의 0 으로 보는 토큰을 학생이 내면 `Â_t` 가 큰 음수가 되어 그 토큰이 강하게 눌린다.
   원문의 언러닝 사례와 같은 작동이다.
4. **놓친 봉우리에는 신호가 없다.** 신호는 학생이 실제로 샘플한 토큰에만 생긴다.
   teacher 가 잘하지만 학생이 한 번도 내지 않는 풀이 방식은 MOPD 로 직접 옮겨지지 않는다.
5. **다양성이 줄 수 있다.** 학생은 teacher 의 주된 봉우리로 모인다. teacher 자체도 RLVR 로 이미 좁아졌다
   (도구 사용 teacher: pass@16 0.91 → 0.77, entropy 0.765 → 0.672, `STATUS.md` 2026-10-10).
   MOPD1 설계 때 entropy·pass@k 를 감시 지표 후보로 검토한다.

## 부록 A. 재현 스크립트

CPU 에서 numpy 만으로 돈다. 원문과 같은 설정이며, forward 는 해석적 NLL 기울기, reverse 는 재매개화 기울기를 쓴다.

```python
import numpy as np

rng = np.random.default_rng(0)
MUS, SIG, PI = np.array([-2.0, 6.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5])


def log_p(x):  # 혼합분포 log P(x) 와 score d log P / dx
    a = np.log(PI) - 0.5 * np.log(2 * np.pi) - np.log(SIG) - (x[:, None] - MUS) ** 2 / (2 * SIG**2)
    m = a.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(a - m).sum(1))
    resp = np.exp(a - lse[:, None])
    return lse, (resp * (-(x[:, None] - MUS) / SIG**2)).sum(1)


def sample_p(n):
    z = rng.choice(2, size=n, p=PI)
    return MUS[z] + SIG[z] * rng.standard_normal(n)


def forward(mu, sig, lr, steps=1000, n=1000):  # min E_P[-log Q]
    for _ in range(steps):
        x = sample_p(n)
        g_mu, g_sig = -np.mean((x - mu) / sig**2), np.mean(1 / sig - (x - mu) ** 2 / sig**3)
        mu, sig = mu - lr * g_mu, max(sig - lr * g_sig, 1e-3)
    return mu, sig


def reverse(mu, sig, lr, steps=1000, n=1000):  # min E_Q[log Q - log P], x = mu + sig*eps
    for _ in range(steps):
        eps = rng.standard_normal(n)
        _, s = log_p(mu + sig * eps)
        g_mu, g_sig = np.mean(-s), -1 / sig + np.mean(-s * eps)
        mu, sig = mu - lr * g_mu, max(sig - lr * g_sig, 1e-3)
    return mu, sig


print(forward(0.0, 1.0, 0.1))   # (1.990, 4.124)
print(reverse(0.0, 1.0, 0.01))  # (-1.998, 1.002)
print(reverse(3.0, 1.0, 0.01))  # (6.001, 0.997)
print(reverse(2.0, 1.0, 0.01))  # (2.001, 3.467) — 20,000 스텝에서도 (1.98, 3.48)
```

§3 의 국소 최소 확인은 공통 난수 ε 200만 개로 `E_ε[log Q(μ+σε) − log P(μ+σε)]` 를 계산했다.
μ ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} 마다 σ ∈ [0.8, 4.5] 75점 격자에서 최소를 취했다.
