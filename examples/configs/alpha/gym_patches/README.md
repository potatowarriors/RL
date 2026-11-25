# Gym 패치 — 고정한 Gym (v0.6.0 `3045a793`) 위에 얹는 upstream 수정

Gym 서브모듈은 v0.6.0 에 고정한다 (`.claude/rules/alpha-submodules.md`). 그 버전에 없고 우리 경로에 필요한 upstream 수정은
**upstream 커밋을 그대로** 여기에 패치 파일로 두고 두 체크아웃에 적용한다 (`RL_PLAN.md` 결정 79, 사용자 2026-11-25).

| 패치 | upstream 커밋 | 왜 필요한가 |
|---|---|---|
| `0001-…normalize-response-tools-for-replay-3190.patch` | `7bfccecc1` (2026-09-09, #3190) | `swe_agents` 가 롤아웃 결과를 조립할 때 모델 응답에서 되돌려 쓴 도구 정의에 `defer_loading: None` 이 들어가 `SWEBenchVerifyResponse` 검증이 실패한다 (`/run` 500, `KNOWN_ISSUES.md` 2026-11-25) |

- 적용 · 확인: `examples/configs/alpha/tools/apply_gym_patches.sh` (인자 없이 = 적용, `check` = 확인만). 대상은 NeMo-RL 서브모듈
  `3rdparty/Gym-workspace/Gym` (학습) 과 측정용 체크아웃 `/home/work/vidsearch/tools/Gym`. 두 곳 모두 NFS 라 세션을 다시 만들어도 남는다 — 새로 클론했을 때만 다시 적용한다.
- 적용하면 서브모듈이 superproject 의 `git status` 에 수정됨으로 보인다. 포인터는 그대로 `3045a793` 이다 — 서브모듈 안에서 커밋하지 않는다.
- 요청 검증 쪽 수정 (upstream #4011 · #4483: `aws_region_name` · 도구 결과 `name` · `provider_specific_fields`) 은 여기 두지 않는다.
  모델 서버는 우리 플러그인 (`gym_plugins/responses_api_models/alpha_vllm_model`) 이고 그 서버 venv 는 Gym 라이브러리를 공개 패키지로 받기 때문에
  Gym 소스를 고쳐도 닿지 않는다 → 같은 동작을 플러그인이 한다.
- Gym 을 upstream 쪽으로 올리지 않는 이유: 세 수정이 모두 든 Gym 은 자기 쪽 vLLM 을 0.29.0 으로 고정하고 (torchvision 0.28 · transformers ≥ 5.10.4 · Ray ≥ 2.58),
  우리 NeMo-RL 의 고정 (torch 2.11 · torchvision 0.26 · transformers < 5.9 · Ray 2.56.1) 과 한 잠금 파일에서 풀리지 않는다 (2026-11-25 사본에서 `uv lock` 실패 확인).
- fork 를 upstream 에 맞춰 올리면 이 디렉토리는 비운다.
