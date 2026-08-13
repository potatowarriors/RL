I have full coverage. Here is the spec.

---

# NeMo-RL Reward Environments → GRPO: implementation spec for alpha RLVR

Repo: `/home/work/vidsearch/repos/project_s/NeMo-RL` (branch `alpha/post-train`). Note: the Ultra recipe **exists locally** at `examples/nemo_gym/nemotron-3-ultra/` — no GitHub fetch was needed (the task's `examples/configs/ultra/` path is the old `ultra-v3` layout; this branch has it under `examples/nemo_gym/nemotron-3-ultra/`).

## 1. Environment interface — how GRPO scores rollouts

### 1.1 The contract

`/home/work/vidsearch/repos/project_s/NeMo-RL/nemo_rl/environments/interfaces.py:54-90` — `EnvironmentInterface[MetadataT]`, two abstract methods:

```python
def step(self, message_log_batch: list[LLMMessageLogType],
         metadata: list[MetadataT]) -> EnvironmentReturn[MetadataT]
def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]
```

`EnvironmentReturn` (`interfaces.py:26-51`) is a 6-field NamedTuple, **all batched**: `observations` (list of `{"role","content"}`), `metadata`, `next_stop_strings`, `rewards` (`Tensor[B]`, or `dict[str, Tensor[B]]` for multi-reward/GDPO), `terminateds` (`Tensor[B]`), `answers`.

### 1.2 Dispatch: task_name → env actor

`/home/work/vidsearch/repos/project_s/NeMo-RL/nemo_rl/experience/rollouts.py:668-800` `calculate_rewards()`:
- strips message logs to `["role","content"]` only (`:687-690`)
- groups batch indices by `batch["task_name"]` (`:694-698`)
- for each group: `task_to_env[task_name].step.remote(messages, env_info)` where `env_info = batch["extra_env_info"][i]` (`:712-715`) — one Ray call per task group, all in flight concurrently
- reassembles by original index (`:767-791`)
- **hard rule** (`:773-777`): you cannot mix dict-rewards and scalar-rewards envs in one batch.

Called from the multi-turn rollout loop at `rollouts.py:921`; `task_to_env` threads down from `run_grpo.py:126-128` → `grpo_train`.

### 1.3 Env registry & Ray spawn

`/home/work/vidsearch/repos/project_s/NeMo-RL/nemo_rl/environments/utils.py:31-56` — `ENV_REGISTRY`, keys: `math_default`, `math`, `math_multi_reward`, `code`, `reward_model`, `code_jaccard`, `vlm`, `nemo_gym`.

`create_env()` (`utils.py:106-132`) resolves the FQN, looks up a per-actor python env via `get_actor_python_env`, builds an isolated uv venv per node if needed (`create_local_venv_on_each_node`), then `actor_class.options(runtime_env=...).remote(env_config)`. `register_env(name, fqn)` at `utils.py:135-139` adds custom envs without touching nemo_rl (docs: `docs/guides/environments.md:189-268`).

### 1.4 Math env internals

`/home/work/vidsearch/repos/project_s/NeMo-RL/nemo_rl/environments/math_environment.py`:
- `MathEnvConfig` (`:43-48`): `num_workers`, `stop_strings?`, `verifier_type?`, `math_verify_impl?`
- `BaseMathEnvironment.__init__` (`:354-369`): spawns `num_workers` **nested** Ray actors (`worker_cls.options(runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}).remote()`). `verifier_type` selects the class from `WORKER_CLASS_DICT` (`:425-429`: `math`→`HFVerifyWorker`, `english_multichoice`, `multilingual_multichoice`).
- The env actor itself: `@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)` (`:420-422`).
- `step()` (`:431-524`): concatenates all `assistant` message contents into one string per sample (`:453-460`), pulls `metadata[i]["ground_truth"]` (`:462`), chunks across workers via `chunk_list_to_workers` (`:464-467`) with round-robin start index (`:469-472`), `ray.get(futures)`, emits reward 1.0/0.0, `observations = "Environment: correct|incorrect"`, `terminateds = ones` (single-turn).
- `math_verify_impl` (`:100-114` in `HFVerifyWorker.verify`): `"hf_math_verify"` wraps GT as `"\\boxed{" + ground_truth + "}"` and calls HF `math_verify.math_metric(gold=LatexExtractionConfig, pred=(Expr,Latex))`; `"dapo_math_verify"` calls the verl-derived `dapo_math_verify`. Any exception (incl. `TimeoutException`) → score 0.0 (`:135-137`).
- `global_post_process_and_metrics` (`:376-417`): **zeroes rewards for non-`is_end` sequences** (`:393`), emits `accuracy`, `pass@samples_per_prompt`, `fraction_of_samples_properly_ended`, lengths.

**Config for the math env** (`examples/configs/grpo_math_1B.yaml:472-475`):
```yaml
env:
  math:
    num_workers: 8
    math_verify_impl: "hf_math_verify"
```

## 2. Data pipeline

### 2.1 Entry: `setup_response_data`

`/home/work/vidsearch/repos/project_s/NeMo-RL/nemo_rl/data/utils.py:107-289`. Called from `examples/run_grpo.py:126-128` with `(tokenizer, config.data, config.env)`.

Flow:
1. `extract_necessary_env_names(data_config)` (`nemo_rl/data/datasets/utils.py:291-310`) — scans `train`/`validation`/`default` for `env_name`; only those envs are constructed (`utils.py:154-160`).
2. Normalizes `data.train` dict → single-element list (`:171-172`), applies `data.default` as fill-in defaults per dataset (`update_single_dataset_config`, `datasets/utils.py:263-267` — only fills **missing** keys).
3. `load_response_dataset(cfg)` per dataset (`:178`).
4. Binds `task_data_processors[task_name] = (task_spec, processor)` and `task_to_env[task_name] = envs[cfg["env_name"]]` (`:185-189`). **`task_name` is derived by the dataset class, not the config.**
5. Merge: `use_multiple_dataloader=true` → dict of per-task `AllTaskProcessedDataset` (`:196-207`); else one concatenated dataset (`:210-218`).
6. Validation: from `split_validation_size` on a train dataset (`:231-245`) and/or from `data.validation` (`:248-270`).

### 2.2 Dataset loading

`nemo_rl/data/datasets/response_datasets/__init__.py:98-141` `load_response_dataset`, 3-way resolution:
1. name in `DATASET_REGISTRY` (`:62-95`, ~30 built-ins incl. `OpenMathInstruct-2`, `DeepScaler`, `AIME2024/25/26`, `openai_format`, `NemoGymDataset`, `ResponseDataset`)
2. name contains `.` → dotted import path, resolved by `resolve_external_dataset_class` (`datasets/utils.py:147+`) — **this is how you add a custom dataset class without editing nemo_rl**
3. else `ValueError`

Then `dataset.set_task_spec(cfg)` + `dataset.set_processor()` (`:137-139`).

**Local files**: `ResponseDataset` (`response_datasets/response_dataset.py:21-85`) → `load_dataset_from_path` (`datasets/utils.py:97-144`) supports by extension `.arrow .csv .json .jsonl .parquet .txt`; otherwise tries HF hub, then `load_from_disk`. `ResponseDataset` maps `{input_key, output_key}` → `{"messages":[user,assistant], "task_name": ...}` (`:78-85`), and **passes through unchanged if the file already has a `messages` column** (`:69-72`, adds `task_name`). `task_name` = last-two path segments joined with `-`, extension stripped (`:56-58`).

`RawDataset` base (`nemo_rl/data/datasets/raw_dataset.py:26-72`): required attrs `dataset`, `task_name`; provides `split_train_validation`, `set_processor` (looks up `data_config["processor"]` in `PROCESSOR_REGISTRY`, default `"default"`), `set_task_spec` (builds `TaskDataSpec` from `prompt_file`/`system_prompt_file`).

### 2.3 Processor contract

`nemo_rl/data/interfaces.py:92-103`:
```python
def my_processor(datum_dict: dict[str, Any],      # one raw dataset row
                 task_data_spec: TaskDataSpec,    # .prompt / .system_prompt (loaded from files)
                 tokenizer: TokenizerType,
                 max_seq_length: int | None,
                 idx: int) -> DatumSpec
```
Invoked per item at `nemo_rl/data/datasets/processed_dataset.py:99-147` (`AllTaskProcessedDataset.__getitem__`); the processor is chosen by `entry["task_name"]` (`:117-126`), so **every raw row must carry `task_name`**. An optional `preprocessor` runs first (`:104-114`).

`DatumSpec` (`nemo_rl/data/interfaces.py:35-43`) — required keys:

| key | type | notes |
|---|---|---|
| `message_log` | `list[{role, content, token_ids}]` | prompt turns; `token_ids` is a `torch.Tensor` |
| `length` | int | sum of `len(token_ids)` |
| `extra_env_info` | `dict\|None` | **this is what `env.step()` receives as `metadata`** |
| `loss_multiplier` | float | set `0.0` to mask an invalid/overlong sample |
| `idx` | int | |
| `task_name` | str (NotRequired but *de facto* required) | routes to env + processor |
| `stop_strings` | `list[str]` (optional) | |

Reference impl `math_hf_data_processor` (`nemo_rl/data/processors.py:382-443`): reads `datum_dict["messages"]`, `problem = messages[0]["content"]`, **`extra_env_info = {"ground_truth": messages[1]["content"]}`** (`:392`) — this is the key `MathEnvironment.step` reads at `math_environment.py:462`. Prepends `system_prompt`, applies `task_data_spec.prompt.format(problem)`, `apply_chat_template(..., add_generation_prompt=True)`, tokenizes, and **truncates + sets `loss_multiplier=0.0` when `length >= max_seq_length`** (`:426-433`).

Registry: `PROCESSOR_REGISTRY` (`processors.py:839-852`); add yours with `register_processor(name, fn)` (`:855-862`) **before** `setup_response_data` runs (i.e. at the top of your launcher script).

### 2.4 Adding a custom local-jsonl RLVR task (minimal recipe)

Two options — pick (A) if your jsonl is already `{input, output}` or `{messages}`; pick (B) if you need custom fields (e.g. `expected_answer`, `test_cases`).

**(A) no code**, reuse `ResponseDataset` + `math_hf_data_processor`:
```yaml
data:
  train:   {data_path: /data/alpha/rlvr_train.jsonl, input_key: question, output_key: answer}
  validation: {data_path: /data/alpha/rlvr_val.jsonl}
  default:
    dataset_name: ResponseDataset
    processor: "math_hf_data_processor"
    env_name: "math"
    prompt_file: "examples/prompts/cot.txt"
    system_prompt_file: null
```
`ResponseDataset.format_data` puts the answer in `messages[1]["content"]`, which `math_hf_data_processor` lifts into `extra_env_info["ground_truth"]`.

**(B) custom processor** — in your launcher, before `setup_response_data`:
```python
from nemo_rl.data.processors import register_processor
def alpha_rlvr_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx):
    ...  # build message_log; extra_env_info = {"ground_truth": datum_dict["expected_answer"]}
register_processor("alpha_rlvr_processor", alpha_rlvr_processor)
```
then `data.default.processor: alpha_rlvr_processor`. For a custom *dataset class*, set `dataset_name: my_pkg.my_module.MyDataset` (dotted path; must subclass/duck-type `RawDataset`, accept `**kwargs`, set `self.dataset` + `self.task_name`) — `docs/guides/grpo.md:127-149`.

## 3. NeMo Gym integration

### 3.1 Architecture (`docs/design-docs/nemo-gym-integration.md`)

Gym is **CPU-only for the policy** — it runs no inference engine (`:7`, `:131-140`). NeMo-RL exposes vLLM as an OpenAI-compatible HTTP server; Gym's Model Server is a pure proxy (`/v1/responses` → NeMo-RL `/v1/chat/completions`). Data-parallel vLLM workers each get their own HTTP server and the Model Server load-balances (`:140`). Server types: **Agent** (orchestrates rollout), **Model** (proxy), **Resource** (tools + reward) (`:202-205`).

### 3.2 The NemoGym actor — *not* a normal environment

`nemo_rl/environments/nemo_gym.py:352-355`: `@ray.remote(max_restarts=-1, max_task_retries=-1) class NemoGym(EnvironmentInterface)`. Crucially:

- **`step()` raises `NotImplementedError`** (`:820-822`) and so does `global_post_process_and_metrics` (`:824-826`). Gym bypasses `calculate_rewards()` entirely.
- The real entrypoint is `async def run_rollouts(...)` (`:472-551`), a **Ray streaming generator** yielding `(row_index, result, timing)` as rollouts complete.
- `_spinup()` (`:377-470`) is deferred from `__init__`: picks a free port in `[port_range_low, port_range_high)`, injects `policy_model_name` / `policy_api_key="dummy_key"` / **`policy_base_url = base_urls`** (list, one per DP rank) into Gym's global config (`:406-410`), sets `default_host` to the node IP for multinode (`:414`), passes `ray_head_node_address = GCS address` so Gym servers **join the same Ray cluster** (`:446`), then `RunHelper().start(...)` spawns the Gym servers as OS subprocesses.
- Called from `spinup_nemo_gym_actor()` (`:923-1005`) — pops NeMo-RL-only keys (`invalid_tool_call_patterns`, `thinking_tags`, `tokenizer_config`) out of the dict that gets forwarded to Gym (`:957-959`), reuses image-baked venvs via `NEMO_GYM_UV_CACHE_DIR`/`NEMO_GYM_VENV_DIR` (`:963-968`), applies soft `NodeAffinitySchedulingStrategy` when `num_gpu_nodes > 0` (`:989-993`), and awaits `_spinup` (`:1004`).
- `setup_nemo_gym_config(config, tokenizer)` (`:904-921`) **force-sets** `vllm_cfg.async_engine=True`, `expose_http_server=True`, and **nulls `stop_strings` / `stop_token_ids`**.
- `_should_use_nemo_gym` (`nemo_rl/algorithms/grpo.py:2108-2142`) asserts async rollouts + `expose_http_server` at startup.

### 3.3 Gym rollout row schema (per-sample env routing)

`nemo_rl/experience/rollouts.py:2300` — the Gym rows *are* `input_batch["extra_env_info"]`, i.e. each jsonl line verbatim. `_prepare_nemo_gym_rows` (`:2185-2207`) requires `row["responses_create_params"]` to be a dict, overwrites `temperature`/`top_p` from sampling params, clamps `max_output_tokens = min(row_value, generation.max_new_tokens)`, and stamps `row["_rowidx"]`.

The env actor is fetched as `task_to_env["nemo_gym"]` (`rollouts.py:2365`) — a **single** actor for all samples. Per-sample routing is **inside Gym**, via `row["agent_ref"]["name"]` (`nemo_gym.py:488`, `:536`; Gym side `3rdparty/Gym-workspace/Gym/nemo_gym/rollout_collection.py:601-604`, `:1014` → `POST /run` to that named agent server). Missing `agent_ref` is a hard error (`rollout_collection.py:652-655`).

A row therefore looks like:
```jsonc
{"responses_create_params": {"input": [{"role":"user","content":"..."}], "tools": [...]},
 "agent_ref": {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"},
 "ground_truth": "...", "id": 0}
```
(verified: `3rdparty/Gym-workspace/Gym/resources_servers/aalcr/data/example_rollouts.jsonl` carries `agent_ref = {"type":"responses_api_agents","name":"aalcr_benchmark_simple_agent"}`). The agent's Gym config binds it to a resources server + model server + datasets — see `3rdparty/Gym-workspace/Gym/resources_servers/math_with_judge/configs/math_with_judge.yaml` (the `math_with_judge_simple_agent` block).

Data side: `NemoGymDataset` (`nemo_rl/data/datasets/response_datasets/nemogym_dataset.py:20-48`) reads the jsonl **as raw strings** (`:35-36`) because HF `Dataset` mangles nested structures; supports `repeat: N`. `nemo_gym_data_processor` (`nemo_rl/data/processors.py:780-803`) does `json.loads` into `extra_env_info` and emits **placeholder** `message_log`/`length` — no tokenization client-side, since Gym builds the cumulative prompt server-side.

Rewards come back with the rollout: scalar `reward`, plus optional `reward_components` for multi-reward/GDPO (`nemo_gym.py:829-843`, `846-869`, and `872-896` which enforces `reward == sum(components)`).

### 3.4 Available Gym envs

`3rdparty/Gym-workspace/Gym/resources_servers/` has **103** servers. Relevant to RLVR: `math_with_judge`, `math_with_code`, `math_with_autograder`, `math_advanced_calculations`, `math_formal_lean`, `code_gen`, `bigcodebench`, `evalplus`, `scicode`, `competitive_coding_challenges`, `ifbench`, `iheval`, `instruction_following`, `inverse_if`, `verifif`, `mcqa`, `gpqa_diamond`, `structured_outputs`, `format_verification`, `reasoning_gym`, `nvarc`, `single_step_tool_use_with_argument_comparison`, `ns_tools`, `toolsandbox`, `equivalence_llm_judge`, `genrm_compare`, `terminus_judge`, `swe_pivot`, `swerl_gen`, plus 34 agents in `responses_api_agents/` (`simple_agent`, `mini_swe_agent`, `hermes_agent`, …) and 7 model servers in `responses_api_models/` (`vllm_model`, `local_vllm_model`, `genrm_model`, …).

### 3.5 Runtime infrastructure requirements

| requirement | where |
|---|---|
| `async_engine: true` + `expose_http_server: true` on vLLM | forced by `nemo_gym.py:908-909`, asserted `grpo.py:2119-2140` |
| no `stop_strings`/`stop_token_ids`/`top_k`/`greedy`/`max_rollout_turns` | `nemo_gym.py:912-913`; `rollouts.py:2304-2337` |
| Gym servers are **OS subprocesses of the actor**, joined to the same Ray cluster; same Ray + Python version | `nemo_gym.py:389-463`; doc `:52-54` |
| Port range 5000-5999 (NeMo-RL 3000-4999, vLLM 7000-8999) | `nemo_rl/distributed/virtual_cluster.py:91`, `99-100` |
| Per-server uv venvs — **prefetch or bake into image**, else first-step stall | `examples/nemo_gym/prefetch_venvs.py`, `prefetch_ultra_all_envs.yaml`; `skip_venv_if_present: true` |
| Extra GPU nodes for judge models (only if you use LLM-judge servers) | `env.nemo_gym.num_gpu_nodes` (`nemo_gym.py:989-993`) — these are **outside** `cluster.num_nodes` |
| Colocated / small-scale | fine: `num_gpu_nodes: 0` (omit) when no judge servers. `examples/nemo_gym/nemotron-3-super/small_scale/README.md` documents `cluster.num_nodes` covering only train+gen and `SBATCH_NUM_NODES` adding Gym nodes (e.g. 20 + 7) |

**Entrypoint caveat (important):** use `examples/nemo_gym/run_grpo_nemo_gym.py`, **not** `examples/run_grpo.py`, for Gym runs. The Gym runner calls `setup_response_data(..., env_configs=None)` (`run_grpo_nemo_gym.py:194-196`) so no env is built from the data config, and after `setup()` binds `task_to_env = {"nemo_gym": nemo_gym}` (`:263-265`) — the literal key `"nemo_gym"` that `rollouts.py:2365` looks up. `run_grpo.py:126-128` passes `config.env`, which would build a second, never-spun-up `NemoGym` actor keyed by the dataset's `task_name`, and the lookup would fail.

## 4. Ultra recipe env/data shape

`examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml` (stages: `student_rlvr1/2`, `ifbench_teacher`, `reasoning_teacher`, `rlhf_teacher`, `swe_teacher`, `mopd`; launcher `ultra_launch.sh`).

**Data (`:360-384`) — one jsonl, no env list:**
```yaml
data:
  max_input_seq_length: null
  shuffle: false
  num_workers: 1
  use_multiple_dataloader: false
  train:      {data_path: null}    # ← launcher sets TRAIN_PATH
  validation: {data_path: null}    # ← launcher sets VAL_PATH
  default:
    dataset_name: NemoGymDataset
    env_name: "nemo_gym"
    processor: "nemo_gym_data_processor"
    prompt_file: null
    system_prompt_file: null
```

**Env (`:385-...`):** `should_use_nemo_gym: true`, `should_log_nemo_gym_responses: true`, then `nemo_gym:` with `nemo_gym_log_dir`, `skip_venv_if_present: true`, `num_gpu_nodes: 20`, `port_range_low/high: 5000/5999`, `invalid_tool_call_patterns`, `thinking_tags`, NeMo-RL-only `effort_levels` (stripped before forwarding — `nemo_gym.py:404`), and **`config_paths:` with 33 entries** (`:408+`) — one `responses_api_models/vllm_model/configs/vllm_model_for_training.yaml` (required, must be `*_for_training`) plus 32 resource servers: `math_with_judge`, `code_gen`, `workplace_assistant`, `mcqa`, `instruction_following`, `equivalence_llm_judge` (×3 variants), `calendar`, `genrm_compare`, `single_step_tool_use_with_argument_comparison` (×5 pivots), `reasoning_gym`, `terminus_judge`, `ns_tools`, `math_formal_lean`, `multichallenge`, `inverse_if`, `abstention`, `nvarc` (×2), `equivalence_rule`, `ether0`, `structured_outputs` (×2), `format_verification` (×2), `jailbreak_detection`, `indirect_prompt_injection`.

Below `config_paths` are **per-server override blocks** keyed by the config's top-level name (`math_with_judge:`, `code_gen:`, `abstention:`, …), each nesting `resources_servers: <name>: {...}`, plus judge model definitions (`nl2bash_judge_model`, `safety_judge_model`, `genrm_model`) with full `vllm_serve_kwargs` (TP/DP/memory), and `policy_model` / `policy_model_reasoning_off` (`_copy: policy_model` + `enable_thinking: false`) which set `num_workers` and `chat_template_kwargs`.

**The Ultra multi-env pattern, precisely:**
> **One** NeMo-RL environment (`nemo_gym`), **one** dataset (`NemoGymDataset` over a single pre-mixed jsonl), **one** task_name. All 30+ verifiers are multiplexed *inside Gym* by the per-row `agent_ref.name`. Mixture weights are baked into the jsonl by Gym's `ng_prepare_data` (row counts / `num_repeats`), not expressed in NeMo-RL config.

Scale (`student_rlvr1.yaml:19-22`, `ultra_launch.sh:501-570`): `cluster.num_nodes: 256`, `gpus_per_node: 4`, `segment_size: 16`; `NUM_TRAIN_NODES=64` + `NUM_GEN_NODES=172` + `NUM_GYM_NODES=20`. GRPO: `num_prompts_per_step: 512`, `num_generations_per_prompt: 16`, `async_grpo.enabled: true`, `in_flight_weight_updates: true`, `metric_name: "val:total_reward/mean"`.

Smaller reference points with identical shape: `examples/nemo_gym/nemotron-3.5-lightning/rlvr.yaml` (`cluster.num_nodes: 64`, `num_gpu_nodes: 2`) and `examples/nemo_gym/grpo_nanov3.yaml` (`data:295`, `env:312`, 7 config_paths, **no `num_gpu_nodes`** → fully colocated, best small-scale template).

## 5. Multi-dataset / multi-env blending in native GRPO

Yes — supported, two independent axes.

**Axis 1 — heterogeneous envs in one run.** `data.train` accepts a **list** (`nemo_rl/data/__init__.py:66`), each element with its own `env_name`, `processor`, `prompt_file`. `setup_response_data` builds `task_to_env[task_name] = envs[cfg["env_name"]]` per dataset (`data/utils.py:188-189`), and `calculate_rewards` fans out per task group (`rollouts.py:694-717`). So per-sample env routing **is** per-record, keyed by the `task_name` the dataset class stamps on every row. Datasets are concatenated into one stream by default (`data/utils.py:210-218`).

Example: `examples/configs/grpo_multiple_datasets.yaml:20-34` (note `_override_: true` on the `data:` block so the list replaces, not merges with, the inherited dict).

**Axis 2 — deterministic per-dataset batch composition.** `data.use_multiple_dataloader: true` + `data.num_prompts_per_dataloader: N` + `data.custom_dataloader: <dotted.path.to.fn>` (`nemo_rl/data/__init__.py:60-64`). Builds one `StatefulDataLoader` per task and wraps them in `MultipleDataloaderWrapper` (`nemo_rl/data/dataloader.py:18-82`) — an **infinite** iterator whose sampling policy is your function `(data_iterators, dataloaders, **records) -> (BatchedDataDict, data_iterators)`; you must reset exhausted iterators yourself. `grpo.num_prompts_per_step` must be a multiple of `num_prompts_per_dataloader`. Reference impl `examples/custom_dataloader/custom_dataloader.py`; docs `docs/guides/grpo.md:188-250`.

Constraints: **not supported with async GRPO** (`examples/run_grpo.py:189-193`), and all envs in a batch must agree on scalar-vs-dict rewards (`rollouts.py:773-777`).

## 6. Recommended spec for alpha RLVR

alpha's constraints (from `examples/configs/alpha/README.md`): Backend.AI, no Slurm, 2 nodes, node0 = train+rollout colocated, node1 = frozen serving, **no IB between nodes (~9.1 Gbit/s)**, Megatron-Core training + vLLM plugin rollout.

**Phase 1 — native `math` env, no Gym.** Lowest infra cost, zero HTTP hops, works fully colocated on node0:

```yaml
data:
  max_input_seq_length: ${policy.max_total_sequence_length}
  shuffle: true
  num_workers: 1
  use_multiple_dataloader: false
  train:
    - {data_path: /data/alpha/math_train.jsonl, input_key: problem, output_key: answer,
       env_name: math, processor: math_hf_data_processor}
    - {data_path: /data/alpha/code_train.jsonl, env_name: code_gen,
       processor: alpha_code_processor}     # register_processor() + register_env() in launcher
  validation:
    - {data_path: /data/alpha/math_val.jsonl}
  default:
    dataset_name: ResponseDataset
    prompt_file: examples/prompts/cot.txt
    system_prompt_file: null
env:
  math:    {num_workers: 8, math_verify_impl: hf_math_verify}
  code_gen: {num_workers: 4, ...}          # your registered env's config
```
Run with `examples/run_grpo.py`. Reminder: `data:` needs `_override_: true` if you inherit from `grpo_math_1B.yaml` and want the list form.

**Phase 2 — Gym, colocated.** Copy `examples/nemo_gym/grpo_nanov3.yaml`'s `data:`/`env:` blocks verbatim (it has no `num_gpu_nodes`, so nothing lands off-node), trim `config_paths` to verifier-only servers that need **no judge model** (`instruction_following`, `mcqa`, `structured_outputs`, `code_gen`, `reasoning_gym`, and `math_with_judge` with `should_use_judge: false` + `judge_model_server.name: policy_model` — exactly the nanov3 pattern at `grpo_nanov3.yaml:337-341`). This keeps every reward on node0 and avoids the inter-node bandwidth cliff. Pre-bake venvs with `examples/nemo_gym/prefetch_venvs.py` into the container. Switch entrypoint to `examples/nemo_gym/run_grpo_nemo_gym.py`, and set `async_grpo.enabled: true` (Gym requires an async generation backend).

Only introduce `num_gpu_nodes: 1` (node1) once you actually need an LLM-judge/GenRM server — that traffic is the one thing that would cross the slow link.
