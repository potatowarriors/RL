# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collection-loop level tests for DAPO dynamic sampling (no Ray cluster).

Runs the real ``_collection_loop`` / target selection / release logic in a
thread with ``ray.get`` patched to identity and a fake replay buffer.
Regression for the 2026-10-20 teacher_codemath_s3 deadlock and for the
lost-wakeup race found by the independent review of 57f86ee1e.
"""

import threading
import time
import unittest.mock as mock

import nemo_rl.algorithms.async_utils.trajectory_collector as tc
from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector
from nemo_rl.algorithms.grpo import AsyncGRPOConfig, GRPOConfig, MasterConfig


class _FakeBatch:
    def __init__(self, items):
        self.items = items

    @property
    def size(self):
        return len(self.items)

    def slice(self, start, end):
        return _FakeBatch(self.items[start:end])


def _cycling(n_rows):
    i = 0
    while True:
        yield _FakeBatch(list(range(i, i + n_rows)))
        i += n_rows


def _collector(monkeypatch, *, drop=True, n=2, k=1, age=1):
    collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    master_config = MasterConfig.model_construct(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=n,
            num_generations_per_prompt=2,
            max_rollout_turns=1,
            async_grpo=AsyncGRPOConfig.model_construct(
                max_trajectory_age_steps=age,
                max_generation_failures=0,
                rollout_overprovision=k,
                drop_zero_std_groups=drop,
            ),
        ),
        policy={
            "max_total_sequence_length": 512,
            "make_sequence_length_divisible_by": 1,
            "generation": {"backend": "vllm", "vllm_cfg": {"async_engine": False}},
        },
        env={"should_use_nemo_gym": False},
        logger={"wandb_enabled": False, "wandb": {}},
    )
    state = {"last": -1, "counts": {}}
    buf = mock.MagicMock()
    buf.get_last_target_weight_already_generated.remote = lambda: state["last"]
    buf.get_trajectories_needed.remote = lambda t, num, a: max(
        0, num - state["counts"].get(t, 0)
    )
    monkeypatch.setattr(tc.ray, "get", lambda x, *a, **kw: x)
    collector = collector_cls(
        policy_generation=mock.MagicMock(),
        tokenizer=mock.MagicMock(),
        task_to_env={},
        master_config=master_config,
        replay_buffer=buf,
        start_step=0,
        next_nemo_gym_task_index=0,
    )
    for attr in (
        "_manual_pause_cleared",
        "_refit_pause_cleared",
        "_generation_limit_cleared",
    ):
        ev = threading.Event()
        ev.set()
        setattr(collector, attr, ev)
    collector.running = True
    launched = []

    # Reserve through the real _get_next_target_for_generation and pretend the
    # whole batch was launched; targets stay reserved until the test releases
    # them (that is the in-flight batch worker).
    def fake_process(batch):
        target = collector._get_next_target_for_generation(
            collector.current_weight_version
        )
        if target is None:
            return 0
        launched.append(target)
        with collector._generation_check_lock:
            collector._launched_targets.add(target)
        return batch.size

    collector._process_batch = fake_process
    collector.dataloader = _cycling(n + k)
    return collector, state, launched


def _wait(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _stop(collector, thread):
    collector.running = False
    collector._generation_limit_cleared.set()
    collector._manual_pause_cleared.set()
    thread.join(timeout=3)


def test_release_after_park_wakes_loop_and_gap_fills(monkeypatch):
    """Both targets in flight -> loop parks; a release after the park wakes it."""
    collector, state, launched = _collector(monkeypatch)
    th = threading.Thread(target=collector._collection_loop, daemon=True)
    th.start()
    assert _wait(
        lambda: launched == [0, 1] and not collector._generation_limit_cleared.is_set()
    )
    state["counts"][0] = 1  # worker for target 0 finished short (1 of 2 kept)
    collector._release_target(0)
    assert _wait(lambda: launched == [0, 1, 0]), launched  # gap-fill launched
    _stop(collector, th)


def test_release_between_pause_check_and_wait_is_not_lost(monkeypatch):
    """Race: the last in-flight worker releases its short target after the
    loop decided to pause but before it waits. The loop must not park."""
    collector, state, launched = _collector(monkeypatch)
    real_should_pause = collector._should_pause_for_generation_limits
    fired = []

    def racy_should_pause():
        decision = real_should_pause()
        if decision and not fired:
            fired.append(True)
            state["counts"][0] = 1
            collector._release_target(0)  # lands inside the window
        return decision

    collector._should_pause_for_generation_limits = racy_should_pause
    th = threading.Thread(target=collector._collection_loop, daemon=True)
    th.start()
    assert _wait(lambda: bool(fired))
    assert _wait(lambda: launched == [0, 1, 0]), launched  # gap-fill, no deadlock
    _stop(collector, th)


def test_gap_fill_is_sized_against_trainer_need_and_keep_rate(monkeypatch):
    collector, state, launched = _collector(monkeypatch, n=256, k=250)
    # first launch for a target: over-provisioned shortfall (N + k - buffered)
    assert collector._trajectories_to_launch(0, 1) == 506
    with collector._generation_check_lock:
        collector._launched_targets.add(0)
    # keep rate 40% observed -> gap-fill for 59 short launches ceil(59 / 0.4)
    with collector._dynamic_sampling_lock:
        collector._dynamic_sampling_counts.update({"kept": 40, "all_wrong": 60})
    state["counts"][0] = 197
    assert collector._trajectories_to_launch(0, 1) == 148
    state["counts"][0] = 256  # trainer need met -> nothing more
    assert collector._trajectories_to_launch(0, 1) == 0
    # keep rate clamped at 0.2 and launch capped at N + k
    with collector._dynamic_sampling_lock:
        collector._dynamic_sampling_counts.clear()
        collector._dynamic_sampling_counts.update({"kept": 1, "all_wrong": 99})
    state["counts"][0] = 0
    assert collector._trajectories_to_launch(0, 1) == 506


def test_without_dynamic_sampling_sizing_is_unchanged(monkeypatch):
    collector, state, launched = _collector(monkeypatch, drop=False, n=256, k=250)
    with collector._generation_check_lock:
        collector._launched_targets.add(0)
    state["counts"][0] = 197
    assert collector._trajectories_to_launch(0, 1) == 309  # N + k - buffered


def test_set_weight_version_prunes_launched_targets(monkeypatch):
    collector, state, launched = _collector(monkeypatch)
    with collector._generation_check_lock:
        collector._launched_targets.update({0, 1, 2})
    collector.set_weight_version(2)
    assert collector._launched_targets == {2}
