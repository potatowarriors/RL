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
"""Rollout over-provisioning for async GRPO.

With ``grpo.async_grpo.rollout_overprovision = k`` the collector launches
``num_prompts_per_step + k`` prompt groups per target step, training takes the
first ``num_prompts_per_step`` ready groups, and the leftovers are carried to the
next step. These tests cover the replay-buffer side (pure ``ReplayBufferImpl``,
no Ray) and the collector loop keeping the unused tail of a dataloader batch.
"""

import threading
import unittest.mock as mock

from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector
from nemo_rl.algorithms.async_utils.replay_buffer import ReplayBufferImpl
from nemo_rl.algorithms.grpo import AsyncGRPOConfig, GRPOConfig, MasterConfig


def _group(name: str) -> dict:
    return {"batch": {"data": name}, "rollout_metrics": {"reward": 1.0}}


def _names(sample_result: dict) -> list[str]:
    return [t["batch"]["data"] for t in sample_result["trajectories"]]


class TestReplayBufferCarryover:
    def test_overprovisioned_step_trains_on_first_ready_groups(self):
        """N=2 consumed, 3 launched: the third group must not block the step."""
        buffer = ReplayBufferImpl(max_size=10, allow_carryover=True)
        buffer.add(_group("a"), weight_version=0, target_weight_version=1)
        buffer.add(_group("b"), weight_version=0, target_weight_version=1)
        # third (over-provisioned) group is still in flight — not added yet

        result = buffer.sample(
            num_prompt_groups=2, current_weight_version=1, max_age_steps=2
        )

        assert result is not None
        assert _names(result) == ["a", "b"]
        assert buffer.size() == 0
        assert buffer.get_last_target_weight_already_generated() == 1

    def test_leftover_group_is_carried_to_next_step_not_dropped(self):
        buffer = ReplayBufferImpl(max_size=10, allow_carryover=True)
        for name in ("a", "b", "c"):
            buffer.add(_group(name), weight_version=0, target_weight_version=1)

        first = buffer.sample(
            num_prompt_groups=2, current_weight_version=1, max_age_steps=2
        )
        assert _names(first) == ["a", "b"]
        # "c" stays and is re-targeted to step 2
        assert buffer.size() == 1
        assert buffer.target_weight_versions == [2]
        assert buffer.get_trajectories_needed(2, num_prompts_per_step=3) == 2

        buffer.add(_group("d"), weight_version=1, target_weight_version=2)
        second = buffer.sample(
            num_prompt_groups=2, current_weight_version=2, max_age_steps=2
        )
        assert second is not None
        # carried group first (it is now intended for step 2), then the new one
        assert _names(second) == ["c", "d"]
        assert buffer.size() == 0

    def test_late_straggler_with_old_target_is_used_by_a_later_step(self):
        """A group that arrives after its step was consumed keeps its old target
        label; carry-over mode still uses it (intended groups first)."""
        buffer = ReplayBufferImpl(max_size=10, allow_carryover=True)
        buffer.add(_group("a"), weight_version=0, target_weight_version=1)
        buffer.add(_group("b"), weight_version=0, target_weight_version=1)
        assert (
            buffer.sample(
                num_prompt_groups=2, current_weight_version=1, max_age_steps=2
            )
            is not None
        )

        # straggler for step 1 lands late, plus one group intended for step 2
        buffer.add(_group("late"), weight_version=0, target_weight_version=1)
        buffer.add(_group("x"), weight_version=1, target_weight_version=2)

        result = buffer.sample(
            num_prompt_groups=2, current_weight_version=2, max_age_steps=2
        )
        assert result is not None
        assert _names(result) == ["x", "late"]

    def test_stale_straggler_is_evicted_by_age(self):
        """A straggler older than max_age_steps is dropped, not trained on."""
        buffer = ReplayBufferImpl(max_size=10, allow_carryover=True)
        buffer.add(_group("stale"), weight_version=0, target_weight_version=1)
        buffer.add(_group("fresh1"), weight_version=3, target_weight_version=4)
        buffer.add(_group("fresh2"), weight_version=3, target_weight_version=4)

        result = buffer.sample(
            num_prompt_groups=2, current_weight_version=4, max_age_steps=2
        )
        assert result is not None
        assert _names(result) == ["fresh1", "fresh2"]
        assert buffer.size() == 0  # "stale" evicted (version 0 < 4 - 2)

    def test_still_stalls_when_not_enough_groups(self):
        buffer = ReplayBufferImpl(max_size=10, allow_carryover=True)
        buffer.add(_group("a"), weight_version=0, target_weight_version=1)
        assert (
            buffer.sample(
                num_prompt_groups=2, current_weight_version=1, max_age_steps=2
            )
            is None
        )
        assert buffer.get_last_target_weight_already_generated() == -1

    def test_default_mode_keeps_exact_count_semantics(self):
        """allow_carryover=False: leftovers keep their label and older-target
        groups are never used for a later step (upstream behaviour)."""
        buffer = ReplayBufferImpl(max_size=10)
        for name in ("a", "b", "c"):
            buffer.add(_group(name), weight_version=0, target_weight_version=1)
        assert (
            buffer.sample(
                num_prompt_groups=2, current_weight_version=1, max_age_steps=2
            )
            is not None
        )
        assert buffer.target_weight_versions == [1]
        buffer.add(_group("d"), weight_version=1, target_weight_version=2)
        assert (
            buffer.sample(
                num_prompt_groups=2, current_weight_version=2, max_age_steps=2
            )
            is None
        )


class _FakeBatch:
    """Minimal stand-in for BatchedDataDict: only ``size`` and ``slice``."""

    def __init__(self, items: list[int]):
        self.items = items

    @property
    def size(self) -> int:
        return len(self.items)

    def slice(self, start: int, end: int) -> "_FakeBatch":
        return _FakeBatch(self.items[start:end])


def _local_collector(rollout_overprovision: int):
    collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    master_config = MasterConfig.model_construct(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=3,
            max_rollout_turns=1,
            async_grpo=AsyncGRPOConfig.model_construct(
                max_trajectory_age_steps=2,
                max_generation_failures=0,
                rollout_overprovision=rollout_overprovision,
            ),
        ),
        policy={
            "max_total_sequence_length": 512,
            "make_sequence_length_divisible_by": 1,
            "generation": {"backend": "vllm", "vllm_cfg": {"async_engine": False}},
        },
        env={"should_use_nemo_gym": False},
        logger={
            "wandb_enabled": False,
            "wandb": {"log_nemo_gym_full_result_tables": False},
        },
    )
    collector = collector_cls(
        policy_generation=mock.MagicMock(),
        tokenizer=mock.MagicMock(),
        task_to_env={},
        master_config=master_config,
        replay_buffer=mock.MagicMock(),
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
    collector._should_pause_for_generation_limits = lambda: False
    collector.running = True
    return collector


class TestCollectorOverprovision:
    def test_prompt_groups_per_target_adds_overprovision(self):
        assert _local_collector(0)._prompt_groups_per_target() == 2
        assert _local_collector(8)._prompt_groups_per_target() == 10

    def test_collection_loop_keeps_unused_batch_tail(self, monkeypatch):
        """With over-provisioning, a partially used batch is offered again
        (its tail first) instead of being dropped."""
        collector = _local_collector(rollout_overprovision=1)
        monkeypatch.setattr(
            "nemo_rl.algorithms.async_utils.trajectory_collector.time.sleep",
            lambda _s: None,
        )
        launched: list[list[int]] = []
        consume_plan = iter([2, 0, 1, 3])  # 2 of 3, then retry, then 1, then all

        def fake_process(batch):
            n = next(consume_plan)
            launched.append(batch.items[:n])
            return n

        collector._process_batch = fake_process
        collector.dataloader = [_FakeBatch([0, 1, 2]), _FakeBatch([3, 4, 5])]

        collector._collection_loop()

        # batch 1: prompts 0,1 launched; tail [2] retried (0), then launched; batch 2 whole
        assert launched == [[0, 1], [], [2], [3, 4, 5]]
        assert collector.data_exhausted is True
        assert collector.collection_failed is False

    def test_collection_loop_appends_next_batch_when_target_needs_more(self):
        """A negative return means "the target needs more rows than this batch":
        the loop pulls the next batch, appends it and retries, so N+k groups are
        launched together."""
        collector = _local_collector(rollout_overprovision=2)
        collector._concat_batches = lambda a, b: _FakeBatch(a.items + b.items)
        launched: list[list[int]] = []
        # 1st call: batch [0,1,2] too small for 5 → ask for more; 2nd: [0..5] → launch 5;
        # 3rd: tail [5] + nothing pending... loop offers [5] → launch 1; 4th: [6,7,8] → launch 3
        consume_plan = iter([-5, 5, 1, 3])

        def fake_process(batch):
            n = next(consume_plan)
            if n > 0:
                launched.append(batch.items[:n])
            return n

        collector._process_batch = fake_process
        collector.dataloader = [
            _FakeBatch([0, 1, 2]),
            _FakeBatch([3, 4, 5]),
            _FakeBatch([6, 7, 8]),
        ]

        collector._collection_loop()

        assert launched == [[0, 1, 2, 3, 4], [5], [6, 7, 8]]
        assert collector.data_exhausted is True
        assert collector._allow_partial_launch is False

    def test_collection_loop_partial_launch_at_dataloader_end(self):
        """When no more batches exist, the target is launched with what is left."""
        collector = _local_collector(rollout_overprovision=2)
        launched: list[list[int]] = []
        seen_partial_flag: list[bool] = []

        def fake_process(batch):
            seen_partial_flag.append(collector._allow_partial_launch)
            if not collector._allow_partial_launch:
                return -5  # needs 5, batch has 3, no more data
            launched.append(batch.items)
            return batch.size

        collector._process_batch = fake_process
        collector.dataloader = [_FakeBatch([0, 1, 2])]

        collector._collection_loop()

        assert launched == [[0, 1, 2]]
        assert seen_partial_flag == [False, True]
        assert collector.data_exhausted is True

    def test_concat_batches_merges_rows_and_recomputes_batch_max_length(self):
        import torch

        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        a = BatchedDataDict(
            message_log=[[{"role": "user", "content": "a"}]],
            length=torch.tensor([3]),
            loss_multiplier=torch.tensor([1.0]),
            extra_env_info=[{"k": 1}],
            task_name=["t"],
            idx=[0],
            batch_max_length=torch.tensor([3]),
            stop_strings=[None],
        )
        b = BatchedDataDict(
            message_log=[
                [{"role": "user", "content": "b"}],
                [{"role": "user", "content": "c"}],
            ],
            length=torch.tensor([5, 2]),
            loss_multiplier=torch.tensor([1.0, 1.0]),
            extra_env_info=[{"k": 2}, {"k": 3}],
            task_name=["t", "t"],
            idx=[1, 2],
            batch_max_length=torch.tensor([5, 5]),
            stop_strings=[None, None],
        )
        collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
        merged = collector_cls._concat_batches(a, b)
        assert merged.size == 3
        assert merged["idx"] == [0, 1, 2]
        assert merged["length"].tolist() == [3, 5, 2]
        assert merged["batch_max_length"].tolist() == [5, 5, 5]
        assert merged["extra_env_info"] == [{"k": 1}, {"k": 2}, {"k": 3}]
        tail = merged.slice(2, 3)
        assert tail.size == 1 and tail["idx"] == [2]

    def test_collection_loop_default_mode_drops_tail_as_before(self):
        collector = _local_collector(rollout_overprovision=0)
        launched: list[list[int]] = []

        def fake_process(batch):
            launched.append(batch.items[:2])
            return 2

        collector._process_batch = fake_process
        collector.dataloader = [_FakeBatch([0, 1, 2]), _FakeBatch([3, 4, 5])]

        collector._collection_loop()

        assert launched == [[0, 1], [3, 4]]
        assert collector.data_exhausted is True
